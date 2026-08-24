/* Forge F1 Phase 4 — admission scheduler unit tests.
 *
 * Mock store with artificial load latency + failure injection exercises:
 * hit path, miss->load->publish, coalescing window (concurrent publisher),
 * fail-fast BUSY, batch dedupe + orchestration + bounded parallelism
 * (timing-bounded), prefetch ordering determinism, default-OFF config.
 */

#include "../admission.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_fail = 0;
#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_fail++;                                                     \
        }                                                                 \
    } while (0)

#define MOCK_LAYERS 4
#define MOCK_EXPERTS 16
#define SLOT_BYTES 64

typedef struct {
    int state[MOCK_LAYERS * MOCK_EXPERTS]; /* 0 empty, 1 reserved, 2 resident */
    int reserved_by[MOCK_LAYERS * MOCK_EXPERTS]; /* slot idx or -1 */
    int resident_slot[MOCK_LAYERS * MOCK_EXPERTS];
    int slot_state[MOCK_LAYERS * MOCK_EXPERTS]; /* per-slot mirror */
    int lease_count[MOCK_LAYERS * MOCK_EXPERTS];
    int load_sleep_ms;
    int fail_loads; /* inject load_fn failure */
    uint64_t physical_loads;
    pthread_mutex_t mx;
} MockState;

static void mock_sleep(int ms) {
    struct timespec ts = {0, (long)ms * 1000000L};
    nanosleep(&ts, NULL);
}

static int mock_lookup(ColiExpertStore *store, ColiExpertKey key,
                       ColiExpertView *view) {
    MockState *s = (MockState *)store->state;
    memset(view, 0, sizeof(*view));
    if (key.layer < 0 || key.layer >= MOCK_LAYERS ||
        key.expert < 0 || key.expert >= MOCK_EXPERTS)
        return -1;
    int ei = key.layer * MOCK_EXPERTS + key.expert;
    pthread_mutex_lock(&s->mx);
    int slot = s->resident_slot[ei];
    if (slot < 0 || s->slot_state[slot] != 2) {
        pthread_mutex_unlock(&s->mx);
        return -1;
    }
    static unsigned char blob[MOCK_LAYERS * MOCK_EXPERTS][SLOT_BYTES];
    view->key = key;
    view->gate.data = blob[ei];
    view->gate.data_bytes = SLOT_BYTES;
    view->lease = (void *)(long)(slot + 1);
    s->lease_count[slot]++;
    pthread_mutex_unlock(&s->mx);
    return 0;
}

static void mock_release(ColiExpertStore *store, ColiExpertView *view) {
    MockState *s = (MockState *)store->state;
    if (!view || !view->lease) return;
    int slot = (int)(long)view->lease - 1;
    pthread_mutex_lock(&s->mx);
    if (s->lease_count[slot] > 0) s->lease_count[slot]--;
    pthread_mutex_unlock(&s->mx);
    memset(view, 0, sizeof(*view));
}

static int mock_reserve(ColiExpertStore *store, const ColiExpertCoreKey *key,
                        ColiExpertReservation *out) {
    MockState *s = (MockState *)store->state;
    if (!key || !out) return COLI_EXPERT_ERR_INVALID;
    memset(out, 0, sizeof(*out));
    if (key->layer < 0 || key->layer >= MOCK_LAYERS ||
        key->index < 0 || key->index >= MOCK_EXPERTS)
        return COLI_EXPERT_ERR_INVALID;
    int ei = key->layer * MOCK_EXPERTS + key->index;
    pthread_mutex_lock(&s->mx);
    if (s->state[ei] != 0) {
        pthread_mutex_unlock(&s->mx);
        return COLI_EXPERT_ERR_BUSY;
    }
    int slot = -1;
    for (int i = 0; i < MOCK_LAYERS * MOCK_EXPERTS; i++)
        if (s->slot_state[i] == 0) { slot = i; break; }
    if (slot < 0) {
        pthread_mutex_unlock(&s->mx);
        return COLI_EXPERT_ERR_NOCAP;
    }
    s->state[ei] = 1;
    s->reserved_by[ei] = slot;
    s->slot_state[slot] = 1;
    pthread_mutex_unlock(&s->mx);

    static ColiExpertSegment segs[MOCK_LAYERS * MOCK_EXPERTS][2];
    static unsigned char bufs[MOCK_LAYERS * MOCK_EXPERTS][SLOT_BYTES];
    segs[ei][0].data = bufs[ei];
    segs[ei][0].bytes = SLOT_BYTES;
    out->key = *key;
    out->segment_count = 1;
    out->segments = segs[ei];
    out->impl = (void *)(long)(ei + 1);
    return COLI_EXPERT_OK;
}

static int mock_publish(ColiExpertStore *store, ColiExpertReservation *res,
                        ColiExpertView *out_view) {
    MockState *s = (MockState *)store->state;
    if (!res || !res->impl) return COLI_EXPERT_ERR_INVALID;
    int ei = (int)(long)res->impl - 1;
    int slot = s->reserved_by[ei];
    pthread_mutex_lock(&s->mx);
    if (slot < 0 || s->slot_state[slot] != 1) {
        pthread_mutex_unlock(&s->mx);
        return COLI_EXPERT_ERR_INVALID;
    }
    s->state[ei] = 2;
    s->reserved_by[ei] = -1;
    s->slot_state[slot] = 2;
    s->resident_slot[ei] = slot;
    s->physical_loads++;
    pthread_mutex_unlock(&s->mx);
    if (out_view) {
        ColiExpertKey k = {res->key.layer, res->key.index};
        mock_lookup(store, k, out_view);
    }
    res->segment_count = 0;
    res->segments = NULL;
    res->impl = NULL;
    return COLI_EXPERT_OK;
}

static void mock_abort(ColiExpertStore *store, ColiExpertReservation *res) {
    MockState *s = (MockState *)store->state;
    if (!res || !res->impl) return;
    int ei = (int)(long)res->impl - 1;
    pthread_mutex_lock(&s->mx);
    int slot = s->reserved_by[ei];
    s->state[ei] = 0;
    s->reserved_by[ei] = -1;
    if (slot >= 0) s->slot_state[slot] = 0;
    pthread_mutex_unlock(&s->mx);
    res->segment_count = 0;
    res->segments = NULL;
    res->impl = NULL;
}

static void mock_stats(const ColiExpertStore *store, ColiExpertStoreStats *st) {
    MockState *s = (MockState *)store->state;
    memset(st, 0, sizeof(*st));
    pthread_mutex_lock(&s->mx);
    st->physical_loads = s->physical_loads;
    pthread_mutex_unlock(&s->mx);
}

static void mock_destroy(ColiExpertStore *store) { (void)store; }

static int mock_fill(void *userdata, const ColiExpertCoreKey *key,
                     ColiExpertReservation *res) {
    MockState *s = (MockState *)userdata;
    if (s->fail_loads) return -1;
    if (s->load_sleep_ms) mock_sleep(s->load_sleep_ms);
    if (res->segments && res->segment_count > 0 && res->segments[0].data)
        memset(res->segments[0].data, 0x7e, res->segments[0].bytes);
    return 0;
}

static ColiExpertKey K(int l, int e) {
    ColiExpertKey k = {l, e};
    return k;
}

/* helper thread: sleeps 40ms then publishes a held reservation */
typedef struct {
    ColiExpertStore *st;
    ColiExpertReservation *r;
    int done;
} PubCtx;

static void *pub_thread(void *arg) {
    PubCtx *ctx = (PubCtx *)arg;
    mock_sleep(40);
    ColiExpertView tmp;
    mock_publish(ctx->st, ctx->r, &tmp);
    coli_expert_release(ctx->st, &tmp);
    ctx->done = 1;
    return NULL;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static const ColiExpertStoreOps g_mock_ops = {
    mock_lookup, mock_release, NULL, mock_stats, mock_destroy,
    mock_reserve, mock_publish, mock_abort, NULL, NULL, NULL,
};

int main(void) {
    MockState ms;
    memset(&ms, 0, sizeof(ms));
    memset(ms.reserved_by, 0xFF, sizeof(ms.reserved_by));
    for (int i = 0; i < MOCK_LAYERS * MOCK_EXPERTS; i++)
        ms.resident_slot[i] = -1; /* -1 = nothing resident (0 is a real slot!) */
    pthread_mutex_init(&ms.mx, NULL);
    ColiExpertStore store = {&g_mock_ops, &ms, NULL};

    /* default-OFF: zeroed config = serial + fail-fast */
    ColiAdmission *adm = coli_admission_new(&store, mock_fill, &ms, NULL);
    CHECK(adm != NULL);

    /* miss -> load -> publish -> hit */
    ColiExpertView v;
    ColiExpertKey k00 = K(0, 0);
    CHECK(coli_admission_acquire(adm, &k00, &v) == 0);
    CHECK(v.lease != NULL);
    CHECK(coli_admission_acquire(adm, &k00, &v) == 0); /* second = hit */
    coli_expert_release(&store, &v);

    /* fail-fast BUSY: hold identity manually, acquire must return BUSY */
    ColiExpertKey k11 = K(1, 1);
    ColiExpertCoreKey core = coli_expert_core_key(k11);
    ColiExpertReservation res;
    CHECK(coli_expert_reserve(&store, &core, &res) == COLI_EXPERT_OK);
    CHECK(coli_admission_acquire(adm, &k11, &v) == COLI_EXPERT_ERR_BUSY);
    /* coalescing window: publish from a helper thread after 40ms */
    PubCtx ctx = {&store, &res, 0};
    pthread_t pt;
    CHECK(pthread_create(&pt, NULL, pub_thread, &ctx) == 0);
    ColiAdmissionConfig cfg = {0};
    cfg.wait_busy_ms = 5;
    cfg.wait_busy_max_ms = 5000;
    ColiAdmission *adm2 = coli_admission_new(&store, mock_fill, &ms, &cfg);
    double t0 = now_ms();
    CHECK(coli_admission_acquire(adm2, &k11, &v) == 0);
    double waited = now_ms() - t0;
    CHECK(waited >= 30.0); /* actually waited for the publisher */
    CHECK(v.lease != NULL);
    pthread_join(pt, NULL);
    CHECK(ctx.done);
    coli_expert_release(&store, &v);
    coli_admission_free(adm2);

    /* batch: dedupe + mixed hit/miss + caller-order leases */
    {
        ColiExpertKey keys[6] = {K(2, 0), K(2, 1), K(2, 0), K(2, 2), K(2, 1), K(3, 3)};
        ColiExpertView views[6];
        int ok = coli_admission_acquire_batch(adm, keys, 6, views);
        CHECK(ok == 6);
        for (int i = 0; i < 6; i++) CHECK(views[i].lease != NULL);
        const ColiAdmissionStats *as = coli_admission_stats(adm);
        CHECK(as->batch_keys_in >= 6);
        CHECK(as->batch_dedupe_removed == 2);
        /* physical loads: unique missing = (2,1),(2,2),(3,3) — (2,0) hit */
        ColiExpertStoreStats ss;
        mock_stats(&store, &ss);
        CHECK(ss.physical_loads >= 3);
        for (int i = 0; i < 6; i++) coli_expert_release(&store, &views[i]);
    }

    /* bounded parallelism: 4 unique misses x 50ms loads */
    ms.load_sleep_ms = 50;
    {
        ColiExpertKey keys[4] = {K(0, 4), K(0, 5), K(0, 6), K(0, 7)};
        ColiExpertView views[4];
        ColiAdmissionConfig pcfg = {0};
        pcfg.max_parallel = 2;
        ColiAdmission *pa = coli_admission_new(&store, mock_fill, &ms, &pcfg);
        double t1 = now_ms();
        CHECK(coli_admission_acquire_batch(pa, keys, 4, views) == 4);
        double wall2 = now_ms() - t1;
        CHECK(wall2 >= 95.0 && wall2 < 190.0); /* 2 waves of 50ms */
        for (int i = 0; i < 4; i++) coli_expert_release(&store, &views[i]);
        coli_admission_free(pa);

        ColiAdmissionConfig pcfg4 = {0};
        pcfg4.max_parallel = 4;
        ColiExpertKey keys2[4] = {K(0, 8), K(0, 9), K(0, 10), K(0, 11)};
        ColiExpertView views2[4];
        ColiAdmission *pa4 = coli_admission_new(&store, mock_fill, &ms, &pcfg4);
        t1 = now_ms();
        CHECK(coli_admission_acquire_batch(pa4, keys2, 4, views2) == 4);
        double wall4 = now_ms() - t1;
        CHECK(wall4 < 95.0); /* one wave */
        for (int i = 0; i < 4; i++) coli_expert_release(&store, &views2[i]);
        coli_admission_free(pa4);
    }
    ms.load_sleep_ms = 0;

    /* prefetch ordering: deterministic sort + dedupe */
    {
        ColiExpertKey keys[7] = {K(3, 9), K(1, 2), K(3, 1), K(1, 2),
                                 K(0, 5), K(3, 9), K(0, 1)};
        size_t kept = coli_admission_order_prefetch(keys, 7);
        CHECK(kept == 5);
        CHECK(keys[0].layer == 0 && keys[0].expert == 1);
        CHECK(keys[1].layer == 0 && keys[1].expert == 5);
        CHECK(keys[2].layer == 1 && keys[2].expert == 2);
        CHECK(keys[3].layer == 3 && keys[3].expert == 1);
        CHECK(keys[4].layer == 3 && keys[4].expert == 9);
    }

    /* injected load failure publishes nothing */
    ms.fail_loads = 1;
    {
        ColiExpertCoreKey cf = coli_expert_core_key(K(1, 5));
        ColiExpertReservation rf;
        CHECK(coli_expert_reserve(&store, &cf, &rf) == COLI_EXPERT_OK);
        CHECK(mock_fill(&ms, &cf, &rf) != 0);
        coli_expert_abort(&store, &rf);
        CHECK(coli_expert_lookup(&store, K(1, 5), &v) != 0);
    }
    ms.fail_loads = 0;

    const ColiAdmissionStats *as = coli_admission_stats(adm);
    CHECK(as->acquires >= 2 && as->hits >= 1 && as->loads >= 1);
    coli_admission_free(adm);

    if (g_fail) {
        fprintf(stderr, "admission tests: %d FAILURES\n", g_fail);
        return 1;
    }
    puts("admission tests: ok");
    return 0;
}
