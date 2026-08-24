/* Forge F1 Lane A — durable liveness + lease-ownership tests (F1-LIVE-1).
 *
 * Real pread backend, REAL minimal safetensors fixtures, tiny pools:
 *   H1 ABORT->FREE:  pool fully RESERVED; a bounded waiter keeps retrying a
 *                    reserve; one owner aborts -> the freed slot becomes
 *                    claimable and the waiter completes. Counters balance.
 *   H2 PUBLISH->PROGRESS: pool fully RESERVED; an owner publishes -> the
 *                    waiter legally evicts the RESIDENT copy (never a
 *                    RESERVED buffer — the other owner can still publish).
 * Both repeat 100 cheap iterations each (fresh store per iteration), plus a
 * real-backend acquire_batch lease-transfer check and the destroy-contract
 * fork probe (debug builds must abort on destroy-with-live-lease).
 */

#include "../admission.h"
#include "../expert_backend_pread.h"
#include "../expert_store_registry.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#define MKDIR(p) mkdir((p), 0755)
#endif

static int g_fail = 0;
#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_fail++;                                                       \
        }                                                                   \
    } while (0)

/* Registry link dependency: never reached from this test. */
int coli_v4_expert_store_open_planned(ColiV4Engine *engine,
                                      const ColiDeepSeekV4Config *config,
                                      const ColiDeepSeekV4ExpertStoreOptions *opts,
                                      ColiExpertStore **out,
                                      char *error, size_t error_size) {
    (void)engine; (void)config; (void)opts; (void)out;
    fprintf(stderr, "FAIL: auto backend dispatched inside liveness test\n");
    if (error && error_size) snprintf(error, error_size, "stub");
    return -99;
}

/* ---- minimal safetensors fixture: 1 layer x 4 experts, INT8 weights ---- */

#define FIX_LAYERS 1
#define FIX_EXPERTS 4
#define WEIGHT_BYTES 2048

static char g_dir[256];

static int write_fixtures(void) {
    snprintf(g_dir, sizeof(g_dir), "/tmp/kilo/forge_f1_liveness_%d",
             (int)getpid());
    if (MKDIR(g_dir) != 0) return -1;

    /* ONE shard, FOUR INT8 expert tensors: [w_l0_e0..w_l0_e3], zero-gap. */
    const char *names[FIX_EXPERTS];
    char namebuf[FIX_EXPERTS][32];
    int64_t offsets[2 * FIX_EXPERTS];
    char hdr[1024];
    size_t used = 0;
    used += (size_t)snprintf(hdr + used, sizeof(hdr) - used, "{");
    for (int e = 0; e < FIX_EXPERTS; e++) {
        snprintf(namebuf[e], sizeof(namebuf[e]), "w_l0_e%d", e);
        names[e] = namebuf[e];
        offsets[2 * e] = (int64_t)e * WEIGHT_BYTES;
        offsets[2 * e + 1] = offsets[2 * e] + WEIGHT_BYTES;
        used += (size_t)snprintf(hdr + used, sizeof(hdr) - used,
                                 "\"%s\":{\"dtype\":\"U8\",\"shape\":[%lld],"
                                 "\"data_offsets\":[%lld,%lld]}%s",
                                 names[e], (long long)WEIGHT_BYTES,
                                 (long long)offsets[2 * e],
                                 (long long)offsets[2 * e + 1],
                                 e + 1 < FIX_EXPERTS ? "," : "");
    }
    used += (size_t)snprintf(hdr + used, sizeof(hdr) - used, "}");
    size_t hlen = used;
    while (hlen % 8 != 0 && hlen < sizeof(hdr) - 1) hdr[hlen++] = ' ';
    hdr[hlen] = '\0';

    unsigned char blob[FIX_EXPERTS * WEIGHT_BYTES];
    for (int e = 0; e < FIX_EXPERTS; e++)
        for (int i = 0; i < WEIGHT_BYTES; i++)
            blob[(size_t)e * WEIGHT_BYTES + i] =
                (unsigned char)(e * 131u + (unsigned)i);

    char path[512];
    snprintf(path, sizeof(path), "%s/w.safetensors", g_dir);
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint64_t hl = (uint64_t)hlen;
    fwrite(&hl, sizeof(hl), 1, f);
    fwrite(hdr, 1, hlen, f);
    fwrite(blob, 1, sizeof(blob), f);
    fclose(f);
    return 0;
}

static ColiExpertStore *open_tiny(int slots_per_layer, char *err,
                                  size_t errsz) {
    ColiExpertStoreDescriptor d;
    memset(&d, 0, sizeof(d));
    d.n_layers = FIX_LAYERS;
    d.n_experts = FIX_EXPERTS;
    d.storage_path = g_dir;
    d.weights_name_template = "w_l%d_e%d";
    d.slots_per_layer = slots_per_layer;
    ColiExpertStore *st = NULL;
    if (coli_expert_backend_pread_open(&d, &st, err, errsz) != 0) return NULL;
    return st;
}

static ColiExpertCoreKey core_key(int layer, int index) {
    ColiExpertCoreKey k = {layer, COLI_EXPERT_ROLE_EXPERT, index};
    return k;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* Fill both tiny-pool slots with live reservations (no I/O involved:
 * segments are caller-writable scratch). Pool ends fully RESERVED. */
static void fill_pool(ColiExpertStore *st, ColiExpertReservation *ra,
                      ColiExpertReservation *rb) {
    CHECK(coli_expert_reserve(st, &(ColiExpertCoreKey){0, COLI_EXPERT_ROLE_EXPERT, 0}, ra) ==
          COLI_EXPERT_OK);
    CHECK(coli_expert_reserve(st, &(ColiExpertCoreKey){0, COLI_EXPERT_ROLE_EXPERT, 1}, rb) ==
          COLI_EXPERT_OK);
}

/* Bounded waiter: retries reserve() for expert 2 on BUSY/SATURATED until a
 * deadline. Records whether it observed the transient saturation result.
 * done/saturated_seen are atomic: the main thread polls them while the
 * waiter runs (pthread_join orders everything after completion). */
typedef struct {
    ColiExpertStore *st;
    double deadline_ms;
    _Atomic int done;    /* 1 = reserved, -1 = unexpected error */
    _Atomic int saturated_seen;
    ColiExpertReservation res;
} Waiter;

static void *waiter_main(void *arg) {
    Waiter *w = (Waiter *)arg;
    ColiExpertCoreKey ck = core_key(0, 2);
    while (now_ms() < w->deadline_ms) {
        int rc = coli_expert_reserve(w->st, &ck, &w->res);
        if (rc == COLI_EXPERT_OK) {
            atomic_store(&w->done, 1);
            return NULL;
        }
        if (rc == COLI_EXPERT_ERR_SATURATED) {
            atomic_store(&w->saturated_seen, 1);
        } else if (rc != COLI_EXPERT_ERR_BUSY) {
            atomic_store(&w->done, -1);
            return NULL;
        }
        usleep(200);
    }
    return NULL;
}

/* Wait bounded for the waiter thread; returns 0 when it finished in time. */
static int wait_bounded(pthread_t t, Waiter *w, double budget_ms) {
    double end = now_ms() + budget_ms;
    while (atomic_load(&w->done) == 0 && now_ms() < end) usleep(1000);
    if (!w->done) {
        /* Never spin on a stalled reservation: detach and flag. The store
         * is intentionally left for process exit on this can't-happen
         * path so no lock the waiter holds is touched again. */
        pthread_detach(t);
        return -1;
    }
    pthread_join(t, NULL);
    return 0;
}

static void h1_abort_frees_slot(int iter) {
    char err[256];
    ColiExpertStore *st = open_tiny(2, err, sizeof(err));
    if (!st) {
        CHECK(0);
        return;
    }
    ColiExpertReservation ra, rb;
    fill_pool(st, &ra, &rb);

    Waiter w;
    atomic_init(&w.done, 0);
    atomic_init(&w.saturated_seen, 0);
    w.st = st;
    w.deadline_ms = now_ms() + 5000;
    pthread_t t;
    CHECK(pthread_create(&t, NULL, waiter_main, &w) == 0);

    /* Give the waiter time to hit the all-RESERVED edge before draining. */
    usleep(20000);

    /* Owner A aborts -> its slot becomes PBS_FREE and MUST become
     * claimable by the next retry (the F1-LIVE-1 hang is a waiter that
     * never rescans FREE). */
    coli_expert_abort(st, &ra);

    if (wait_bounded(t, &w, 6000) != 0) {
        fprintf(stderr, "FAIL H1 iter %d: waiter stalled (liveness hang)\n",
                iter);
        g_fail++;
        return;
    }
    CHECK(w.done == 1);
    CHECK(w.saturated_seen == 1); /* classification: transient saturation */

    /* Counters balance: three aborts (owner A drained the pool edge, owner
     * B + waiter cleaned up), zero active, nothing published. */
    coli_expert_abort(st, &w.res);
    coli_expert_abort(st, &rb);
    ColiExpertStoreStats st_;
    st->ops->stats(st, &st_);
    CHECK(st_.aborts == 3);
    CHECK(st_.publishes == 0);
    CHECK(st_.reservations_active == 0);
    coli_expert_destroy(st);
}

static void h2_publish_permits_progress(int iter) {
    char err[256];
    ColiExpertStore *st = open_tiny(2, err, sizeof(err));
    if (!st) {
        CHECK(0);
        return;
    }
    ColiExpertReservation ra, rb;
    fill_pool(st, &ra, &rb);

    Waiter w;
    atomic_init(&w.done, 0);
    atomic_init(&w.saturated_seen, 0);
    w.st = st;
    w.deadline_ms = now_ms() + 5000;
    pthread_t t;
    CHECK(pthread_create(&t, NULL, waiter_main, &w) == 0);
    usleep(20000);

    /* Owner A publishes -> RESIDENT; the waiter must progress LEGALLY by
     * evicting that RESIDENT copy, never by stealing B's RESERVED buffers. */
    memset(ra.segments[0].data, 0x3c, ra.segments[0].bytes);
    CHECK(coli_expert_publish(st, &ra, NULL) == COLI_EXPERT_OK);

    if (wait_bounded(t, &w, 6000) != 0) {
        fprintf(stderr, "FAIL H2 iter %d: waiter stalled (liveness hang)\n",
                iter);
        g_fail++;
        return;
    }
    CHECK(w.done == 1);
    CHECK(w.saturated_seen == 1);

    ColiExpertKey ka = {0, 0};
    ColiExpertView v;
    CHECK(coli_expert_lookup(st, ka, &v) != 0); /* A's copy was evicted */

    /* B was RESERVED throughout: its owner must still be able to publish
     * (a stolen RESERVED slot would fail this with ERR_INVALID). */
    memset(rb.segments[0].data, 0x5b, rb.segments[0].bytes);
    CHECK(coli_expert_publish(st, &rb, NULL) == COLI_EXPERT_OK);

    coli_expert_abort(st, &w.res); /* waiter cleanup */
    ColiExpertStoreStats st_;
    st->ops->stats(st, &st_);
    CHECK(st_.publishes == 2);
    CHECK(st_.aborts == 1);
    CHECK(st_.reservations_active == 0);
    coli_expert_destroy(st);
}

/* ---- real-backend batch lease transfer ---------------------------------- */

static int real_load(void *userdata, const ColiExpertCoreKey *key,
                     ColiExpertReservation *res) {
    return coli_expert_backend_pread_load(userdata, key, res);
}

static int g_fail_index = -1; /* inject partial load failure */
static int failing_load(void *userdata, const ColiExpertCoreKey *key,
                        ColiExpertReservation *res) {
    if (key->index == g_fail_index) return -1;
    return real_load(userdata, key, res);
}

static void batch_lease_checks(void) {
    char err[256];
    /* unique keys + duplicate entries: every returned view leases exactly
     * once, job leases transfer (none leak inside the scheduler) */
    {
        ColiExpertStore *st = open_tiny(FIX_EXPERTS, err, sizeof(err));
        CHECK(st != NULL);
        ColiAdmission *adm =
            coli_admission_new(st, real_load, st, NULL); /* default OFF */
        CHECK(adm != NULL);
        ColiExpertKey keys[4] = {{0, 0}, {0, 1}, {0, 0}, {0, 1}};
        ColiExpertView views[4];
        memset(views, 0, sizeof(views));
        int ok = coli_admission_acquire_batch(adm, keys, 4, views);
        CHECK(ok == 4);
        for (int i = 0; i < 4; i++) CHECK(views[i].lease != NULL);
        ColiAdmissionStats as;
        memcpy(&as, coli_admission_stats(adm), sizeof(as));
        CHECK(as.loads == 2); /* exactly-once physical admission per key */
        CHECK(as.batch_dedupe_removed == 2);
        for (int i = 0; i < 4; i++) coli_expert_release(st, &views[i]);
        /* debug builds assert inside destroy if any lease leaked */
        coli_admission_free(adm);
        coli_expert_destroy(st);
    }
    /* mixed success/failure: failed unique job leaves its entries cleared,
     * successful ones lease normally, nothing leaks either way */
    {
        ColiExpertStore *st = open_tiny(FIX_EXPERTS, err, sizeof(err));
        CHECK(st != NULL);
        ColiAdmission *adm = coli_admission_new(st, failing_load, st, NULL);
        CHECK(adm != NULL);
        g_fail_index = 2;
        ColiExpertKey keys[4] = {{0, 2}, {0, 3}, {0, 2}, {0, 3}};
        ColiExpertView views[4];
        memset(views, 0, sizeof(views));
        int ok = coli_admission_acquire_batch(adm, keys, 4, views);
        CHECK(ok == 2);
        CHECK(views[0].lease == NULL && views[1].lease != NULL);
        CHECK(views[2].lease == NULL && views[3].lease != NULL);
        g_fail_index = -1;
        for (int i = 0; i < 4; i++) coli_expert_release(st, &views[i]);
        coli_admission_free(adm);
        coli_expert_destroy(st);
    }
}

/* ---- destroy contract: debug builds abort on live-lease destroy --------- */

static void destroy_contract_probe(void) {
#if !defined(_WIN32) && !defined(NDEBUG)
    fflush(stderr);
    pid_t pid = fork();
    if (pid == 0) {
        /* child: hold a lease across destroy -> debug build must assert
         * before freeing (abnormal exit expected, never a clean 0) */
        if (!freopen("/dev/null", "w", stderr)) { /* silence expected report */
        }
        char err[256];
        ColiExpertStore *st = open_tiny(2, err, sizeof(err));
        if (!st) _Exit(70);
        ColiExpertKey k = {0, 0};
        if (st->ops->prefetch(st, &k, 1) != 1) _Exit(71);
        ColiExpertView v;
        if (coli_expert_lookup(st, k, &v) != 0) _Exit(72);
        coli_expert_destroy(st); /* contract violation: v still leased */
        _Exit(0);                /* REACHED = contract not enforced */
    }
    int status = 0;
    CHECK(waitpid(pid, &status, 0) == pid);
    int clean_zero = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    CHECK(!clean_zero);
#else
    fprintf(stderr,
            "destroy contract probe: skipped (release/NDEBUG build — "
            "contract is a debug-build assert)\n");
#endif
}

int main(void) {
    if (write_fixtures() != 0) {
        fprintf(stderr, "fixture generation failed\n");
        return 2;
    }

    enum { ITERATIONS = 100 };
    for (int i = 0; i < ITERATIONS; i++) {
        h1_abort_frees_slot(i);
        if (g_fail) break;
    }
    for (int i = 0; i < ITERATIONS; i++) {
        h2_publish_permits_progress(i);
        if (g_fail) break;
    }

    batch_lease_checks();
    destroy_contract_probe();

    /* unlink fixtures, best-effort */
    {
        char p[512];
        snprintf(p, sizeof(p), "%s/w.safetensors", g_dir);
        remove(p);
        remove(g_dir);
    }

    if (g_fail) {
        fprintf(stderr, "pread liveness tests: %d FAILURES\n", g_fail);
        return 1;
    }
    printf("pread liveness tests: ok (%dx H1, %dx H2)\n", ITERATIONS,
           ITERATIONS);
    return 0;
}
