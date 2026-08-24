/* Synthetic reservation-surface tests for ExpertStore v1.1 (forge F1 phase 1).
 * Mock stores only — no model bytes, no I/O. Exercises the full lifecycle:
 * reserve -> fill segments -> publish/abort, pin/unpin eviction immunity,
 * batch lookup accounting, and the widened stats invariants. */

#include "../expert_store.h"

#include <stdio.h>
#include <string.h>

#define SEG_WEIGHTS_BYTES 64
#define SEG_SCALES_BYTES 16

typedef struct {
    /* single-slot residency */
    int resident;
    ColiExpertCoreKey resident_key;
    unsigned char blob[SEG_WEIGHTS_BYTES];
    unsigned char scales[SEG_SCALES_BYTES];
    int lease_held;

    /* in-flight identity */
    int reserved;
    ColiExpertCoreKey reserved_key;
    unsigned char stage_w[SEG_WEIGHTS_BYTES];
    unsigned char stage_s[SEG_SCALES_BYTES];

    int pinned;
    int destroyed_with_lease;
    uint64_t evict_clock;
    ColiExpertStoreStats stats;
} ResState;

static void res_key_of(ColiExpertKey k, ColiExpertCoreKey *out) {
    out->layer = k.layer;
    out->role = COLI_EXPERT_ROLE_EXPERT;
    out->index = k.expert;
}

static int core_eq(const ColiExpertCoreKey *a, const ColiExpertCoreKey *b) {
    return a->layer == b->layer && a->role == b->role && a->index == b->index;
}

static int res_lookup(ColiExpertStore *store, ColiExpertKey key,
                      ColiExpertView *view) {
    ResState *s = (ResState *)store->state;
    ColiExpertCoreKey core;
    res_key_of(key, &core);
    if (!view) return COLI_EXPERT_ERR_INVALID;
    memset(view, 0, sizeof(*view));
    if (!s->resident || s->lease_held || !core_eq(&s->resident_key, &core))
        return -1;
    view->key = key;
    view->gate.data = s->blob;
    view->gate.data_bytes = SEG_WEIGHTS_BYTES;
    view->gate.scales = s->scales;
    view->gate.scale_bytes = SEG_SCALES_BYTES;
    view->lease = s;
    s->lease_held = 1;
    s->stats.requests++;
    s->stats.logical_requests++;
    s->stats.demand_requests++;
    s->stats.hits++;
    s->stats.demand_hits++;
    return 0;
}

static void res_release(ColiExpertStore *store, ColiExpertView *view) {
    ResState *s = (ResState *)store->state;
    if (view && view->lease == s) {
        s->lease_held = 0;
        view->lease = NULL;
    }
    if (view) memset(view, 0, sizeof(*view));
}

/* Eviction: pick the victim the way real LRU would; pinned keys are immune.
 * Returns 0 if eviction succeeded (resident replaced), -1 if blocked. */
static int res_evict(ResState *s, const ColiExpertCoreKey *incoming) {
    (void)incoming;
    if (!s->resident) return 0;
    if (s->pinned && core_eq(&s->resident_key, &s->reserved_key)) {
        /* pinned resident is immune */
        return -1;
    }
    if (s->pinned && s->resident_key.index == 77) return -1; /* hot slot */
    if (s->lease_held) return -1; /* leased slots are not victims */
    s->resident = 0;
    s->evict_clock++;
    return 0;
}

static int res_reserve(ColiExpertStore *store, const ColiExpertCoreKey *key,
                       ColiExpertReservation *out) {
    ResState *s = (ResState *)store->state;
    if (!key || !out) return COLI_EXPERT_ERR_INVALID;
    memset(out, 0, sizeof(*out));
    if (s->reserved) return COLI_EXPERT_ERR_BUSY;
    if (!s->resident && 0) return COLI_EXPERT_ERR_NOCAP; /* capacity unused here */
    s->reserved = 1;
    s->reserved_key = *key;
    s->stats.reservations_active++;
    out->key = *key;
    out->segment_count = 2;
    out->segments = (ColiExpertSegment *)store->gpu; /* borrowed scratch table */
    out->impl = s;
    return COLI_EXPERT_OK;
}

/* The store hands segment pointers at reserve time through a side channel:
 * for this mock, reserve fills them via impl. We re-purpose publish to copy
 * staging into residency. */
static int res_publish(ColiExpertStore *store, ColiExpertReservation *res,
                       ColiExpertView *out_view) {
    ResState *s = (ResState *)store->state;
    if (!res || res->segment_count != 2 || !res->segments || res->impl != s)
        return COLI_EXPERT_ERR_INVALID;
    if (s->resident && s->lease_held) return COLI_EXPERT_ERR_BUSY;
    /* Copy from exactly what the loader wrote via the segment pointers. */
    if (res->segments[0].bytes > sizeof(s->blob)) return COLI_EXPERT_ERR_INVALID;
    if (res->segments[1].bytes > sizeof(s->scales)) return COLI_EXPERT_ERR_INVALID;
    memcpy(s->blob, res->segments[0].data, res->segments[0].bytes);
    memcpy(s->scales, res->segments[1].data, res->segments[1].bytes);
    s->resident = 1;
    s->resident_key = res->key;
    s->reserved = 0;
    s->stats.reservations_active--;
    s->stats.publishes++;
    s->stats.physical_loads++;
    s->stats.admitted_bytes += SEG_WEIGHTS_BYTES + SEG_SCALES_BYTES;
    s->stats.admitted_weight_bytes += SEG_WEIGHTS_BYTES;
    s->stats.admitted_scale_bytes += SEG_SCALES_BYTES;
    s->stats.admission_ms_total += 1.5;
    if (out_view) {
        memset(out_view, 0, sizeof(*out_view));
        out_view->key.layer = res->key.layer;
        out_view->key.expert = res->key.index;
        out_view->gate.data = s->blob;
        out_view->gate.data_bytes = SEG_WEIGHTS_BYTES;
        out_view->gate.scales = s->scales;
        out_view->gate.scale_bytes = SEG_SCALES_BYTES;
        out_view->lease = s;
        s->lease_held = 1;
    }
    res->segment_count = 0;
    res->segments = NULL;
    return COLI_EXPERT_OK;
}

static void res_abort(ColiExpertStore *store, ColiExpertReservation *res) {
    ResState *s = (ResState *)store->state;
    if (!res || res->impl != s) { if (res) memset(res, 0, sizeof(*res)); return; }
    s->reserved = 0;
    s->stats.reservations_active--;
    s->stats.aborts++;
    memset(s->stage_w, 0, sizeof(s->stage_w));
    memset(s->stage_s, 0, sizeof(s->stage_s));
    res->segment_count = 0;
    res->segments = NULL;
}

static int res_pin(ColiExpertStore *store, const ColiExpertCoreKey *key) {
    ResState *s = (ResState *)store->state;
    if (!key) return COLI_EXPERT_ERR_INVALID;
    if (!s->pinned) s->stats.pinned_count++;
    s->pinned = 1;
    s->reserved_key = *key; /* mock: remembers which key is hot */
    return COLI_EXPERT_OK;
}

static void res_unpin(ColiExpertStore *store, const ColiExpertCoreKey *key) {
    ResState *s = (ResState *)store->state;
    (void)key;
    if (s->pinned && s->stats.pinned_count) s->stats.pinned_count--;
    s->pinned = 0;
}

static int res_lookup_batch(ColiExpertStore *store, const ColiExpertKey *keys,
                            size_t count, ColiExpertView *views) {
    int ok = 0;
    for (size_t i = 0; i < count; i++)
        if (res_lookup(store, keys[i], &views[i]) == 0) ok++;
    return ok;
}

static void res_stats(const ColiExpertStore *store,
                      ColiExpertStoreStats *stats) {
    *stats = ((const ResState *)store->state)->stats;
}

static void res_destroy(ColiExpertStore *store) {
    ResState *s = (ResState *)store->state;
    if (s->lease_held) s->destroyed_with_lease = 1;
}

static int view_is_cleared(const ColiExpertView *view) {
    static const ColiExpertView zero;
    return memcmp(view, &zero, sizeof(*view)) == 0;
}

int main(void) {
    static const ColiExpertStoreOps ops = {
        res_lookup, res_release, NULL, res_stats, res_destroy,
        res_reserve, res_publish, res_abort, res_pin, res_unpin,
        res_lookup_batch
    };
    /* Segment scratch handed to reservations via store.gpu borrow (mock). */
    static ColiExpertSegment segments[2];
    unsigned char stage_w[SEG_WEIGHTS_BYTES];
    unsigned char stage_s[SEG_SCALES_BYTES];
    ResState state;
    ColiExpertStore store;
    ColiExpertReservation res;
    ColiExpertView view;
    ColiExpertStoreStats stats;
    ColiExpertKey key = {3, 11};
    ColiExpertCoreKey core = coli_expert_core_key(key);

    memset(&state, 0, sizeof(state));
    segments[0].data = stage_w;
    segments[0].bytes = sizeof(stage_w);
    segments[1].data = stage_s;
    segments[1].bytes = sizeof(stage_s);
    store.ops = &ops;
    store.state = &state;
    store.gpu = segments;

    /* 1) Happy path: reserve -> fill -> publish with atomic lease. */
    if (coli_expert_reserve(&store, &core, &res) != COLI_EXPERT_OK) return 1;
    if (res.segment_count != 2 || res.segments != segments) return 1;
    memset(stage_w, 0xAB, sizeof(stage_w));
    memset(stage_s, 0xCD, sizeof(stage_s));
    if (coli_expert_publish(&store, &res, &view) != COLI_EXPERT_OK) return 1;
    if (res.segments != NULL || res.segment_count != 0) return 1;
    if (view.lease != &state || view.gate.data != state.blob) return 1;
    coli_expert_release(&store, &view);
    if (!view_is_cleared(&view)) return 1;

    /* Published bytes must now serve lookups as hits. */
    if (coli_expert_lookup(&store, key, &view) != 0) return 1;
    const unsigned char *blob = (const unsigned char *)view.gate.data;
    if (blob[0] != 0xAB || blob[SEG_WEIGHTS_BYTES - 1] != 0xAB) return 1;
    coli_expert_release(&store, &view);

    /* 2) Double reserve on the same key fails BUSY while one is active. */
    if (coli_expert_reserve(&store, &core, &res) != COLI_EXPERT_OK) return 1;
    {
        ColiExpertReservation res2;
        if (coli_expert_reserve(&store, &core, &res2) !=
            COLI_EXPERT_ERR_BUSY) return 1;
    }

    /* 3) Abort discards: nothing resident changes, counters move. */
    if (coli_expert_publish(&store, &res, NULL) != COLI_EXPERT_OK) return 1;
    if (coli_expert_reserve(&store, &core, &res) != COLI_EXPERT_OK) return 1;
    coli_expert_abort(&store, &res);
    if (res.segments != NULL || state.stats.aborts != 1) return 1;
    if (state.stats.reservations_active != 0) return 1;
    if (coli_expert_lookup(&store, key, &view) != 0) return 1; /* still old */
    coli_expert_release(&store, &view);

    /* 4) Pin protects the resident from eviction; unpin restores it. */
    if (coli_expert_reserve(&store, &core, &res) != COLI_EXPERT_OK) return 1;
    if (coli_expert_publish(&store, &res, NULL) != COLI_EXPERT_OK) return 1;
    if (coli_expert_pin(&store, &core) != COLI_EXPERT_OK) return 1;
    if (res_evict(&state, &core) != -1) return 1; /* pinned: immune */
    coli_expert_unpin(&store, &core);
    if (res_evict(&state, &core) != 0) return 1;  /* unpinned: evictable */
    if (state.resident) return 1;

    /* 5) Batch lookup: hit + miss mix returns per-entry semantics. */
    {
        ColiExpertKey keys[2] = {{3, 11}, {4, 12}};
        ColiExpertView views[2];
        /* republish so first key hits */
        if (coli_expert_reserve(&store, &core, &res) != COLI_EXPERT_OK) return 1;
        if (coli_expert_publish(&store, &res, NULL) != COLI_EXPERT_OK) return 1;
        memset(views, 0x5a, sizeof(views));
        int ok = coli_expert_lookup_batch(&store, keys, 2, views);
        if (ok != 1) return 1;
        if (views[0].lease != &state) return 1;
        if (!view_is_cleared(&views[1])) return 1;
        coli_expert_release(&store, &views[0]);
    }

    /* 6) Widened stats conservation. */
    ops.stats(&store, &stats);
    if (stats.physical_loads != 4) return 1;             /* publishes above */
    if (stats.publishes != 4 || stats.aborts != 1) return 1;
    if (stats.admitted_weight_bytes != 4 * SEG_WEIGHTS_BYTES) return 1;
    if (stats.admitted_scale_bytes != 4 * SEG_SCALES_BYTES) return 1;
    if (!(stats.admission_ms_total > 0.0)) return 1;
    if (stats.pinned_count != 0) return 1;               /* pinned then unpinned */

    ops.destroy(&store);
    if (state.destroyed_with_lease) return 1;
    puts("expert reservation tests: ok");
    return 0;
}
