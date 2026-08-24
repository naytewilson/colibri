#ifndef COLIBRI_EXPERT_STORE_H
#define COLIBRI_EXPERT_STORE_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiExpertStore ColiExpertStore;

typedef struct {
    int layer;
    int expert;
} ColiExpertKey;

typedef struct {
    ColiExpertKey key;
    ColiTensorView gate;
    ColiTensorView down;
    ColiTensorView up;
    void *lease;
    /* v1.2 additive: physical slot id of the resident copy this view aliases
     * (trace-tooling provenance). -1 when the backend does not expose one. */
    long slot_hint;
} ColiExpertView;

typedef struct {
    uint64_t requests;
    uint64_t hits;
    uint64_t misses;
    uint64_t prefetched;
    uint64_t prefetch_hits;
    uint64_t bytes_read;
    uint64_t resident_bytes;
    uint64_t capacity_bytes;
    /* --- v1.1 additive widening (zero when a backend does not supply it) --- */
    uint64_t demand_requests;     /* demand-path acquisitions */
    uint64_t demand_hits;
    uint64_t demand_misses;
    uint64_t prefetch_requests;   /* advisory prefetch calls (keys summed) */
    uint64_t logical_requests;    /* acquisitions incl coalesced duplicates */
    uint64_t physical_loads;      /* device reads actually performed */
    uint64_t admitted_bytes;      /* bytes published into residency */
    uint64_t admitted_weight_bytes;
    uint64_t admitted_scale_bytes;
    double admission_ms_total;    /* summed wall time of admissions */
    uint64_t coalesced_requests;  /* served by waiting on an in-flight load */
    uint64_t publishes;
    uint64_t aborts;
    uint64_t reservations_active;
    uint64_t pinned_count;
} ColiExpertStoreStats;

/* Error codes returned by the reservation/pin surface. */
#define COLI_EXPERT_OK           0
#define COLI_EXPERT_ERR_INVALID   (-1) /* NULL/garbage arguments */
#define COLI_EXPERT_ERR_BUSY      (-2) /* key already has an active reservation */
#define COLI_EXPERT_ERR_NOCAP     (-3) /* no capacity to admit */
#define COLI_EXPERT_ERR_UNSUPPORTED (-4) /* backend does not implement this op */

/* Internal core identity: (layer, role, index). role=EXPERT is first-class;
 * future WeightStore roles extend this enum without changing the contract.
 * The public typed facade remains ColiExpertKey{layer, expert}. */
#define COLI_EXPERT_ROLE_EXPERT 0

typedef struct {
    int layer;
    int role;
    int index;
} ColiExpertCoreKey;

static inline ColiExpertCoreKey coli_expert_core_key(ColiExpertKey key) {
    ColiExpertCoreKey core;
    core.layer = key.layer;
    core.role = COLI_EXPERT_ROLE_EXPERT;
    core.index = key.expert;
    return core;
}

/* One writable destination segment of a reservation's slot memory (e.g. a
 * weights blob and a scales blob). A loader may pread directly into `data`
 * instead of staging + copying, so admission stays zero-extra-copy. */
typedef struct {
    void *data;
    size_t bytes; /* capacity of this segment */
} ColiExpertSegment;

/* In-flight admission identity. Lifecycle: reserve -> fill segments ->
 * publish XOR abort (exactly one terminal op; see contract block below). */
typedef struct {
    ColiExpertCoreKey key;
    uint32_t segment_count;
    ColiExpertSegment *segments; /* store-owned; valid until publish/abort */
    void *impl;                  /* backend-private state */
} ColiExpertReservation;

/*
 * ExpertStore lease contract:
 *
 * - After a successful lookup(), the caller must call release() exactly once
 *   on the same view (same store). Do not copy ColiExpertView; the lease is
 *   not shareable. Do not pass a view that already holds an active lease to
 *   lookup().
 * - On lookup failure the view is cleared; the caller must not use it and
 *   must not call release().
 * - release() clears the entire view. release() on an already-cleared or
 *   zero-initialized view is a no-op.
 * - destroy() requires zero active leases (debug builds assert).
 * - Thread-safety is implementation-specific. Callers must not assume that
 *   lookup/release/prefetch/stats/destroy are safe to call concurrently on
 *   the same store unless the concrete store documents that guarantee.
 *   Views must not be used concurrently from multiple threads.
 * - prefetch() is advisory, holds no lease, and must not evict a slot that
 *   still has an active lease.
 *
 * Reservation lifecycle contract (v1.1 additive; backends may omit these ops):
 *
 * - reserve() claims the in-flight admission identity for one core key. At
 *   most ONE active reservation per key exists at any time; a second reserve
 *   for the same key fails with COLI_EXPERT_ERR_BUSY (callers coalesce by
 *   waiting and re-running lookup instead of duplicating the load).
 * - Between a successful reserve() and its terminal op, the reservation's
 *   segment buffers are writable by exactly the owning thread (single writer).
 *   The buffers' contents are undefined; no residency exists yet.
 * - publish() makes the bytes resident. If `out_view` is non-NULL, an
 *   exclusive lease on the freshly published slot is handed back atomically
 *   with publication (no window where another thread can evict unseen); that
 *   lease follows all release() rules above. publish() clears the reservation
 *   struct (segment_count=0, segments=NULL).
 * - abort() discards the reservation without publishing anything; it must
 *   leave residency and accounting unchanged except reservations_active.
 *   abort() also clears the reservation struct.
 * - Exactly one terminal op per reservation; calling either twice, or on an
 *   already-cleared/zero reservation, is invalid (debug builds assert).
 * - pin()/unpin() mark a key eviction-protected (advisory where unsupported).
 *   Pinned slots must not be selected as eviction victims while unpinned
 *   candidates exist; unpin restores normal eligibility.
 * - Thread-safety of reserve/publish/abort/pin/unpin/lookup_batch is
 *   implementation-specific, same rule as lookup/release above.
 */
typedef struct {
    /* Returns zero on success. The view remains valid until release(). */
    int (*lookup)(ColiExpertStore *store, ColiExpertKey key,
                  ColiExpertView *view);
    void (*release)(ColiExpertStore *store, ColiExpertView *view);
    /* Prefetch is advisory. Unsupported or rejected requests return zero. */
    int (*prefetch)(ColiExpertStore *store, const ColiExpertKey *keys,
                    size_t count);
    void (*stats)(const ColiExpertStore *store, ColiExpertStoreStats *stats);
    void (*destroy)(ColiExpertStore *store);
    /* --- v1.1 additive reservation surface (NULL = unsupported; the inline
     * wrappers below fall back so existing backends compile unchanged) --- */
    int (*reserve)(ColiExpertStore *store, const ColiExpertCoreKey *key,
                   ColiExpertReservation *out);
    /* Publishes and optionally hands back an atomic lease (see contract). */
    int (*publish)(ColiExpertStore *store, ColiExpertReservation *res,
                   ColiExpertView *out_view);
    void (*abort)(ColiExpertStore *store, ColiExpertReservation *res);
    /* Advisory eviction protection. pin returns zero on success (or when
     * already pinned); unpin is a no-op when unsupported. */
    int (*pin)(ColiExpertStore *store, const ColiExpertCoreKey *key);
    void (*unpin)(ColiExpertStore *store, const ColiExpertCoreKey *key);
    /* Batch lookup: leases what it can, clears views of misses, returns the
     * number of successful leases. Never fails wholesale for bad keys — a
     * miss is not an error. */
    int (*lookup_batch)(ColiExpertStore *store, const ColiExpertKey *keys,
                        size_t count, ColiExpertView *views);
    /* --- v1.2 additive: pure residency query (NO accounting side effects;
     * policy probes must not pollute hit/miss streams). NULL falls back to
     * lookup+release, which DOES count. Returns 1 resident, 0 not. --- */
    int (*probe)(ColiExpertStore *store, const ColiExpertKey *key);
} ColiExpertStoreOps;

struct ColiExpertStore {
    const ColiExpertStoreOps *ops;
    void *state;
    /* Optional CUDA-tier mirror cache owned by the GPU translation unit
     * (deepseek_v4.c COLI_V4_UNIT_GPU). NULL when the tier is inactive; the
     * CPU lease path ignores it entirely. */
    void *gpu;
};

static inline int coli_expert_lookup(ColiExpertStore *store,
                                     ColiExpertKey key,
                                     ColiExpertView *view) {
    if (!store || !store->ops || !store->ops->lookup) {
        if (view) memset(view, 0, sizeof(*view));
        return -1;
    }
    int result = store->ops->lookup(store, key, view);
    if (result != 0 && view) memset(view, 0, sizeof(*view));
    return result;
}

static inline void coli_expert_release(ColiExpertStore *store,
                                       ColiExpertView *view) {
    if (store && store->ops && store->ops->release)
        store->ops->release(store, view);
    if (view) memset(view, 0, sizeof(*view));
}

/* --- v1.1 reservation/pin/batch wrappers with fallback semantics ---
 * Backends without the ops return COLI_EXPERT_ERR_UNSUPPORTED for the
 * reservation surface (callers fall back to synchronous lookup+load),
 * treat pin/unpin as advisory no-ops, and synthesize lookup_batch from
 * per-key lookup. Existing stores need no changes. */

static inline int coli_expert_reserve(ColiExpertStore *store,
                                      const ColiExpertCoreKey *key,
                                      ColiExpertReservation *out) {
    if (out) memset(out, 0, sizeof(*out));
    if (!store || !store->ops || !key || !out) return COLI_EXPERT_ERR_INVALID;
    if (!store->ops->reserve) return COLI_EXPERT_ERR_UNSUPPORTED;
    return store->ops->reserve(store, key, out);
}

static inline int coli_expert_publish(ColiExpertStore *store,
                                      ColiExpertReservation *res,
                                      ColiExpertView *out_view) {
    if (out_view) memset(out_view, 0, sizeof(*out_view));
    if (!store || !store->ops || !res) return COLI_EXPERT_ERR_INVALID;
    if (!store->ops->publish) return COLI_EXPERT_ERR_UNSUPPORTED;
    int rc = store->ops->publish(store, res, out_view);
    if (rc != 0 && out_view) memset(out_view, 0, sizeof(*out_view));
    return rc;
}

static inline void coli_expert_abort(ColiExpertStore *store,
                                     ColiExpertReservation *res) {
    if (store && store->ops && store->ops->abort && res)
        store->ops->abort(store, res);
    if (res) memset(res, 0, sizeof(*res));
}

static inline int coli_expert_pin(ColiExpertStore *store,
                                  const ColiExpertCoreKey *key) {
    if (!store || !store->ops || !key) return COLI_EXPERT_ERR_INVALID;
    if (!store->ops->pin) return COLI_EXPERT_OK; /* advisory: succeed */
    return store->ops->pin(store, key);
}

static inline void coli_expert_unpin(ColiExpertStore *store,
                                     const ColiExpertCoreKey *key) {
    if (store && store->ops && store->ops->unpin && key)
        store->ops->unpin(store, key);
}

/* destroy() requires zero active leases and reservations (debug builds
 * report both); double-destroy is a safe no-op. */
static inline void coli_expert_destroy(ColiExpertStore *store) {
    if (store && store->ops && store->ops->destroy)
        store->ops->destroy(store);
}

/* v1.2: pure residency probe — no accounting. Falls back to a counted
 * lookup+release when the backend has no probe (documented pollution). */
static inline int coli_expert_probe(ColiExpertStore *store,
                                    const ColiExpertKey *key) {
    if (!store || !store->ops || !key) return 0;
    if (store->ops->probe) return store->ops->probe(store, key) ? 1 : 0;
    ColiExpertView tmp;
    if (coli_expert_lookup(store, *key, &tmp) == 0) {
        coli_expert_release(store, &tmp);
        return 1;
    }
    return 0;
}

static inline int coli_expert_lookup_batch(ColiExpertStore *store,
                                           const ColiExpertKey *keys,
                                           size_t count,
                                           ColiExpertView *views) {
    if (!store || !store->ops) {
        if (views) memset(views, 0, count * sizeof(*views));
        return 0; 
    }
    if (count && (!keys || !views)) return 0;
    if (!store->ops->lookup_batch && store->ops->lookup) {
        /* Fallback: sequential per-key lookup; misses leave cleared views. */
        int ok = 0;
        for (size_t i = 0; i < count; i++) {
            if (coli_expert_lookup(store, keys[i], &views[i]) == 0) ok++;
        }
        return ok;
    }
    if (!store->ops->lookup_batch) {
        memset(views, 0, count * sizeof(*views));
        return 0;
    }
    return store->ops->lookup_batch(store, keys, count, views);
}

#ifdef __cplusplus
}
#endif

#endif
