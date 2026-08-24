#ifndef COLIBRI_ADMISSION_H
#define COLIBRI_ADMISSION_H

/*
 * Admission scheduler — POLICY over ExpertStore reservation state
 * (Forge F1 Phase 4).
 *
 * Owns everything the census assigned to the policy side and nothing the
 * store owns: dedupe, coalescing windows, bounded parallel admission,
 * batch acquisition orchestration, prefetch ordering. It maintains NO
 * second authoritative in-flight/resident table — every decision consults
 * the store (lookup / reserve / publish / abort), so the store remains
 * the single source of residency truth.
 *
 * All knobs default OFF (serial, fail-fast): a zero-initialized config
 * reproduces plain synchronous lookups, matching promoted-Qwen semantics
 * where every advanced arm is opt-in.
 */

#include <stddef.h>

#include "expert_store.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Loader callback: fill a reservation's segments with the expert's bytes.
 * Called by the scheduler between a successful reserve() and its publish();
 * runs OUTSIDE any store lock. Zero on success; non-zero aborts the
 * admission (nothing is published). */
typedef int (*ColiAdmissionLoadFn)(void *userdata,
                                   const ColiExpertCoreKey *key,
                                   ColiExpertReservation *res);

typedef struct {
    int max_parallel;     /* concurrent admissions per batch; <=1 = serial */
    int wait_busy_ms;     /* coalescing poll interval; 0 = fail-fast on BUSY */
    int wait_busy_max_ms; /* total coalescing budget; 0 = unlimited retries */
} ColiAdmissionConfig;

typedef struct {
    uint64_t acquires;
    uint64_t hits;
    uint64_t loads;
    uint64_t load_failures;
    uint64_t coalesced_waits;
    uint64_t busy_timeouts;
    uint64_t batches;
    uint64_t batch_keys_in;
    uint64_t batch_dedupe_removed;
    uint64_t prefetch_enqueued;
} ColiAdmissionStats;

typedef struct ColiAdmission ColiAdmission;

/* Takes ownership of NO memory: store and load userdata are borrowed and
 * must outlive the admission. cfg may be NULL (all defaults). */
ColiAdmission *coli_admission_new(ColiExpertStore *store,
                                  ColiAdmissionLoadFn load_fn,
                                  void *load_userdata,
                                  const ColiAdmissionConfig *cfg);
void coli_admission_free(ColiAdmission *adm);

/* One demand acquisition. On success *out holds an exclusive lease
 * (release through the store). Returns 0, COLI_EXPERT_ERR_BUSY (busy and
 * fail-fast or window expired), or a negative store error. */
int coli_admission_acquire(ColiAdmission *adm, const ColiExpertKey *key,
                           ColiExpertView *out);

/* Batch acquisition orchestration: dedupe (first-touch order kept),
 * acquire phase for every unique key, concurrent load+publish of the
 * misses up to max_parallel, join, then lease every requested entry in
 * caller order (duplicates share nothing — each view is its own lease).
 * Returns the number of successfully leased entries. */
int coli_admission_acquire_batch(ColiAdmission *adm,
                                 const ColiExpertKey *keys, size_t count,
                                 ColiExpertView *views);

/* Deterministic prefetch ordering: stable-sort keys by (layer, index)
 * ascending and drop duplicates in place. Returns the kept count. */
size_t coli_admission_order_prefetch(ColiExpertKey *keys, size_t count);

/* Advisory prefetch through the store, in the given order. */
int coli_admission_prefetch(ColiAdmission *adm, const ColiExpertKey *keys,
                            size_t count);

/* Bounded-parallel job runner (shared concurrency policy): runs `count`
 * jobs via job(arg, i) across at most `max_workers` pthreads, joining
 * before return. max_workers <= 1 runs inline (default OFF == serial);
 * workers are fanned out round-robin over contiguous ranges. Loads belong
 * INSIDE jobs; anything touching shared trace order belongs to the caller's
 * own locking discipline inside job bodies. */
void coli_admission_run_parallel(ColiAdmission *adm,
                                 void (*job)(void *arg, int idx),
                                 void *arg, int count, int max_workers);

/* Read back the effective config (engines derive coalescing-window
 * behavior from it instead of hard-coding their own knobs). */
const ColiAdmissionConfig *coli_admission_config(const ColiAdmission *adm);

const ColiAdmissionStats *coli_admission_stats(const ColiAdmission *adm);

#ifdef __cplusplus
}
#endif
#endif /* COLIBRI_ADMISSION_H */
