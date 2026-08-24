#ifndef COLIBRI_EXPERT_TELEMETRY_H
#define COLIBRI_EXPERT_TELEMETRY_H

/*
 * Engine-agnostic expert-cache telemetry (Forge F1 Phase 5).
 *
 * Two stream formats are shared surfaces — their exact byte layouts are
 * load-bearing (oracle fixtures and replay tooling parse them):
 *
 *   v3 "moe trace" TSV (COLI_MOE_TRACE), one row per cache mutation:
 *     seq \t class \t event \t tok \t layer \t eid \t fmt \t bytes \t
 *     adm_ms \t victim_eid \t slot \n
 *     class ∈ {DEMAND, PILOT}; event ∈ {HIT, EVICT, INSERT};
 *     seq is assigned under the caller's mutation lock so rows carry the
 *     total cache-mutation order.
 *
 *   v4 "request stream" (COLI_TRACE_REQ), policy-independent intents only:
 *     seq SP kind SP body \n      (kind ∈ {E, B, R, C})
 *
 * Emitters here reproduce the promoted qwen36.c implementations byte for
 * byte; tests/test_expert_telemetry.c pins the exact bytes.
 *
 * The counters struct mirrors the widened census counter set so engines
 * can adopt one aggregate instead of scattered statics.
 */

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    FILE *fp;                 /* NULL = disabled */
    unsigned long long seq;   /* total-order sequence, caller-serialized */
} ColiExpertTraceV3;

typedef struct {
    FILE *fp;                 /* NULL = disabled */
    unsigned long long seq;
} ColiExpertTraceV4;

static inline void coli_expert_telemetry_v3_attach(ColiExpertTraceV3 *t, FILE *fp) {
    t->fp = fp;
}

static inline void coli_expert_telemetry_v3_emit(
    ColiExpertTraceV3 *t, const char *cls, const char *event, long long tok,
    int layer, int eid, int fmt, long long bytes, double adm_ms,
    long long victim_eid, int slot) {
    if (!t || !t->fp) return;
    unsigned long long seq = ++t->seq;
    fprintf(t->fp,
            "%llu\t%s\t%s\t%lld\t%d\t%d\t%d\t%lld\t%.3f\t%lld\t%d\n",
            seq, cls, event, (long long)tok, layer, eid, fmt,
            (long long)bytes, adm_ms, (long long)victim_eid, slot);
}

static inline void coli_expert_telemetry_v4_attach(ColiExpertTraceV4 *t, FILE *fp) {
    t->fp = fp;
}

static inline void coli_expert_telemetry_v4_emit(ColiExpertTraceV4 *t,
                                                 const char *kind,
                                                 const char *body) {
    if (!t || !t->fp) return;
    unsigned long long seq = ++t->seq;
    fprintf(t->fp, "%llu %s %s\n", seq, kind, body);
}

/* Engine-agnostic counter aggregate (the census section 1.3 set). Engines
 * may keep richer locals; this is the portable subset other tools read. */
typedef struct {
    /* requests / outcomes */
    unsigned long long demand_loads;
    unsigned long long pilot_loads;
    unsigned long long acq_hits;
    unsigned long long acq_miss;
    unsigned long long coalesce_waits;
    unsigned long long coalesce_skips;
    /* bytes */
    unsigned long long demand_bytes;
    unsigned long long pilot_bytes;
    unsigned long long admitted_bytes_int3;
    unsigned long long admitted_bytes_int4;
    /* admission latency decomposition (ms) */
    double demand_admission_ms;
    double demand_weight_ms;
    double demand_scale_ms;
    double pilot_admission_ms;
    double pilot_weight_ms;
    double pilot_scale_ms;
    double eg_lock_wait_ms;
    double eg_lookup_ms;
    double eg_victim_ms;
    /* format mix */
    unsigned long long routed_int3;
    unsigned long long routed_int4;
    unsigned long long cache_hit_int3;
    unsigned long long cache_hit_int4;
    unsigned long long cache_miss_int3;
    unsigned long long cache_miss_int4;
    /* batch windows */
    unsigned long long pb_io_us;
    double pb_wall_ms;
    unsigned long long expert_gemv_parallel_invocations;
} ColiExpertTelemetryCounters;

#ifdef __cplusplus
}
#endif
#endif /* COLIBRI_EXPERT_TELEMETRY_H */
