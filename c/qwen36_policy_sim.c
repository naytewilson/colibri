/* qwen36_policy_sim — counterfactual cache-policy simulator for qwen36 MoE.
 *
 * Reads ONLY the policy-independent v4 request stream (COLI_TRACE_REQ output:
 * E/B/R/C/Q/P events). It owns ALL cache decisions — hit/miss, admission,
 * victim, slot, occupancy, bytes — deriving them from its own state, exactly
 * like the runtime policy (2abb9cf): per-layer LRU cap slots, in-flight
 * loading registry, coalescing waits, single-worker pilot queue with
 * is_queued gating and ring-full drops.
 *
 * The v3 outcome trace may be passed as an ORACLE for post-hoc comparison
 * (--oracle <file>). The simulator's decisions NEVER read it: simulation runs
 * to completion first, aggregation of the oracle happens afterwards, and only
 * aggregate counters / sequences are diffed.
 *
 * Frozen-schedule semantics (documented limitation): PILOT loads are async in
 * the runtime; here a pilot admission reserved at Q publishes at its recorded
 * P marker, and demand requests blocked on that eid resolve at P. Load
 * latencies are NOT modeled — no wall-clock predictions may be drawn.
 *
 * Usage: qwen36_policy_sim <req.v4> [--oracle <trace_v3.tsv>] [--cap N]
 * Exit 0 = derived outcomes match oracle within stated tolerances.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_LAYERS 64
#define MAX_EXPERTS 512

/* ---------------- simulator-owned state ---------------- */
typedef struct {
    int cap;
    int nslots;
    int slot_eid[512];
    int slot_fmt[512];
    long used[512];
    int16_t loading[MAX_EXPERTS];      /* eid -> slot index, -1 none */
} Sim;

static Sim g_sim[MAX_LAYERS];
static uint8_t g_is_queued[MAX_LAYERS * MAX_EXPERTS];
static int g_qdepth = 0;
static long g_clock = 0;

static int   g_meta_fmt[MAX_LAYERS][MAX_EXPERTS];
static long long g_meta_bytes[MAX_LAYERS][MAX_EXPERTS];

/* derived outcome record */
typedef struct { int kind; /*0=devict,1=pvictim*/ int layer, eid; } EvRec;
static EvRec *g_evicts = NULL; static size_t g_n_evicts = 0, g_cap_evicts = 0;
static void rec_evict(int kind, int layer, int eid) {
    if (g_n_evicts == g_cap_evicts) {
        g_cap_evicts = g_cap_evicts ? g_cap_evicts * 2 : 4096;
        g_evicts = realloc(g_evicts, g_cap_evicts * sizeof(EvRec));
        if (!g_evicts) exit(2);
    }
    g_evicts[g_n_evicts].kind = kind; g_evicts[g_n_evicts].layer = layer; g_evicts[g_n_evicts].eid = eid;
    g_n_evicts++;
}

static long d_hits, d_misses, w_dhits, w_dmisses, d_evicts, p_evicts, d_ins, p_ins, d_waits, p_skips;
static unsigned long long d_bytes, p_bytes;
static FILE *g_dump_fp = NULL;   /* --dump-derived <file>: per-request outcome trace */
static long g_req_no = 0;

/* blocked demand acquisitions waiting for an in-flight pilot publish */
typedef struct { int layer, eid, tok; } Waiter;
static Waiter g_waiters[65536]; static int g_n_waiters = 0;
/* demand reservations deferred because every slot was in flight (rare) */
typedef struct { int layer, eid, tok; } Deferred;
static Deferred g_deferred[65536]; static int g_n_deferred = 0;
/* pending (reserved, not yet published) admissions: kind 1=demand 0=pilot */
static int16_t g_pend_slot[MAX_LAYERS][MAX_EXPERTS];
static uint8_t g_pend_dem[MAX_LAYERS][MAX_EXPERTS];

static int find_resident(Sim *c, int eid) {
    for (int i = 0; i < c->nslots; i++)
        if (c->slot_eid[i] == eid) return i;
    return -1;
}

static Sim *ensure_sim(int layer, int cap) {
    Sim *c = &g_sim[layer];
    if (!c->cap) {
        c->cap = cap;
        for (int e2 = 0; e2 < MAX_EXPERTS; e2++) c->loading[e2] = -1;  /* static arrays start zeroed */
    }
    return c;
}

static int pick_victim(Sim *c, int *have_any_inflight_only) {
    /* lowest 'used' among residents; runtime skips pinned (none here) and
     * in-flight (eid<0). If every allocated slot is in flight -> -1. */
    int lru = -1;
    for (int i = 0; i < c->nslots; i++) {
        if (c->slot_eid[i] < 0) continue;
        if (lru < 0 || c->used[i] < c->used[lru]) lru = i;
    }
    if (lru >= 0) { *have_any_inflight_only = 0; return lru; }
    *have_any_inflight_only = 1;
    return -1;
}

static void sim_reserve_demand(int layer, int eid, int tok) {
    Sim *c = &g_sim[layer];
    int v;
    if (c->nslots < c->cap) {
        v = c->nslots++;                       /* fresh capacity, no eviction */
    } else {
        int inflight_only = 0;
        v = pick_victim(c, &inflight_only);
        if (v < 0) {
            if (g_n_deferred < 65536) { g_deferred[g_n_deferred].layer = layer; g_deferred[g_n_deferred].eid = eid; g_deferred[g_n_deferred].tok = tok; g_n_deferred++; }
            return;
        }
        if (c->slot_eid[v] >= 0) { d_evicts++; rec_evict(0, layer, c->slot_eid[v]); }
        c->slot_eid[v] = -1;                   /* in-flight */
    }
    c->loading[eid] = (int16_t)v;
    /* miss is counted at reservation (like the runtime); the INSERT (bytes,
     * admission) lands at the recorded publish marker P DP */
    d_misses++; if (tok >= 0) w_dmisses++;
    g_pend_slot[layer][eid] = (int16_t)v; g_pend_dem[layer][eid] = 1;
}

static void sim_publish(int layer, int eid) {
    Sim *c = &g_sim[layer];
    int s = c->loading[eid];
    if (s < 0) return;                        /* unknown publish (defensive) */
    c->slot_eid[s] = eid; c->slot_fmt[s] = g_meta_fmt[layer][eid]; c->used[s] = ++g_clock;
    c->loading[eid] = -1;
    if (g_pend_dem[layer][eid]) { d_ins++; d_bytes += (unsigned long long)g_meta_bytes[layer][eid]; }
    else { p_ins++; p_bytes += (unsigned long long)g_meta_bytes[layer][eid]; }
    g_pend_slot[layer][eid] = -1;
}

int main(int argc, char **argv) {
    const char *path = NULL, *oracle = NULL;
    int cap = 128;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--oracle") && i + 1 < argc) oracle = argv[++i];
        else if (!strcmp(argv[i], "--cap") && i + 1 < argc) cap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-derived") && i + 1 < argc) g_dump_fp = fopen(argv[++i], "w");
        else if (!path) path = argv[i];
        else { fprintf(stderr, "unexpected arg %s\n", argv[i]); return 2; }
    }
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return 2; }

    char line[4096];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[1] == ' ') continue;   /* header lines start "<seq> #" */
        unsigned long long seq;
        char tag[8], sub[16];
        if (sscanf(line, "%llu %7s %15s", &seq, tag, sub) != 3) continue;

        if (!strcmp(tag, "E")) {
            int layer, eid, fmt2; long long b;
            sscanf(line, "%llu %*s %d %d %d %lld", &seq, &layer, &eid, &fmt2, &b);
            if (layer < MAX_LAYERS && eid < MAX_EXPERTS) { g_meta_fmt[layer][eid] = fmt2; g_meta_bytes[layer][eid] = b; }
        } else if (!strcmp(tag, "R")) {
            long long tok; int layer, eid; double mass;
            sscanf(line, "%llu %*s %*s %lld %d %d %lf", &seq, &tok, &layer, &eid, &mass);
            Sim *c = ensure_sim(layer, cap);
            if (!c->cap) {
                c->cap = cap;
                for (int e2 = 0; e2 < MAX_EXPERTS; e2++) c->loading[e2] = -1;  /* static arrays start zeroed */
            }
            int r = find_resident(c, eid);
            if (r >= 0) { c->used[r] = ++g_clock; d_hits++; if (tok >= 0) w_dhits++; if (g_dump_fp) fprintf(g_dump_fp, "%ld R %lld %d %d HIT\n", g_req_no++, tok, layer, eid); continue; }
            if (c->loading[eid] >= 0) {
                if (g_n_waiters < 65536) { g_waiters[g_n_waiters].layer = layer; g_waiters[g_n_waiters].eid = eid; g_waiters[g_n_waiters].tok = (int)tok; g_n_waiters++; }
                d_waits++;
                continue;
            }
            sim_reserve_demand(layer, eid, (int)tok);
        } else if (!strcmp(tag, "C")) {
            long long srctok; int layer, eid, rank; double conf;
            sscanf(line, "%llu %*s %*s %lld %d %d %d %lf", &seq, &srctok, &layer, &eid, &rank, &conf);
            uint8_t *q = &g_is_queued[(size_t)layer * MAX_EXPERTS + eid];
            Sim *c = ensure_sim(layer, cap);
            if (!c->cap) c->cap = cap;
            if (*q) continue;                                   /* already queued */
            if (find_resident(c, eid) >= 0) continue;           /* runtime: resident -> no enqueue */
            if (g_qdepth < 4096) { *q = 1; g_qdepth++; }
            /* ring full -> drop silently, like the runtime */
        } else if (!strcmp(tag, "Q")) {
            int layer, eid;
            sscanf(line, "%llu %*s %*s %d %d", &seq, &layer, &eid);
            uint8_t *q = &g_is_queued[(size_t)layer * MAX_EXPERTS + eid];
            if (g_qdepth > 0) g_qdepth--;
            Sim *c = ensure_sim(layer, cap);
            if (!*q) continue;                                  /* stale pop: runtime early-outs */
            if (find_resident(c, eid) >= 0) { *q = 0; continue; }
            if (c->loading[eid] >= 0) { *q = 0; p_skips++; continue; }
            int v;
            if (c->nslots < c->cap) {
                v = c->nslots++;                                /* fresh capacity */
            } else {
                int inflight_only = 0;
                v = pick_victim(c, &inflight_only);
                if (v < 0) { *q = 0; continue; }                /* runtime giveup */
                if (c->slot_eid[v] >= 0) { p_evicts++; rec_evict(1, layer, c->slot_eid[v]); }
                c->slot_eid[v] = -1;
            }
            c->loading[eid] = (int16_t)v;
            g_pend_slot[layer][eid] = (int16_t)v; g_pend_dem[layer][eid] = 0;
            *q = 0;
        } else if (!strcmp(tag, "P")) {
            int layer, eid;
            sscanf(line, "%llu %*s %*s %d %d", &seq, &layer, &eid);
            sim_publish(layer, eid);
        } else if (!strcmp(tag, "B")) {
            /* boundary: no state effect beyond ordering */
        } else if (!strcmp(tag, "W")) {
            /* a coalesced demand waiter woke at this point in the lock order:
             * touch the resident copy now (exact LRU recency ordering) */
            int layer, eid;
            sscanf(line, "%llu %*s %*s %d %d", &seq, &layer, &eid);
            Sim *c = ensure_sim(layer, cap);
            int resolved = 0;
            for (int i = 0; i < g_n_waiters; i++) {
                if (g_waiters[i].layer == layer && g_waiters[i].eid == eid) {
                    int r = find_resident(c, eid);
                    if (r >= 0) { c->used[r] = ++g_clock; d_hits++; if (g_waiters[i].tok >= 0) w_dhits++; }
                    if (g_dump_fp) fprintf(g_dump_fp, "%ld W %d %d %s\n", g_req_no++, layer, eid, r >= 0 ? "HIT" : "ABSENT");
                    g_waiters[i] = g_waiters[--g_n_waiters]; i--;
                    resolved = 1;
                    break;
                }
            }
            if (!resolved) { /* wake without waiter: defensive, no-op */ }
        }
        /* retry deferred demand reservations after each event — bounded: only
         * the items present when the pass starts; re-deferrals wait for the
         * next event (prevents unbounded pop/re-append loops) */
        {
            int limit = g_n_deferred;
            for (int i = 0; i < limit && i < g_n_deferred; i++) {
                Deferred d = g_deferred[i];
                g_deferred[i] = g_deferred[--g_n_deferred]; i--;
                Sim *c = &g_sim[d.layer];
                if (find_resident(c, d.eid) >= 0 || c->loading[d.eid] >= 0) continue;
                sim_reserve_demand(d.layer, d.eid, d.tok);
            }
        }
    }
    fclose(f);

    printf("== SIMULATOR DERIVED (from request stream only) ==\n");
    printf("demand hits=%ld (window %ld) misses=%ld (window %ld)\n", d_hits, w_dhits, d_misses, w_dmisses);
    printf("hit rate: %.2f%% cumulative | %.2f%% decode-window\n",
           (d_hits + d_misses) ? 100.0 * d_hits / (d_hits + d_misses) : 0.0,
           (w_dhits + w_dmisses) ? 100.0 * w_dhits / (w_dhits + w_dmisses) : 0.0);
    printf("admissions: demand=%ld pilot=%ld | coalesce: waits=%ld pilot-skips=%ld\n", d_ins, p_ins, d_waits, p_skips);
    printf("evictions: demand=%ld pilot=%ld | bytes: demand=%llu pilot=%llu\n", d_evicts, p_evicts, d_bytes, p_bytes);

    int fail = 0;
    if (!oracle) { printf("VERDICT: NO_ORACLE_GIVEN\n"); return 0; }

    /* -------- post-hoc oracle aggregation (never feeds decisions) -------- */
    FILE *of = fopen(oracle, "r");
    if (!of) { perror(oracle); return 2; }
    long o_dhits = 0, o_dmiss = 0, o_phits = 0, o_wdh = 0, o_wdm = 0, o_dev = 0, o_pev = 0, o_dins = 0, o_pins = 0;
    unsigned long long o_dbytes = 0, o_pbytes = 0;
    size_t o_n_evicts = 0, ev_mismatch_at = SIZE_MAX;
    static int occ[MAX_LAYERS][512];   /* eid resident per slot, -1 free */
    memset(occ, 0xFF, sizeof occ);
    EvRec *o_evicts = NULL; 
    while (fgets(line, sizeof line, of)) {
        if (line[0] == '#') continue;
        unsigned long long seq; char cls[16], ev[16];
        long long tok, victim, bytes; int layer, eid, fmt2, slot; double adm;
        if (sscanf(line, "%llu\t%15s\t%15s\t%lld\t%d\t%d\t%d\t%lld\t%lf\t%lld\t%d",
                   &seq, cls, ev, &tok, &layer, &eid, &fmt2, &bytes, &adm, &victim, &slot) != 11) continue;
        int is_pilot = !strcmp(cls, "PILOT");
        int win = (tok >= 0);
        if (!strcmp(ev, "HIT")) { if (!is_pilot) { o_dhits++; if (win) o_wdh++; } else o_phits++; }
        else if (!strcmp(ev, "EVICT")) {
            if (is_pilot) o_pev++; else o_dev++;
            if (slot >= 0 && slot < 512) occ[layer][slot] = -1;   /* in-flight */
            { EvRec *r2 = realloc(o_evicts, (o_n_evicts + 1) * sizeof(EvRec));
              if (r2) { o_evicts = r2; o_evicts[o_n_evicts].kind = is_pilot; o_evicts[o_n_evicts].layer = layer; o_evicts[o_n_evicts].eid = eid; } }
            if (o_n_evicts < g_n_evicts && (g_evicts[o_n_evicts].kind != is_pilot || g_evicts[o_n_evicts].layer != layer || g_evicts[o_n_evicts].eid != eid)) { if (ev_mismatch_at == SIZE_MAX) ev_mismatch_at = o_n_evicts; }
            o_n_evicts++;
        }
        else if (!strcmp(ev, "INSERT")) {
            occ[layer][slot] = eid;
            if (is_pilot) { o_pins++; o_pbytes += (unsigned long long)bytes; }
            else { o_dins++; o_dbytes += (unsigned long long)bytes; o_dmiss++; if (win) o_wdm++; }
        }
    }
    fclose(of);

    /* final resident fingerprint: per-layer sorted resident-eid multiset */
    unsigned long long fp_o = 1469598103934665603ULL, fp_s = 1469598103934665603ULL;
    for (int l = 0; l < MAX_LAYERS; l++) {
        int oe[512], se[512]; int on = 0, sn = 0;
        for (int s = 0; s < 512; s++) if (occ[l][s] >= 0 && occ[l][s] < MAX_EXPERTS) oe[on++] = occ[l][s];
        for (int s = 0; s < g_sim[l].nslots; s++) if (g_sim[l].slot_eid[s] >= 0) se[sn++] = g_sim[l].slot_eid[s];
        /* insertion sort (small n) then hash order-independently via sum of h(eid) */
        unsigned long long ho = 0, hs = 0;
        for (int i = 0; i < on; i++) { for (int j = i + 1; j < on; j++) if (oe[j] < oe[i]) { int t = oe[i]; oe[i] = oe[j]; oe[j] = t; } ho = ho * 1000003u + (unsigned)oe[i]; }
        for (int i = 0; i < sn; i++) { for (int j = i + 1; j < sn; j++) if (se[j] < se[i]) { int t = se[i]; se[i] = se[j]; se[j] = t; } hs = hs * 1000003u + (unsigned)se[i]; }
        fp_o = (fp_o ^ (unsigned)(ho & 0xFFFFFFFFu) ^ (unsigned)l) * 1099511628211ULL;
        fp_s = (fp_s ^ (unsigned)(hs & 0xFFFFFFFFu) ^ (unsigned)l) * 1099511628211ULL;
    }

    printf("== ORACLE RECORDED (v3 outcome trace) ==\n");
    printf("demand hits=%ld (window %ld) misses=%ld (window %ld)\n", o_dhits, o_wdh, o_dmiss, o_wdm);
    printf("admissions: demand=%ld pilot=%ld | evictions: demand=%ld pilot=%ld\n", o_dins, o_pins, o_dev, o_pev);
    printf("bytes: demand=%llu pilot=%llu\n", o_dbytes, o_pbytes);

    printf("== COMPARISON ==\n");
    #define CMP_INT(name, a, b) do { \
        if ((a) != (b)) { printf("  %-34s DIFF sim=%ld oracle=%ld\n", name, (long)(a), (long)(b)); fail = 1; } \
        else printf("  %-34s EXACT (%ld)\n", name, (long)(a)); \
    } while (0)
    CMP_INT("cumulative demand hits", d_hits, o_dhits);
    CMP_INT("cumulative demand misses", d_misses, o_dmiss);
    CMP_INT("decode-window demand hits", w_dhits, o_wdh);
    CMP_INT("decode-window demand misses", w_dmisses, o_wdm);
    CMP_INT("demand admissions", d_ins, o_dins);
    CMP_INT("pilot admissions", p_ins, o_pins);
    CMP_INT("demand evictions", d_evicts, o_dev);
    CMP_INT("pilot evictions", p_evicts, o_pev);
    if (d_bytes != o_dbytes) { printf("  %-34s DIFF sim=%llu oracle=%llu\n", "demand admitted bytes", d_bytes, o_dbytes); fail = 1; }
    else printf("  %-34s EXACT (%llu)\n", "demand admitted bytes", d_bytes);
    if (p_bytes != o_pbytes) { printf("  %-34s DIFF sim=%llu oracle=%llu\n", "pilot admitted bytes", p_bytes, o_pbytes); fail = 1; }
    else printf("  %-34s EXACT (%llu)\n", "pilot admitted bytes", p_bytes);
    if (o_n_evicts != g_n_evicts || ev_mismatch_at != SIZE_MAX) {
        printf("  victim sequence                      DIFF (len sim=%zu oracle=%zu%s", g_n_evicts, o_n_evicts,
               ev_mismatch_at != SIZE_MAX ? ", first mismatch" : "");
        if (ev_mismatch_at != SIZE_MAX && ev_mismatch_at < g_n_evicts)
            printf(" at %zu: sim(%s l%d e%d)", ev_mismatch_at, g_evicts[ev_mismatch_at].kind ? "PILOT" : "DEMAND", g_evicts[ev_mismatch_at].layer, g_evicts[ev_mismatch_at].eid);
        printf(")\n");
        if (ev_mismatch_at != SIZE_MAX && o_evicts) {
            size_t lo = ev_mismatch_at > 4 ? ev_mismatch_at - 4 : 0;
            size_t hi = ev_mismatch_at + 5;
            if (hi > o_n_evicts) hi = o_n_evicts;
            if (hi > g_n_evicts) hi = g_n_evicts;
            printf("  context [idx] oracle | sim:\n");
            for (size_t k = lo; k < hi; k++) {
                if (k < o_n_evicts && k < g_n_evicts)
                    printf("    [%zu] %s l%d e%d | %s l%d e%d%s\n", k,
                           o_evicts[k].kind ? "PILOT" : "DEMAND", o_evicts[k].layer, o_evicts[k].eid,
                           g_evicts[k].kind ? "PILOT" : "DEMAND", g_evicts[k].layer, g_evicts[k].eid,
                           k == ev_mismatch_at ? "  <-- FIRST MISMATCH" : "");
                else
                    printf("    [%zu] oracle:%s sim:%s\n", k, k < o_n_evicts ? "yes" : "no", k < g_n_evicts ? "yes" : "no");
            }
        }
    } else printf("  victim sequence                      EXACT (%zu evictions, same order)\n", g_n_evicts);
    if (fp_o != fp_s) { printf("  final resident fingerprint           DIFF (%llx vs %llx)\n", fp_s, fp_o); fail = 1; }
    else printf("  final resident fingerprint           EXACT (%llx)\n", fp_o);

    printf("VERDICT: %s\n", fail ? "ORACLE_MISMATCH" : "COUNTERFACTUAL_CURRENT_POLICY_REPRODUCED");
    return fail ? 1 : 0;
}
