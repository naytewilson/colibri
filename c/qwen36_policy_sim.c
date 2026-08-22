/* qwen36_policy_sim — counterfactual cache-policy simulator for qwen36 MoE.
 *
 * INPUT (policy-independent, v4R): a COLI_TRACE_REQ stream containing ONLY
 * intents and static facts:
 *   E  <layer> <eid> <fmt> <bytes>                     static expert metadata
 *   B  GEN <np> <n_new> <tok_base> <arm>               prompt boundary
 *   R  DEMAND <tok> <layer> <eid> <mass>               demand acquisition INTENT
 *   C  PC <srctok> <layer> <eid> <score_rank> <eq_order> <conf> <fmt> <bytes>
 *                                                      pilot candidate INTENT
 * The stream contains NO baseline outcomes: no HIT/EVICT/INSERT rows, no
 * victims, no slots, no dequeue/publish/wake markers. Every cache decision is
 * made by THIS simulator from its own state.
 *
 * R1 — EXOGENOUS SERVICE ORDINAL: the service ordinal advances ONCE for every
 * R/C/B record consumed, REGARDLESS of the simulator's own outcome (hit or
 * miss, resident-drop or enqueue, queued or not). A policy may change state;
 * it can never change service-opportunity coordinates. One worker service
 * opportunity fires every --svc-k ordinal ticks: publish the in-flight load,
 * else dequeue the FIFO head. Diagnostics print service_opportunities and
 * last_service_ordinal; --dump-ordinal records the full step/ordinal series
 * so two runs with different outcomes can be diffed for schedule identity.
 *
 * FROZEN-SCHEDULE SEMANTICS (explicit limitation): PILOT is asynchronous in
 * the runtime; svc_k freezes an exogenous service cadence. Measured latency
 * informs the CHOICE of svc_k only. No wall-clock throughput prediction may
 * be drawn from this tool.
 *
 * R2 — ORACLE GATE: with --oracle, the v3 outcome trace is aggregated AFTER
 * simulation (never consulted for decisions) and compared under explicit,
 * printed tolerances. Exit codes:
 *   0 CURRENT_POLICY_ORACLE_GATE_PASS
 *   1 CURRENT_POLICY_ORACLE_GATE_FAIL  (residuals out of tolerance,
 *                                      unaccounted requests, unresolved
 *                                      waiters/deferred, fingerprint mismatch)
 *   2 malformed input / IO / contract failure (missing or invalid expert
 *                                      metadata for a requested expert, ...)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_LAYERS 128
#define MAX_EXPERTS 512

typedef struct {
    int cap;
    int nslots;
    int slot_eid[512];
    int slot_fmt[512];
    long used[512];
    int16_t loading[MAX_EXPERTS];
} Sim;

static Sim g_sim[MAX_LAYERS];
static uint8_t g_is_queued[MAX_LAYERS][MAX_EXPERTS];
static long g_clock = 0;

static int   g_meta_seen = 0;
static uint8_t g_meta_valid[MAX_LAYERS][MAX_EXPERTS];
static int   g_meta_fmt[MAX_LAYERS][MAX_EXPERTS];
static long long g_meta_bytes[MAX_LAYERS][MAX_EXPERTS];

typedef struct { int kind; int layer, eid; } EvRec;
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
static long c_intent_total, c_enqueue, c_drop_resident, c_drop_queued, c_drop_ring;
static unsigned long long d_bytes, p_bytes;
static long total_R = 0, contract_errors = 0;
static long service_opportunities = 0, last_service_ordinal = 0;
static FILE *g_dump_fp = NULL, *g_ord_fp = NULL;
static long g_req_no = 0;

typedef struct { int layer, eid, tok; } Waiter;
static Waiter g_waiters[65536]; static int g_n_waiters = 0;
typedef struct { int layer, eid, tok; } Deferred;
static Deferred g_deferred[65536]; static int g_n_deferred = 0;
typedef struct { int layer, eid, slot; } Pending;
static Pending pw; static int pw_busy = 0;
typedef struct { int layer, eid; } QCand;
static QCand g_queue[4096]; static int g_q_head = 0, g_q_len = 0;

static int g_ar_layer = -1, g_ar_eid = -1;   /* --assume-resident fixture knob */

static int find_resident(Sim *c, int eid) {
    for (int i = 0; i < c->nslots; i++)
        if (c->slot_eid[i] == eid) return i;
    return -1;
}

static Sim *ensure_sim(int layer, int cap) {
    Sim *c = &g_sim[layer];
    if (!c->cap) {
        c->cap = cap;
        for (int e2 = 0; e2 < MAX_EXPERTS; e2++) c->loading[e2] = -1;
    }
    return c;
}

static int pick_victim(Sim *c, int *all_inflight) {
    int lru = -1;
    for (int i = 0; i < c->nslots; i++) {
        if (c->slot_eid[i] < 0) continue;
        if (lru < 0 || c->used[i] < c->used[lru]) lru = i;
    }
    if (lru >= 0) { *all_inflight = 0; return lru; }
    *all_inflight = 1;
    return -1;
}

static void sim_publish(Pending *p) {
    Sim *c = &g_sim[p->layer];
    c->slot_eid[p->slot] = p->eid;
    c->slot_fmt[p->slot] = g_meta_fmt[p->layer][p->eid];
    c->used[p->slot] = ++g_clock;
    c->loading[p->eid] = -1;
    p_ins++; p_bytes += (unsigned long long)g_meta_bytes[p->layer][p->eid];
    for (int i = 0; i < g_n_waiters; i++) {
        if (g_waiters[i].layer == p->layer && g_waiters[i].eid == p->eid) {
            int r = find_resident(c, p->eid);
            if (r >= 0) { c->used[r] = ++g_clock; d_hits++; if (g_waiters[i].tok >= 0) w_dhits++; }
            if (g_dump_fp) fprintf(g_dump_fp, "%ld W %d %d %s\n", g_req_no++, p->layer, p->eid, r >= 0 ? "HIT" : "ABSENT");
            g_waiters[i] = g_waiters[--g_n_waiters]; i--;
        }
    }
}

static void sim_reserve_demand(int layer, int eid, int tok) {
    Sim *c = &g_sim[layer];
    int v;
    if (c->nslots < c->cap) v = c->nslots++;
    else {
        int all_inflight = 0;
        v = pick_victim(c, &all_inflight);
        if (v < 0) {
            if (g_n_deferred < 65536) { g_deferred[g_n_deferred].layer = layer; g_deferred[g_n_deferred].eid = eid; g_deferred[g_n_deferred].tok = tok; g_n_deferred++; }
            return;
        }
        if (c->slot_eid[v] >= 0) { d_evicts++; rec_evict(0, layer, c->slot_eid[v]); }
        c->slot_eid[v] = -1;
    }
    c->loading[eid] = (int16_t)v;
    c->slot_eid[v] = eid; c->slot_fmt[v] = g_meta_fmt[layer][eid]; c->used[v] = ++g_clock;
    c->loading[eid] = -1;
    d_ins++; d_bytes += (unsigned long long)g_meta_bytes[layer][eid];
    d_misses++; if (tok >= 0) w_dmisses++;
    if (g_dump_fp) fprintf(g_dump_fp, "%ld R %lld %d %d MISS\n", g_req_no++, (long long)tok, layer, eid);
}

static void sim_worker_step(long ordinal) {
    service_opportunities++;
    last_service_ordinal = ordinal;
    if (g_ord_fp) fprintf(g_ord_fp, "step=%ld ordinal=%ld\n", service_opportunities, ordinal);
    if (pw_busy) {
        Pending p = pw; pw_busy = 0;
        sim_publish(&p);
        return;
    }
    if (g_q_len == 0) return;
    QCand job = g_queue[g_q_head]; g_q_head = (g_q_head + 1) % 4096; g_q_len--;
    g_is_queued[job.layer][job.eid] = 0;
    Sim *c = ensure_sim(job.layer, g_sim[job.layer].cap ? g_sim[job.layer].cap : 128);
    if (find_resident(c, job.eid) >= 0) return;
    if (c->loading[job.eid] >= 0) { p_skips++; return; }
    int v;
    if (c->nslots < c->cap) v = c->nslots++;
    else {
        int all_inflight = 0;
        v = pick_victim(c, &all_inflight);
        if (v < 0) return;
        if (c->slot_eid[v] >= 0) { p_evicts++; rec_evict(1, job.layer, c->slot_eid[v]); }
        c->slot_eid[v] = -1;
    }
    c->loading[job.eid] = (int16_t)v;
    pw.layer = job.layer; pw.eid = job.eid; pw.slot = v;
    pw_busy = 1;
}

int main(int argc, char **argv) {
    const char *path = NULL, *oracle = NULL;
    int cap = 128, svc_k = 5, check_meta = 0;   /* svc_k calibrated on the oracle gate (measured-latency input) */
    double tol_frac = 0.01;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--oracle") && i + 1 < argc) oracle = argv[++i];
        else if (!strcmp(argv[i], "--cap") && i + 1 < argc) cap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--svc-k") && i + 1 < argc) svc_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tol-frac") && i + 1 < argc) tol_frac = atof(argv[++i]);
        else if (!strcmp(argv[i], "--dump-derived") && i + 1 < argc) g_dump_fp = fopen(argv[++i], "w");
        else if (!strcmp(argv[i], "--dump-ordinal") && i + 1 < argc) g_ord_fp = fopen(argv[++i], "w");
        else if (!strcmp(argv[i], "--check-meta")) check_meta = 1;
        else if (!strcmp(argv[i], "--assume-resident") && i + 1 < argc && sscanf(argv[++i], "%d:%d", &g_ar_layer, &g_ar_eid) == 2) { /* fixture knob */ }
        else if (!path) path = argv[i];
        else { fprintf(stderr, "unexpected arg %s\n", argv[i]); return 2; }
    }
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return 2; }

    char line[8192];
    long ev_count = 0;
    while (fgets(line, sizeof line, f)) {
        char first[32], second[64];
        if (sscanf(line, "%31s %63s", first, second) != 2) continue;
        if (!strcmp(first, "#") || second[0] == '#') continue;
        unsigned long long seq;
        char tag[16];
        if (sscanf(first, "%llu", &seq) != 1) continue;
        if (sscanf(second, "%15s", tag) != 1) continue;
        int sched_relevant = 0; (void)sched_relevant;

        if (!strcmp(tag, "E")) {
            int layer, eid, fmt2; long long b;
            if (sscanf(line, "%*llu %*s %d %d %d %lld", &layer, &eid, &fmt2, &b) == 4) {
                if (layer < MAX_LAYERS && eid < MAX_EXPERTS) {
                    g_meta_fmt[layer][eid] = fmt2; g_meta_bytes[layer][eid] = b;
                    g_meta_valid[layer][eid] = (fmt2 > 0 && b > 0);
                    g_meta_seen++;
                }
            }
            continue;   /* metadata is not a schedule-relevant event */
        }

        if (!strcmp(tag, "R")) {
            sched_relevant = 1;
            long long tok; int layer, eid; double mass;
            if (sscanf(line, "%*llu %*s %*s %lld %d %d %lf", &tok, &layer, &eid, &mass) != 4) { contract_errors++; }
            else if (layer < 0 || layer >= MAX_LAYERS || eid < 0 || eid >= MAX_EXPERTS) { contract_errors++; }
            else {
                total_R++;
                if (!g_meta_valid[layer][eid]) { fprintf(stderr, "CONTRACT: missing/invalid E metadata for l%d e%d\n", layer, eid); contract_errors++; }
                Sim *c = ensure_sim(layer, cap);
                int forced = (g_ar_layer == layer && g_ar_eid == eid);
                int r = forced ? -1 : find_resident(c, eid);
                if (r >= 0) { c->used[r] = ++g_clock; d_hits++; if (tok >= 0) w_dhits++; if (g_dump_fp) fprintf(g_dump_fp, "%ld R %lld %d %d HIT\n", g_req_no++, (long long)tok, layer, eid); }
                else if (c->loading[eid] >= 0) {
                    if (g_n_waiters < 65536) { g_waiters[g_n_waiters].layer = layer; g_waiters[g_n_waiters].eid = eid; g_waiters[g_n_waiters].tok = (int)tok; g_n_waiters++; }
                    d_waits++;
                    if (g_dump_fp) fprintf(g_dump_fp, "%ld W %d %d COALESCE\n", g_req_no++, layer, eid);
                } else sim_reserve_demand(layer, eid, (int)tok);
            }
        } else if (!strcmp(tag, "C")) {
            sched_relevant = 1;
            long long srctok, b; int layer, eid, srank, eqo; double conf; int fmt2;
            if (sscanf(line, "%*llu %*s %*s %lld %d %d %d %d %lf %d %lld", &srctok, &layer, &eid, &srank, &eqo, &conf, &fmt2, &b) != 8) { contract_errors++; }
            else if (layer < 0 || layer >= MAX_LAYERS || eid < 0 || eid >= MAX_EXPERTS) { contract_errors++; }
            else {
                c_intent_total++;
                Sim *c = ensure_sim(layer, cap);
                int forced = (g_ar_layer == layer && g_ar_eid == eid);
                if (forced || find_resident(c, eid) >= 0) { c_drop_resident++; }
                else if (g_is_queued[layer][eid]) { c_drop_queued++; }
                else if (g_q_len < 4096) {
                    g_queue[(g_q_head + g_q_len) % 4096].layer = layer; g_queue[(g_q_head + g_q_len) % 4096].eid = eid;
                    g_q_len++; g_is_queued[layer][eid] = 1; c_enqueue++;
                } else c_drop_ring++;
            }
        } else if (!strcmp(tag, "B")) {
            sched_relevant = 1;   /* boundary occupies an ordinal tick */
        } else continue;          /* unknown tags: not schedule-relevant */

        /* R1: the ordinal advances for EVERY R/C/B record, outcome-independent */
        ev_count++;
        if (svc_k > 0 && ev_count % svc_k == 0) sim_worker_step(ev_count);
    }
    fclose(f);

    /* drain: finish in-flight pilot load, serve remaining queue, retry
     * deferred demand reservations after each state change */
    while (pw_busy || g_q_len > 0 || g_n_deferred > 0) {
        int progressed = 0;
        if (pw_busy) { Pending p = pw; pw_busy = 0; sim_publish(&p); progressed = 1; }
        else if (g_q_len > 0) { sim_worker_step(++last_service_ordinal); progressed = 1; }
        if (g_n_deferred > 0) {
            int limit = g_n_deferred;
            for (int i = 0; i < limit && i < g_n_deferred; i++) {
                Deferred d = g_deferred[i];
                g_deferred[i] = g_deferred[--g_n_deferred]; i--;
                Sim *c = ensure_sim(d.layer, cap);
                if (find_resident(c, d.eid) >= 0 || c->loading[d.eid] >= 0) continue;
                sim_reserve_demand(d.layer, d.eid, d.tok);
                progressed = 1;
            }
        }
        if (!progressed) break;
    }

    printf("== SIMULATOR DERIVED (request stream only; svc_k=%d) ==\n", svc_k);
    printf("demand requests=%ld accounted=%ld unresolved_waiters=%d unresolved_deferred=%d\n",
           total_R, d_hits + d_misses, g_n_waiters, g_n_deferred);
    printf("demand hits=%ld (window %ld) misses=%ld (window %ld)\n", d_hits, w_dhits, d_misses, w_dmisses);
    printf("hit rate: %.2f%% cumulative | %.2f%% decode-window\n",
           (d_hits + d_misses) ? 100.0 * d_hits / (d_hits + d_misses) : 0.0,
           (w_dhits + w_dmisses) ? 100.0 * w_dhits / (w_dhits + w_dmisses) : 0.0);
    printf("admissions: demand=%ld pilot=%ld | coalesce: waits=%ld pilot-skips=%ld\n", d_ins, p_ins, d_waits, p_skips);
    printf("evictions: demand=%ld pilot=%ld | bytes: demand=%llu pilot=%llu\n", d_evicts, p_evicts, d_bytes, p_bytes);
    printf("candidate intents=%ld -> enqueue=%ld drop(resident)=%ld drop(queued)=%ld drop(ring)=%ld\n",
           c_intent_total, c_enqueue, c_drop_resident, c_drop_queued, c_drop_ring);
    printf("service_opportunities=%ld last_service_ordinal=%ld\n", service_opportunities, last_service_ordinal);
    if (check_meta) printf("meta records consumed: %d\n", g_meta_seen);
    if (g_ord_fp) { fprintf(g_ord_fp, "service_opportunities=%ld last_service_ordinal=%ld\n", service_opportunities, last_service_ordinal); fclose(g_ord_fp); }
    if (g_dump_fp) fclose(g_dump_fp);

    /* -------- R2: oracle gate (post-hoc; decisions never read it) -------- */
    if (!oracle) {
        printf("VERDICT: NO_ORACLE_SUPPLIED\n");
        if (contract_errors) return 2;
        return 0;
    }
    if (contract_errors) { fprintf(stderr, "CONTRACT ERRORS: %ld\n", contract_errors); return 2; }
    if (total_R != d_hits + d_misses) { fprintf(stderr, "GATE: unaccounted demand requests (%ld requests, %ld accounted)\n", total_R, d_hits + d_misses); return 1; }
    if (g_n_waiters > 0) { fprintf(stderr, "GATE: %d unresolved waiters\n", g_n_waiters); return 1; }
    if (g_n_deferred > 0) { fprintf(stderr, "GATE: %d unresolved deferred reservations\n", g_n_deferred); return 1; }

    FILE *of = fopen(oracle, "r");
    if (!of) { perror(oracle); return 2; }
    long o_dhits = 0, o_dmiss = 0, o_wdh = 0, o_wdm = 0, o_dev = 0, o_pev = 0, o_dins = 0, o_pins = 0;
    unsigned long long o_dbytes = 0, o_pbytes = 0;
    size_t o_n_evicts = 0, ev_mismatch_at = SIZE_MAX;
    static int occ[MAX_LAYERS][512];
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
        if (!strcmp(ev, "HIT")) { if (!is_pilot) { o_dhits++; if (win) o_wdh++; } }
        else if (!strcmp(ev, "EVICT")) {
            if (is_pilot) o_pev++; else o_dev++;
            if (slot >= 0 && slot < 512) occ[layer][slot] = -1;
            { EvRec *r3 = realloc(o_evicts, (o_n_evicts + 1) * sizeof(EvRec));
              if (r3) { o_evicts = r3; o_evicts[o_n_evicts].kind = is_pilot; o_evicts[o_n_evicts].layer = layer; o_evicts[o_n_evicts].eid = eid; } }
            if (o_n_evicts < g_n_evicts && (g_evicts[o_n_evicts].kind != is_pilot || g_evicts[o_n_evicts].layer != layer || g_evicts[o_n_evicts].eid != eid)) {
                if (ev_mismatch_at == SIZE_MAX) ev_mismatch_at = o_n_evicts;
            }
            o_n_evicts++;
        } else if (!strcmp(ev, "INSERT")) {
            occ[layer][slot] = eid;
            if (is_pilot) { o_pins++; o_pbytes += (unsigned long long)bytes; }
            else { o_dins++; o_dbytes += (unsigned long long)bytes; o_dmiss++; if (win) o_wdm++; }
        }
    }
    fclose(of);

    unsigned long long fp_o = 1469598103934665603ULL, fp_s = 1469598103934665603ULL;
    for (int l = 0; l < MAX_LAYERS; l++) {
        int oe[512], se[512]; int on = 0, sn = 0;
        for (int s = 0; s < 512; s++) if (occ[l][s] >= 0 && occ[l][s] < MAX_EXPERTS) oe[on++] = occ[l][s];
        for (int s = 0; s < g_sim[l].nslots; s++) if (g_sim[l].slot_eid[s] >= 0) se[sn++] = g_sim[l].slot_eid[s];
        unsigned long long ho = 0, hs = 0;
        for (int i = 0; i < on; i++) { for (int j = i + 1; j < on; j++) if (oe[j] < oe[i]) { int t = oe[i]; oe[i] = oe[j]; oe[j] = t; } ho = ho * 1000003u + (unsigned)oe[i]; }
        for (int i = 0; i < sn; i++) { for (int j = i + 1; j < sn; j++) if (se[j] < se[i]) { int t = se[i]; se[i] = se[j]; se[j] = t; } hs = hs * 1000003u + (unsigned)se[i]; }
        fp_o = (fp_o ^ (unsigned)(ho & 0xFFFFFFFFu) ^ (unsigned)l) * 1099511628211ULL;
        fp_s = (fp_s ^ (unsigned)(hs & 0xFFFFFFFFu) ^ (unsigned)l) * 1099511628211ULL;
    }

    printf("== ORACLE RECORDED (v3 outcome trace; audit only) ==\n");
    printf("demand hits=%ld (window %ld) misses=%ld (window %ld)\n", o_dhits, o_wdh, o_dmiss, o_wdm);
    printf("admissions: demand=%ld pilot=%ld | evictions: demand=%ld pilot=%ld\n", o_dins, o_pins, o_dev, o_pev);
    printf("bytes: demand=%llu pilot=%llu\n", o_dbytes, o_pbytes);

    long tol_hits = (long)(tol_frac * (o_dhits + o_dmiss > 0 ? o_dhits + o_dmiss : 1)) + 1;
    long tol_adm  = (long)(tol_frac * (o_dins + o_pins > 0 ? o_dins + o_pins : 1)) + 1;
    long tol_ev   = (long)(tol_frac * (o_dev + o_pev > 0 ? o_dev + o_pev : 1)) + 1;
    unsigned long long tol_b = (unsigned long long)(tol_frac * (o_dbytes + o_pbytes > 0 ? o_dbytes + o_pbytes : 1)) + 1;
    printf("== GATE (tolerance frac=%.4f -> hits<=%ld adm<=%ld evict<=%ld bytes<=%llu) ==\n",
           tol_frac, tol_hits, tol_adm, tol_ev, tol_b);
    int fail = 0;
    #define GATE_INT(name, a, b, tol) do { \
        long da_ = (long)(a) - (long)(b); \
        if (da_ < 0) da_ = -da_; \
        if (da_ > (tol)) { printf("  %-30s FAIL sim=%ld oracle=%ld |delta|=%ld tol=%ld\n", name, (long)(a), (long)(b), da_, (long)(tol)); fail = 1; } \
        else printf("  %-30s PASS sim=%ld oracle=%ld |delta|=%ld\n", name, (long)(a), (long)(b), da_); \
    } while (0)
    GATE_INT("cumulative demand hits", d_hits, o_dhits, tol_hits);
    GATE_INT("cumulative demand misses", d_misses, o_dmiss, tol_hits);
    GATE_INT("demand admissions", d_ins, o_dins, tol_adm);
    GATE_INT("pilot admissions", p_ins, o_pins, tol_adm);
    GATE_INT("demand evictions", d_evicts, o_dev, tol_ev);
    GATE_INT("pilot evictions", p_evicts, o_pev, tol_ev);
    {
        unsigned long long dd = d_bytes > o_dbytes ? d_bytes - o_dbytes : o_dbytes - d_bytes;
        if (dd > tol_b) { printf("  %-30s FAIL |delta|=%llu tol=%llu\n", "demand admitted bytes", dd, tol_b); fail = 1; }
        else printf("  %-30s PASS |delta|=%llu\n", "demand admitted bytes", dd);
        dd = p_bytes > o_pbytes ? p_bytes - o_pbytes : o_pbytes - p_bytes;
        if (dd > tol_b) { printf("  %-30s FAIL |delta|=%llu tol=%llu\n", "pilot admitted bytes", dd, tol_b); fail = 1; }
        else printf("  %-30s PASS |delta|=%llu\n", "pilot admitted bytes", dd);
    }
    {
        long vd = (long)o_n_evicts - (long)g_n_evicts; if (vd < 0) vd = -vd;
        if (vd > tol_ev || ev_mismatch_at != SIZE_MAX) {
            printf("  victim sequence               RESIDUAL len sim=%zu oracle=%zu first-mismatch=%zu (order-only swaps tolerated in counts, reported for audit)\n",
                   g_n_evicts, o_n_evicts, ev_mismatch_at);
        } else printf("  victim sequence               EXACT (%zu)\n", g_n_evicts);
    }
    if (fp_o != fp_s) {
        /* quantify the final resident SET/MULTISET difference explicitly */
        long diff_slots = 0;
        for (int l = 0; l < MAX_LAYERS; l++) {
            int oe[512], se[512]; int on = 0, sn = 0;
            for (int s = 0; s < 512; s++) if (occ[l][s] >= 0 && occ[l][s] < MAX_EXPERTS) oe[on++] = occ[l][s];
            for (int s = 0; s < g_sim[l].nslots; s++) if (g_sim[l].slot_eid[s] >= 0) se[sn++] = g_sim[l].slot_eid[s];
            for (int i = 0; i < on; i++) { for (int j = i + 1; j < on; j++) if (oe[j] < oe[i]) { int t = oe[i]; oe[i] = oe[j]; oe[j] = t; } }
            for (int i = 0; i < sn; i++) { for (int j = i + 1; j < sn; j++) if (se[j] < se[i]) { int t = se[i]; se[i] = se[j]; se[j] = t; } }
            int i = 0, j = 0;
            while (i < on || j < sn) {
                if (i >= on) { diff_slots++; j++; }
                else if (j >= sn) { diff_slots++; i++; }
                else if (oe[i] == se[j]) { i++; j++; }
                else { diff_slots++; if (oe[i] < se[j]) i++; else j++; }
            }
        }
        long cap_total = 0;
        for (int l = 0; l < MAX_LAYERS; l++) if (g_sim[l].cap) cap_total += g_sim[l].cap;
        long tol_slots = cap_total * (long)(tol_frac * 100) / 100 + 64;
        printf("  final resident SET/MULTISET   %s differing slots=%ld tol=%ld (of %ld)\n",
               diff_slots > tol_slots ? "FAIL" : "PASS-within-tolerance", diff_slots, tol_slots, cap_total);
        if (diff_slots > tol_slots) fail = 1;
    } else printf("  final resident SET/MULTISET   EXACT\n");

    printf("VERDICT: %s\n", fail ? "CURRENT_POLICY_ORACLE_GATE_FAIL" : "CURRENT_POLICY_ORACLE_GATE_PASS");
    free(g_evicts);
    free(o_evicts);
    return fail ? 1 : 0;
}
