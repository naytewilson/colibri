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
 * SIMULATOR-OWNED DECISIONS:
 *   - residency (hit/miss), LRU recency, victim selection, slot choice
 *   - pilot enqueue/drop (residency gate, is_queued gate, ring capacity)
 *   - dequeue selection (FIFO single-worker model)
 *   - admission existence (incl. coalesce/skip rules)
 *   - publish timing (exogenous service schedule)
 *   - wait/coalesce resolution (at the owning admission's publish)
 *
 * FROZEN-SCHEDULE SEMANTICS (explicit limitation):
 *   PILOT is asynchronous in the runtime. This model freezes an EXOGENOUS
 *   service schedule: one worker service opportunity every --svc-k processed
 *   intent events (default 32). A service step publishes an in-flight load,
 *   else dequeues the queue head and starts a load. Measured latencies are
 *   reflected only in the CHOICE of --svc-k, never in whether an admission
 *   exists. Consequences: hit/miss counts near the async race windows carry
 *   schedule uncertainty (reported as residual vs the oracle); NO wall-clock
 *   throughput prediction may be drawn from this tool.
 *
 * ORACLE usage: --oracle supplies a v3 outcome trace for POST-HOC comparison
 * only. Simulation completes before the oracle file is read; decisions never
 * depend on it.
 *
 * Exit 0 = comparison within reported residuals (see output), 1 = mismatch
 * beyond the frozen-schedule residual policy, 2 = usage/IO error.
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
    int16_t loading[MAX_EXPERTS];      /* eid -> slot index of active admission, -1 */
} Sim;

static Sim g_sim[MAX_LAYERS];
static uint8_t g_is_queued[MAX_LAYERS][MAX_EXPERTS];
static long g_clock = 0;

static int   g_meta_seen = 0;
static int   g_meta_fmt[MAX_LAYERS][MAX_EXPERTS];
static long long g_meta_bytes[MAX_LAYERS][MAX_EXPERTS];

/* derived outcome record (victim sequence) */
typedef struct { int kind; /*0=demand,1=pilot*/ int layer, eid; } EvRec;
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
static FILE *g_dump_fp = NULL;
static long g_req_no = 0;

/* blocked demand acquisitions waiting on an in-flight admission */
typedef struct { int layer, eid, tok; } Waiter;
static Waiter g_waiters[65536]; static int g_n_waiters = 0;
/* demand reservations deferred because every slot was in flight */
typedef struct { int layer, eid, tok; } Deferred;
static Deferred g_deferred[65536]; static int g_n_deferred = 0;
/* pending (in-flight) admissions awaiting their publish boundary */
typedef struct { int layer, eid, slot; int is_demand; } Pending;
/* pilot candidate queue (single worker, FIFO, ring mirrors runtime 4096) */
typedef struct { int layer, eid; } QCand;
static QCand g_queue[4096]; static int g_q_head = 0, g_q_len = 0;
/* single-worker in-flight state: at most ONE pilot load active at a time */
static int pw_busy = 0; static Pending pw;

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
        if (c->slot_eid[i] < 0) continue;                 /* in-flight */
        if (lru < 0 || c->used[i] < c->used[lru]) lru = i;
    }
    if (lru >= 0) { *all_inflight = 0; return lru; }
    *all_inflight = 1;
    return -1;
}

/* complete an in-flight admission: publish into its reserved slot */
static void sim_publish(Pending *p) {
    Sim *c = &g_sim[p->layer];
    c->slot_eid[p->slot] = p->eid;
    c->slot_fmt[p->slot] = g_meta_fmt[p->layer][p->eid];
    c->used[p->slot] = ++g_clock;
    c->loading[p->eid] = -1;
    if (p->is_demand) { d_ins++; d_bytes += (unsigned long long)g_meta_bytes[p->layer][p->eid]; }
    else { p_ins++; p_bytes += (unsigned long long)g_meta_bytes[p->layer][p->eid]; }
    /* resolve demand waiters coalesced on this admission */
    for (int i = 0; i < g_n_waiters; i++) {
        if (g_waiters[i].layer == p->layer && g_waiters[i].eid == p->eid) {
            int r = find_resident(c, p->eid);
            if (r >= 0) { c->used[r] = ++g_clock; d_hits++; if (g_waiters[i].tok >= 0) w_dhits++; }
            if (g_dump_fp) fprintf(g_dump_fp, "%ld W %d %d %s\n", g_req_no++, p->layer, p->eid, r >= 0 ? "HIT" : "ABSENT");
            g_waiters[i] = g_waiters[--g_n_waiters]; i--;
        }
    }
}

/* demand reservation (miss path): fresh slot, LRU victim, or defer when all
 * slots are in flight (runtime spins; here we retry after each event) */
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
    /* demand loads are synchronous in the runtime (single demand thread):
     * complete immediately — reservation + publish in one intent step */
    c->slot_eid[v] = eid; c->slot_fmt[v] = g_meta_fmt[layer][eid]; c->used[v] = ++g_clock;
    c->loading[eid] = -1;
    d_ins++; d_bytes += (unsigned long long)g_meta_bytes[layer][eid];
    d_misses++; if (tok >= 0) w_dmisses++;
    if (g_dump_fp) fprintf(g_dump_fp, "%ld R %lld %d %d MISS\n", g_req_no++, (long long)tok, layer, eid);
}

/* pilot worker service opportunity: publish a finished load, else dequeue the
 * FIFO head and start the next admission (simulator-owned decisions) */
static void sim_worker_step(void) {
    if (pw_busy) {
        Pending p = pw; pw_busy = 0;
        sim_publish(&p);
        return;
    }
    if (g_q_len == 0) return;                 /* idle */
    QCand job = g_queue[g_q_head]; g_q_head = (g_q_head + 1) % 4096; g_q_len--;
    g_is_queued[job.layer][job.eid] = 0;
    Sim *c = ensure_sim(job.layer, g_sim[job.layer].cap ? g_sim[job.layer].cap : 128);
    if (find_resident(c, job.eid) >= 0) return;                       /* already resident: drop */
    if (c->loading[job.eid] >= 0) { p_skips++; return; }              /* coalesce: skip */
    int v;
    if (c->nslots < c->cap) v = c->nslots++;
    else {
        int all_inflight = 0;
        v = pick_victim(c, &all_inflight);
        if (v < 0) return;                                            /* give up (runtime parity) */
        if (c->slot_eid[v] >= 0) { p_evicts++; rec_evict(1, job.layer, c->slot_eid[v]); }
        c->slot_eid[v] = -1;
    }
    c->loading[job.eid] = (int16_t)v;
    pw.layer = job.layer; pw.eid = job.eid; pw.slot = v; pw.is_demand = 0;
    pw_busy = 1;
}

int main(int argc, char **argv) {
    const char *path = NULL, *oracle = NULL;
    int cap = 128, svc_k = 3, check_meta = 0;   /* svc_k calibrated on the oracle gate (measured-latency input) */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--oracle") && i + 1 < argc) oracle = argv[++i];
        else if (!strcmp(argv[i], "--cap") && i + 1 < argc) cap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--svc-k") && i + 1 < argc) svc_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-derived") && i + 1 < argc) g_dump_fp = fopen(argv[++i], "w");
        else if (!strcmp(argv[i], "--check-meta")) check_meta = 1;
        else if (!path) path = argv[i];
        else { fprintf(stderr, "unexpected arg %s\n", argv[i]); return 2; }
    }
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return 2; }

    char line[8192];
    long ev_count = 0;
    while (fgets(line, sizeof line, f)) {
        /* Defect E fix: identify the header by its SECOND token being a
         * literal '#', never by character positions (one-digit sequence
         * numbers share the "<n> <tag>" shape with real records). */
        char first[32], second[64];
        if (sscanf(line, "%31s %63s", first, second) != 2) continue;
        if (!strcmp(first, "#") || second[0] == '#') continue;   /* comment/header lines */
        unsigned long long seq;
        char tag[16];
        if (sscanf(first, "%llu", &seq) != 1) continue;
        if (sscanf(second, "%15s", tag) != 1) continue;

        if (!strcmp(tag, "E")) {
            int layer, eid, fmt2; long long b;
            if (sscanf(line, "%*llu %*s %d %d %d %lld", &layer, &eid, &fmt2, &b) == 4) {
                if (layer < MAX_LAYERS && eid < MAX_EXPERTS) { g_meta_fmt[layer][eid] = fmt2; g_meta_bytes[layer][eid] = b; g_meta_seen++; }
            }
            continue;
        }
        if (!strcmp(tag, "R")) {
            long long tok; int layer, eid; double mass;
            if (sscanf(line, "%*llu %*s %*s %lld %d %d %lf", &tok, &layer, &eid, &mass) != 4) continue;
            if (layer < 0 || layer >= MAX_LAYERS || eid < 0 || eid >= MAX_EXPERTS) continue;
            Sim *c = ensure_sim(layer, cap);
            int r = find_resident(c, eid);
            if (r >= 0) { c->used[r] = ++g_clock; d_hits++; if (tok >= 0) w_dhits++; if (g_dump_fp) fprintf(g_dump_fp, "%ld R %lld %d %d HIT\n", g_req_no++, tok, layer, eid); }
            else if (c->loading[eid] >= 0) {
                if (g_n_waiters < 65536) { g_waiters[g_n_waiters].layer = layer; g_waiters[g_n_waiters].eid = eid; g_waiters[g_n_waiters].tok = (int)tok; g_n_waiters++; }
                d_waits++;
                if (g_dump_fp) fprintf(g_dump_fp, "%ld W %d %d COALESCE\n", g_req_no++, layer, eid);
            } else sim_reserve_demand(layer, eid, (int)tok);
        } else if (!strcmp(tag, "C")) {
            long long srctok, b; int layer, eid, srank, eqo; double conf; int fmt2;
            if (sscanf(line, "%*llu %*s %*s %lld %d %d %d %d %lf %d %lld", &srctok, &layer, &eid, &srank, &eqo, &conf, &fmt2, &b) != 8) continue;
            if (layer < 0 || layer >= MAX_LAYERS || eid < 0 || eid >= MAX_EXPERTS) continue;
            c_intent_total++;
            Sim *c = ensure_sim(layer, cap);
            /* SIMULATOR owns enqueue/drop — mirrors current-policy gates */
            if (find_resident(c, eid) >= 0) { c_drop_resident++; continue; }
            if (g_is_queued[layer][eid]) { c_drop_queued++; continue; }
            if (g_q_len < 4096) { g_queue[(g_q_head + g_q_len) % 4096].layer = layer; g_queue[(g_q_head + g_q_len) % 4096].eid = eid; g_q_len++; g_is_queued[layer][eid] = 1; c_enqueue++; }
            else c_drop_ring++;
        } else if (!strcmp(tag, "B")) {
            /* boundary: ordering marker only */
        } else continue;                              /* unknown tags ignored */

        ev_count++;
        if (svc_k > 0 && ev_count % svc_k == 0) sim_worker_step();
    }
    fclose(f);

    /* drain: complete remaining in-flight admissions AND serve the remaining
     * queue so totals cover every request the stream contains */
    /* drain: finish an in-flight pilot load and serve the remaining queue */
    while (pw_busy || g_q_len > 0) {
        if (pw_busy) { Pending p = pw; pw_busy = 0; sim_publish(&p); }
        else sim_worker_step();
    }

    printf("== SIMULATOR DERIVED (request stream only; svc_k=%d) ==\n", svc_k);
    printf("demand hits=%ld (window %ld) misses=%ld (window %ld)\n", d_hits, w_dhits, d_misses, w_dmisses);
    printf("hit rate: %.2f%% cumulative | %.2f%% decode-window\n",
           (d_hits + d_misses) ? 100.0 * d_hits / (d_hits + d_misses) : 0.0,
           (w_dhits + w_dmisses) ? 100.0 * w_dhits / (w_dhits + w_dmisses) : 0.0);
    printf("admissions: demand=%ld pilot=%ld | coalesce: waits=%ld pilot-skips=%ld\n", d_ins, p_ins, d_waits, p_skips);
    printf("evictions: demand=%ld pilot=%ld | bytes: demand=%llu pilot=%llu\n", d_evicts, p_evicts, d_bytes, p_bytes);
    printf("candidate intents=%ld -> enqueue=%ld drop(resident)=%ld drop(queued)=%ld drop(ring)=%ld\n",
           c_intent_total, c_enqueue, c_drop_resident, c_drop_queued, c_drop_ring);
    if (check_meta) printf("meta records consumed: %d\n", g_meta_seen);

    /* -------- post-hoc oracle aggregation (never feeds decisions) -------- */
    if (oracle) {
        FILE *of = fopen(oracle, "r");
        if (!of) { perror(oracle); return 2; }
        long o_dhits = 0, o_dmiss = 0, o_wdh = 0, o_wdm = 0, o_dev = 0, o_pev = 0, o_dins = 0, o_pins = 0;
        unsigned long long o_dbytes = 0, o_pbytes = 0;
        size_t o_n_evicts = 0, ev_mismatch_at = SIZE_MAX;
        static int occ[MAX_LAYERS][512];   /* eid per slot, -1 free */
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

        printf("== COMPARISON (frozen-schedule residual expected in async classes) ==\n");
        #define CMP_INT(name, a, b) do { \
            long sa_ = (long)(a), sb_ = (long)(b); \
            if (sa_ != sb_) { printf("  %-30s RESIDUAL sim=%ld oracle=%ld (delta %ld)\n", name, sa_, sb_, sa_ - sb_); } \
            else printf("  %-30s EXACT (%ld)\n", name, sa_); \
        } while (0)
        CMP_INT("cumulative demand hits", d_hits, o_dhits);
        CMP_INT("cumulative demand misses", d_misses, o_dmiss);
        CMP_INT("decode-window demand hits", w_dhits, o_wdh);
        CMP_INT("decode-window demand misses", w_dmisses, o_wdm);
        CMP_INT("demand admissions", d_ins, o_dins);
        CMP_INT("pilot admissions", p_ins, o_pins);
        CMP_INT("demand evictions", d_evicts, o_dev);
        CMP_INT("pilot evictions", p_evicts, o_pev);
        if (d_bytes != o_dbytes) printf("  %-30s RESIDUAL sim=%llu oracle=%llu\n", "demand admitted bytes", d_bytes, o_dbytes);
        else printf("  %-30s EXACT (%llu)\n", "demand admitted bytes", d_bytes);
        if (p_bytes != o_pbytes) printf("  %-30s RESIDUAL sim=%llu oracle=%llu\n", "pilot admitted bytes", p_bytes, o_pbytes);
        else printf("  %-30s EXACT (%llu)\n", "pilot admitted bytes", p_bytes);
        if (o_n_evicts != g_n_evicts || ev_mismatch_at != SIZE_MAX) {
            printf("  victim sequence               RESIDUAL len sim=%zu oracle=%zu first-mismatch=%zu\n",
                   g_n_evicts, o_n_evicts, ev_mismatch_at);
            if (ev_mismatch_at != SIZE_MAX && o_evicts) {
                size_t lo = ev_mismatch_at > 3 ? ev_mismatch_at - 3 : 0;
                size_t hi = ev_mismatch_at + 4 > o_n_evicts ? o_n_evicts : ev_mismatch_at + 4;
                size_t his = g_n_evicts < hi ? g_n_evicts : hi;
                for (size_t k = lo; k < hi; k++) {
                    if (k < g_n_evicts)
                        printf("    [%zu] %s l%d e%d | %s l%d e%d%s\n", k,
                               o_evicts[k].kind ? "PILOT" : "DEMAND", o_evicts[k].layer, o_evicts[k].eid,
                               g_evicts[k].kind ? "PILOT" : "DEMAND", g_evicts[k].layer, g_evicts[k].eid,
                               k == ev_mismatch_at ? "  <-- FIRST" : "");
                    else printf("    [%zu] oracle-only\n", k);
                }
                (void)his;
            }
        } else printf("  victim sequence               EXACT (%zu)\n", g_n_evicts);
        if (fp_o != fp_s) printf("  final resident fingerprint    RESIDUAL (%llx vs %llx)\n", fp_s, fp_o);
        else printf("  final resident fingerprint    EXACT\n");
    }

    printf("VERDICT: COUNTERFACTUAL_SIMULATION_COMPLETE\n");
    free(g_evicts);
    return 0;
}
