/* Forge F1 QWEN worker-lifecycle + telemetry-contract gates (2026-08-24).
 *
 * Two defect classes are pinned here:
 *
 * 1. Worker ownership. pilot workers / spec loader used to be detached
 *    infinite threads dereferencing a global Model* past main's return —
 *    the exit-time SIGSEGV the independent verifier reproduced (2/10
 *    baseline, 1/10 candidate). They are now joinable, observe a stop
 *    flag before claiming new work, and background_workers_shutdown()
 *    is an idempotent join boundary called on every normal exit path.
 *
 * 2. Telemetry truth. Under Forge, physical admission accounting lives in
 *    the shared ExpertStore; engine-side legacy counters observe nothing
 *    on normal runs and must never print a misleading zero or a fake
 *    pilot load. tm_report() must agree EXACTLY with the store's own
 *    stats() output.
 *
 * No model file needed: the lifecycle probes drive the real thread code,
 * and the telemetry contract drives tm_report() against a stub store whose
 * stats are known constants.
 */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include <unistd.h>

static int fails = 0;
static void ck(int cond, const char *what) {
    if (cond) { printf("  ok   %s\n", what); return; }
    printf("  FAIL %s\n", what);
    fails++;
}

/* ---- stderr capture at the fd level (code under test writes to fd 2) --- */
static char cap_path[] = "/tmp/q36_lifecycle_capXXXXXX";
static int cap_fd = -1;
static int cap_saved = -1;

static void cap_begin(void) {
    fflush(stderr);
    if (cap_fd < 0) {
        int t = mkstemp(cap_path);
        ck(t >= 0, "capture temp file created");
        cap_fd = t;
    }
    if (lseek(cap_fd, 0, SEEK_SET) < 0) {}
    if (ftruncate(cap_fd, 0) < 0) {}
    cap_saved = dup(STDERR_FILENO);
    dup2(cap_fd, STDERR_FILENO);
}
static void cap_end(char *buf, size_t bufsz) {
    fflush(stderr);
    dup2(cap_saved, STDERR_FILENO);
    close(cap_saved);
    cap_saved = -1;
    off_t n = lseek(cap_fd, 0, SEEK_END);
    if ((size_t)n > bufsz - 1) n = bufsz - 1;
    lseek(cap_fd, 0, SEEK_SET);
    ssize_t got = read(cap_fd, buf, (size_t)n);
    if (got < 0) got = 0;
    buf[got] = 0;
}

/* ---- stub store with known stats for the telemetry contract ---- */
static ColiExpertStoreStats g_stub_stats;
static void stub_stats(const ColiExpertStore *s, ColiExpertStoreStats *out) {
    (void)s;
    *out = g_stub_stats;
}
static ColiExpertStore g_stub_store;
static ColiExpertStoreOps g_stub_ops = { .stats = stub_stats };

static int contains(const char *hay, const char *needle) {
    return strstr(hay, needle) != NULL;
}

int main(void) {
    char buf[16384];

    /* ================= lifecycle: start / idempotency / join ================= */
    printf("[lifecycle]\n");
    setenv("COLI_PILOT_W", "3", 1);

    static Model m;   /* pointer identity only: no jobs are ever executed */
    memset(&m, 0, sizeof(m));

    ck(g_pilot_thr_n == 0 && pilot_m == NULL && !g_workers_stop,
       "initial state: no workers, no model, stop flag clear");

    ensure_pilot_worker_started(&m);
    ck(g_pilot_thr_n == 3, "COLI_PILOT_W=3 honored exactly (3 joinable workers)");
    ck(pilot_m == &m, "worker ownership bound to the calling Model*");

    ensure_pilot_worker_started(&m);   /* second call must be a no-op */
    ensure_pilot_worker_started(&m);
    ck(g_pilot_thr_n == 3, "startup idempotent (still 3 workers after re-ensure)");

    background_workers_shutdown();
    ck(g_workers_stop == 1, "shutdown sets the stop flag");

    /* after join, NO thread may consume queue state ever again: enqueue and
     * prove pilot_r never moves (a live worker would claim within ~1 ms) */
    unsigned r_before = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
    unsigned w_before = __atomic_load_n(&pilot_w, __ATOMIC_ACQUIRE);
    for (int i = 0; i < 8; i++) {
        pilot_q[(w_before + (unsigned)i) & 4095].l = 0;
        pilot_q[(w_before + (unsigned)i) & 4095].e = i;
    }
    __atomic_store_n(&pilot_w, w_before + 8, __ATOMIC_RELEASE);
    usleep(200000);   /* 200 ms >> the 1 ms idle poll */
    unsigned r_after = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
    ck(r_after == r_before,
       "queue state cannot be consumed after shutdown begins (Mission B)");

    cap_begin();
    background_workers_shutdown();          /* second overall call: must be silent */
    cap_end(buf, sizeof(buf));
    ck(!contains(buf, "[workers] shutdown"),
       "second shutdown is a silent no-op (idempotence)");

    /* ================= telemetry contract vs store stats ================= */
    printf("[telemetry]\n");
    memset(&g_stub_stats, 0, sizeof(g_stub_stats));
    g_stub_store.ops = &g_stub_ops;
    g_xstore_report = &g_stub_store;
    g_demand_loads = 0; g_pilot_loads = 0; g_spec_issued = 0;
    cap_begin();
    tm_report();
    cap_end(buf, sizeof(buf));
    ck(contains(buf, "[expert_store] physical_loads 0"),
       "zero physical loads prints zero (contract 1)");
    ck(!contains(buf, "legacy demand sync-loads"),
       "defaults-off run prints no fake demand load line (contract 5)");
    ck(!contains(buf, "spec-shadow private-slot loads"),
       "defaults-off run prints no fake spec-shadow line (contract 5)");

    /* Contract 6 (agreement): one authoritative source, byte-agreed numbers. */
    memset(&g_stub_stats, 0, sizeof(g_stub_stats));
    g_stub_stats.requests = 100;
    g_stub_stats.logical_requests = 120;
    g_stub_stats.hits = 70;
    g_stub_stats.misses = 30;
    g_stub_stats.coalesced_requests = 20;
    g_stub_stats.physical_loads = 7;
    g_stub_stats.admitted_bytes = 42u * 1048576u;
    g_stub_stats.admitted_weight_bytes = 40u * 1048576u;
    g_stub_stats.admitted_scale_bytes = 2u * 1048576u;
    g_stub_stats.resident_bytes = 16u * 1048576u;
    g_stub_stats.capacity_bytes = 64u * 1048576u;
    g_stub_stats.publishes = 7;
    g_stub_stats.aborts = 1;
    g_stub_stats.reservations_active = 3;
    g_stub_stats.pinned_count = 5;
    g_stub_stats.prefetch_requests = 11;
    cap_begin();
    tm_report();
    cap_end(buf, sizeof(buf));
    ck(contains(buf, "[expert_store] requests 100 (logical 120) | hits 70 misses 30 | coalesced 20"),
       "request/hit/miss/coalesced lines agree with stats() (contract 6)");
    ck(contains(buf, "[expert_store] physical_loads 7 | admitted 42.00 MB"),
       "physical loads + admitted bytes agree with stats() (contract 6)");
    ck(contains(buf, "publishes 7 aborts 1 reservations_active 3 pinned 5 prefetch_requests 11"),
       "residency/publish/abort/pin lines agree with stats() (contract 6)");

    /* nonzero shadow counter must surface under its TRUTHFUL label only */
    g_pilot_loads = 4; g_spec_issued = 9;
    cap_begin();
    tm_report();
    cap_end(buf, sizeof(buf));
    ck(contains(buf, "[expert_io] spec-shadow private-slot loads: 4"),
       "shadow staging reported as spec-shadow, not as store admission");
    ck(contains(buf, "not store admissions"),
       "shadow line carries its not-authoritative disclaimer");
    g_pilot_loads = 0; g_spec_issued = 0;

    /* ================= repeat stop/start stress (bounded) ================= */
    printf("[stress]\n");
    {
        /* full start->stop cycles on fresh globals are exercised by the
         * repeated-exit harness; here we pin bounded re-entry safety of the
         * flag protocol itself. */
        int ok = 1;
        for (int i = 0; i < 200; i++) {
            background_workers_shutdown();   /* must never block or crash */
            if (!g_workers_stop) { ok = 0; break; }
        }
        ck(ok, "200x repeated shutdown calls stay non-blocking and flagged");
    }

    if (fails) { fprintf(stderr, "test_qwen36_lifecycle: %d FAILURES\n", fails); return 1; }
    puts("test_qwen36_lifecycle: ok");
    close(cap_fd);
    unlink(cap_path);
    return 0;
}
