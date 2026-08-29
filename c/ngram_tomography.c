/* ngram_tomography.c - NGRAM-FORGE N1 storage tomography for deterministic
 * sparse n-gram row access. Replays reference-derived token traces through
 * the exact N0 addressing and measures the plan's measurement set at the
 * point the model consumes gathered rows.
 *
 * subcommands:
 *   gen     --in CORPUS --out TRACE [--eos-every N]
 *   replay  --trace TRACE --mode ram|mmap|pread|async [--table FILE]
 *           [--rows N] [--tokens K] [--compute-us X] [--cache-rows C]
 *           [--workers W] [--depth D] [--pass cold|warm]
 *   bulk    --trace TRACE --table FILE --order token|row [--tokens K]
 *           [--pass cold|warm]   (prefill gather locality, no overlap)
 */
#define _GNU_SOURCE
#include "ngram_forge.h"

#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "oom %zu\n", n); exit(1); }
    return p;
}

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : (x > y);
}

static uint64_t pct(const uint64_t *sorted, size_t n, double p)
{
    if (!n) return 0;
    size_t i = (size_t)(p * (double)(n - 1) + 0.5);
    return sorted[i];
}

/* ---------------- gen ---------------- */

static uint64_t fnv1a(const char *w, size_t n)
{
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char)w[i];
        h *= 1099511628211ull;
    }
    return h;
}

static int cmd_gen(int argc, char **argv)
{
    const char *in = NULL, *out = NULL;
    int eos_every = 64;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc) in = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--eos-every") && i + 1 < argc) eos_every = atoi(argv[++i]);
    }
    if (!in || !out) { fprintf(stderr, "gen: --in --out\n"); return 2; }
    FILE *f = fopen(in, "rb");
    if (!f) { perror("open in"); return 1; }
    FILE *o = fopen(out, "wb");
    if (!o) { perror("open out"); return 1; }
    int c, wlen = 0;
    char word[256];
    int since_eos = 0;
    long ntok = 0;
    while ((c = fgetc(f)) != EOF) {
        int alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') || c == '_';
        if (alpha && wlen < 255) { word[wlen++] = (char)c; continue; }
        if (wlen) {
            uint64_t id = fnv1a(word, (size_t)wlen) % 248319u + 1;
            fprintf(o, "%" PRIu64 "\n", id);
            ntok++;
            since_eos++;
            wlen = 0;
        }
        if ((c == '\n' || c == '.') && since_eos >= eos_every) {
            fprintf(o, "%lld\n", (long long)NGRAM_EOS_TOKEN_ID);
            ntok++;
            since_eos = 0;
        }
    }
    if (wlen) {
        uint64_t id = fnv1a(word, (size_t)wlen) % 248319u + 1;
        fprintf(o, "%" PRIu64 "\n", id);
        ntok++;
    }
    fclose(f);
    fclose(o);
    fprintf(stderr, "gen tokens=%ld out=%s\n", ntok, out);
    return 0;
}

static int64_t *load_trace(const char *path, size_t *n_out, size_t cap)
{
    FILE *f = fopen(path, "r");
    if (!f) { perror("trace"); exit(1); }
    size_t capn = cap ? cap : (64u << 20);
    int64_t *t = xmalloc(capn * sizeof(int64_t));
    size_t n = 0;
    long long v;
    while (n < capn && fscanf(f, "%lld", &v) == 1) t[n++] = v;
    fclose(f);
    *n_out = n;
    return t;
}

/* ---------------- replay ---------------- */

static int cmd_replay(int argc, char **argv)
{
    const char *trace = NULL, *table = NULL, *mode_s = "mmap", *pass = "cold";
    nsr_mode mode;
    uint64_t rows = 320001536ull;
    size_t tokens = 0;
    uint64_t compute_us = 0;
    uint32_t cache_rows = 0;
    int workers = 4, depth = 8;
    uint64_t repeat = 1;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--trace") && i + 1 < argc) trace = argv[++i];
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode_s = argv[++i];
        else if (!strcmp(argv[i], "--table") && i + 1 < argc) table = argv[++i];
        else if (!strcmp(argv[i], "--rows") && i + 1 < argc) rows = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = (size_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--compute-us") && i + 1 < argc) compute_us = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--cache-rows") && i + 1 < argc) cache_rows = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--workers") && i + 1 < argc) workers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--depth") && i + 1 < argc) depth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pass") && i + 1 < argc) pass = argv[++i];
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = strtoull(argv[++i], NULL, 10);
    }
    if (!trace) { fprintf(stderr, "replay: --trace required\n"); return 2; }
    if (!strcmp(mode_s, "ram")) mode = NSR_RAM;
    else if (!strcmp(mode_s, "mmap")) mode = NSR_MMAP;
    else if (!strcmp(mode_s, "pread")) mode = NSR_PREAD;
    else if (!strcmp(mode_s, "async")) mode = NSR_ASYNC;
    else { fprintf(stderr, "bad mode\n"); return 2; }
    if (mode != NSR_RAM && !table) { fprintf(stderr, "file modes need --table\n"); return 2; }

    ngram_params p;
    ngram_params_flash_next(&p);
    if (rows > p.padded_rows) rows = p.padded_rows;

    size_t nt = 0;
    int64_t *tok = load_trace(trace, &nt, tokens);
    if (tokens && tokens < nt) nt = tokens;
    if (nt < 2) { fprintf(stderr, "trace too short\n"); return 1; }

    nsr_cfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.mode = mode;
    cfg.file = mode == NSR_RAM ? NULL : table;
    cfg.rows = mode == NSR_RAM ? (1ull << 30) : rows;
    cfg.row_bytes = NGRAM_ROW_BYTES;
    cfg.cache_rows = cache_rows;
    cfg.workers = workers;
    cfg.queue_depth = depth;
    char err[160];
    nsr_store *s = nsr_open(&cfg, err, sizeof(err));
    if (!s) { fprintf(stderr, "nsr_open: %s\n", err); return 1; }

    int evict_first = !strcmp(pass, "cold") && mode != NSR_RAM;
    if (evict_first) nsr_evict(s);

    uint64_t *wait_s = xmalloc(nt * sizeof(uint64_t));
    uint64_t *cons_s = xmalloc(nt * sizeof(uint64_t));
    uint64_t *addr_s = xmalloc(nt * sizeof(uint64_t));
    uint64_t *allrows = xmalloc(nt * (size_t)p.n_heads * sizeof(uint64_t));
    uint64_t *rows_cur = xmalloc((size_t)p.n_heads * sizeof(uint64_t));
    uint64_t *rows_nxt = xmalloc((size_t)p.n_heads * sizeof(uint64_t));

    struct rusage ru0, ru1;
    uint64_t t_wall = 0;
    uint64_t prefetched = 0, sync_fallback = 0;
    size_t e = (size_t)-1;
    nsr_stats_reset(s);
    for (uint64_t it = 0; it < repeat; it++) {
    /* steady-state semantics: reported pass is the FINAL iteration;
     * earlier iterations warm caches exactly like a long decode session */
    e = (size_t)-1;
    prefetched = sync_fallback = 0;
    getrusage(RUSAGE_SELF, &ru0);
    t_wall = 0;
    if (it > 0) nsr_stats_reset(s);
    uint64_t t_wall0 = now_ns();
    int batch_held = -1; /* batch carrying the CURRENT token's rows */
    for (size_t t = 0; t < nt; t++) {
        int64_t w[3];
        for (int j = 0; j < 3; j++) {
            if ((int64_t)t - j < 0 || (e != (size_t)-1 && t - (size_t)j <= e))
                w[j] = p.eos;
            else
                w[j] = tok[t - (size_t)j];
        }
        uint64_t a0 = now_ns();
        ngram_rows_window(&p, w, rows_cur);
        addr_s[t] = now_ns() - a0;
        memcpy(allrows + t * (size_t)p.n_heads, rows_cur,
               (size_t)p.n_heads * sizeof(uint64_t));

        /* consume token t's rows: batch issued at t-1 if async, else sync */
        uint64_t wait_ns = 0, lookup_ns = 0;
        int hit = 0;
        void *blk = NULL;
        int prc;
        if (batch_held >= 0) {
            int b = batch_held;
            batch_held = -1;
            prc = nsr_consume_batch(s, b, &blk, &wait_ns, &lookup_ns, &hit);
        } else {
            prc = nsr_consume(s, rows_cur, p.n_heads, &blk, &wait_ns);
        }
        if (prc != 0) { fprintf(stderr, "consume failed at t=%zu\n", t); return 1; }
        if (mode != NSR_RAM) {
            for (int h = 0; h < p.n_heads; h++) {
                uint64_t id;
                memcpy(&id, (uint8_t *)blk + (size_t)h * NGRAM_ROW_BYTES, 8);
                if (id != rows_cur[h]) {
                    fprintf(stderr, "ROW CORRUPTION t=%zu h=%d got=%" PRIu64
                            " want=%" PRIu64 "\n", t, h, id, rows_cur[h]);
                    return 1;
                }
            }
        }
        free(blk);
        wait_s[t] = wait_ns;
        cons_s[t] = lookup_ns ? lookup_ns : wait_ns;
        if (tok[t] == p.eos) e = t;

        /* token t selected: prefetch t+1's rows, then one unit of useful
         * compute while the workers fetch them */
        if (mode == NSR_ASYNC && t + 1 < nt) {
            size_t e2 = e;
            int64_t w2[3];
            for (int j = 0; j < 3; j++) {
                if (t + 1 < (size_t)j || (e2 != (size_t)-1 && t + 1 - (size_t)j <= e2))
                    w2[j] = p.eos;
                else
                    w2[j] = tok[t + 1 - (size_t)j];
            }
            ngram_rows_window(&p, w2, rows_nxt);
            batch_held = nsr_prefetch(s, rows_nxt, p.n_heads);
            if (batch_held >= 0) prefetched++; else sync_fallback++;
        }
        if (compute_us) {
            uint64_t spin_until = now_ns() + compute_us * 1000ull;
            volatile double acc = 0;
            while (now_ns() < spin_until) { acc += 1.0; if (acc > 1e18) acc = 0; }
        }
    }
    t_wall = now_ns() - t_wall0;
    getrusage(RUSAGE_SELF, &ru1);
    }

    qsort(wait_s, nt, sizeof(uint64_t), cmp_u64);
    qsort(cons_s, nt, sizeof(uint64_t), cmp_u64);
    qsort(addr_s, nt, sizeof(uint64_t), cmp_u64);
    nsr_stats st;
    nsr_stats_get(s, &st);
    uint64_t total_rows = nt * (uint64_t)p.n_heads;
    qsort(allrows, total_rows, sizeof(uint64_t), cmp_u64);
    uint64_t uniq = 0;
    for (uint64_t i = 0; i < total_rows; i++)
        if (i == 0 || allrows[i] != allrows[i - 1]) uniq++;
    int cached_pct = nsr_cached_fraction(s, allrows,
                                         (int)(uniq > 4096 ? 4096 : uniq));
    double wait_sum = 0;
    for (size_t t = 0; t < nt; t++) wait_sum += (double)wait_s[t];

    printf("pass=%s mode=%s table=%s rows=%llu tokens=%zu compute_us=%llu repeat=%llu "
           "cache_rows=%u workers=%d depth=%d\n",
           pass, mode_s, table ? table : "-", (unsigned long long)rows, nt,
           (unsigned long long)compute_us, (unsigned long long)repeat, cache_rows, workers, depth);
    printf("rows_per_token=%d unique_rows=%llu repeated_row_rate=%.4f unique_per_token=%.2f\n",
           p.n_heads, (unsigned long long)uniq,
           1.0 - (double)uniq / (double)total_rows, (double)uniq / (double)nt);
    printf("wall_s=%.3f bytes_read=%" PRIu64 " bytes_per_token=%.0f issued_reads=%" PRIu64 "\n",
           (double)t_wall / 1e9, st.bytes_read,
           (double)st.bytes_read / (double)nt, st.issued_reads);
    printf("faults_minor=%ld faults_major=%ld phys_read_bytes_est=%ld\n",
           ru1.ru_minflt - ru0.ru_minflt, ru1.ru_majflt - ru0.ru_majflt,
           (ru1.ru_majflt - ru0.ru_majflt) * 4096L);
    printf("cache_hits=%" PRIu64 " cache_hit_rate=%.4f batch_hit_rate=%.4f "
           "prefetch_issued=%llu sync_fallback=%llu lookups=%" PRIu64 "\n",
           st.cache_hits, (double)st.cache_hits / (double)(st.lookups ? st.lookups : 1),
           (double)st.batch_hits / (double)(st.batches ? st.batches : 1),
           (unsigned long long)prefetched, (unsigned long long)sync_fallback, st.lookups);
    printf("addr_ns p50=%" PRIu64 " p95=%" PRIu64 " p99=%" PRIu64 "\n",
           pct(addr_s, nt, .50), pct(addr_s, nt, .95), pct(addr_s, nt, .99));
    printf("ngram_wait_ns p50=%" PRIu64 " p95=%" PRIu64 " p99=%" PRIu64 " mean=%.0f\n",
           pct(wait_s, nt, .50), pct(wait_s, nt, .95), pct(wait_s, nt, .99),
           wait_sum / (double)nt);
    printf("consume_ns p50=%" PRIu64 " p95=%" PRIu64 " p99=%" PRIu64 "\n",
           pct(cons_s, nt, .50), pct(cons_s, nt, .95), pct(cons_s, nt, .99));
    printf("exposed_wait_pct_of_wall=%.2f sampled_cached_frac_pct=%d\n",
           100.0 * wait_sum / (double)t_wall, cached_pct);

    nsr_close(s);
    free(tok); free(wait_s); free(cons_s); free(addr_s); free(allrows);
    free(rows_cur); free(rows_nxt);
    return 0;
}

/* ---------------- bulk (prefill gather locality) ---------------- */

static int cmd_bulk(int argc, char **argv)
{
    const char *trace = NULL, *table = NULL, *order = "row", *pass = "cold";
    size_t tokens = 0;
    uint64_t rows = 320001536ull;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--trace") && i + 1 < argc) trace = argv[++i];
        else if (!strcmp(argv[i], "--table") && i + 1 < argc) table = argv[++i];
        else if (!strcmp(argv[i], "--order") && i + 1 < argc) order = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--rows") && i + 1 < argc) rows = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--pass") && i + 1 < argc) pass = argv[++i];
    }
    if (!trace || !table) { fprintf(stderr, "bulk: --trace --table\n"); return 2; }
    ngram_params p;
    ngram_params_flash_next(&p);
    size_t nt = 0;
    int64_t *tok = load_trace(trace, &nt, tokens);
    if (tokens && tokens < nt) nt = tokens;

    uint64_t total = nt * (size_t)p.n_heads;
    uint64_t *rows_stream = xmalloc(total * sizeof(uint64_t));
    ngram_rows_seq(&p, tok, nt, rows_stream); /* exact stream-order rows */

    uint64_t *sorted = xmalloc(total * sizeof(uint64_t));
    memcpy(sorted, rows_stream, total * sizeof(uint64_t));
    qsort(sorted, total, sizeof(uint64_t), cmp_u64);
    uint64_t uniq = 0;
    for (uint64_t i = 0; i < total; i++)
        if (i == 0 || sorted[i] != sorted[i - 1]) sorted[uniq++] = sorted[i];

    uint64_t *work = xmalloc(uniq * sizeof(uint64_t));
    if (!strcmp(order, "row")) {
        memcpy(work, sorted, uniq * sizeof(uint64_t)); /* = page-grouped */
    } else {
        /* token-appearance order */
        char *done = calloc(1, uniq);
        size_t si = 0;
        for (uint64_t i = 0; i < total && si < uniq; i++) {
            uint64_t r = rows_stream[i];
            uint64_t *pos = bsearch(&r, sorted, uniq, sizeof(uint64_t), cmp_u64);
            size_t idx = (size_t)(pos - sorted);
            if (!done[idx]) { done[idx] = 1; work[si++] = r; }
        }
        free(done);
    }

    nsr_cfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.mode = NSR_PREAD;
    cfg.file = table;
    cfg.rows = rows;
    cfg.row_bytes = NGRAM_ROW_BYTES;
    char err[160];
    nsr_store *st = nsr_open(&cfg, err, sizeof(err));
    if (!st) { fprintf(stderr, "open: %s\n", err); return 1; }
    if (!strcmp(pass, "cold")) nsr_evict(st);

    struct rusage ru0, ru1;
    uint64_t rep = 1;
    for (int i = 0; i < argc; i++)
        if (!strcmp(argv[i], "--repeat") && i + 1 < argc) rep = (uint64_t)strtoull(argv[i + 1], NULL, 10);
    if (rep < 1) rep = 1;
    if (!strcmp(pass, "warm") && rep == 1) rep = 50; /* steady-state bulk */
    double wall = 0; long majf = 0, minf = 0;
    for (uint64_t it = 0; it < rep; it++) {
    getrusage(RUSAGE_SELF, &ru0);
    uint64_t t0 = now_ns();
    void *blk = NULL;
    for (uint64_t i = 0; i < uniq; i += 16) {
        int n = uniq - i < 16 ? (int)(uniq - i) : 16;
        if (nsr_consume(st, work + i, n, &blk, NULL) != 0) {
            fprintf(stderr, "bulk consume failed\n");
            return 1;
        }
        free(blk);
    }
    uint64_t t1 = now_ns();
    getrusage(RUSAGE_SELF, &ru1);
    wall += (double)(t1 - t0) / 1e9;
    majf += ru1.ru_majflt - ru0.ru_majflt;
    minf += ru1.ru_minflt - ru0.ru_minflt;
    }
    printf("bulk order=%s pass=%s repeat=%u uniq_rows=%llu tokens=%zu wall_s=%.3f "
           "ns_per_row=%.0f majflt=%ld minflt=%ld\n",
           order, pass, (unsigned)rep, (unsigned long long)uniq, nt,
           wall, wall * 1e9 / (double)(uniq * rep),
           majf, minf);
    nsr_close(st);
    free(tok); free(sorted); free(work); free(rows_stream);
    return 0;
}

static void usage(void)
{
    fputs("ngram_tomography - NGRAM-FORGE N1 storage tomography\n"
          "usage: ngram_tomography gen --in CORPUS --out TRACE --eos-every N\n"
          "   or: ngram_tomography replay --trace TRACE --mode ram|mmap|pread|async\n"
          "        --table FILE --rows N --tokens K --compute-us X --cache-rows C\n"
          "        --workers W --depth D --pass cold|warm\n"
          "   or: ngram_tomography bulk --trace TRACE --table FILE\n"
          "        --order token|row --rows N --tokens K --pass cold|warm\n"
          "flags: --in --out --eos-every --trace --mode --table --rows --tokens\n"
          "       --compute-us --cache-rows --workers --depth --pass --order --repeat\n",
          stdout);
}

int main(int argc, char **argv)
{
    if (argc < 2 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
        usage();
        return argc < 2 ? 2 : 0;
    }
    if (!strcmp(argv[1], "gen")) return cmd_gen(argc - 2, argv + 2);
    if (!strcmp(argv[1], "replay")) return cmd_replay(argc - 2, argv + 2);
    if (!strcmp(argv[1], "bulk")) return cmd_bulk(argc - 2, argv + 2);
    fprintf(stderr, "unknown subcommand %s\n", argv[1]);
    return 2;
}
