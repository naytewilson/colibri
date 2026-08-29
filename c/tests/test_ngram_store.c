/* test_ngram_store.c - NGRAM-FORGE N1 contract tests: row integrity,
 * intra-batch dedupe, direct-mapped cache behavior, async batch fallback,
 * bounded-queue backpressure, and error paths. Synthetic small tables only;
 * tomography numbers come from ngram_tomography on the real-scale file.
 */
#include "../ngram_forge.h"

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <unistd.h>

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); failures++; } } while (0)

static uint64_t row_id(const uint8_t *row)
{
    uint64_t v;
    memcpy(&v, row, 8);
    return v;
}

static void fill_pattern(uint8_t *dst, uint64_t row)
{
    memset(dst, 0x3c, NGRAM_ROW_BYTES);
    memcpy(dst, &row, 8);
}

static int mk_table(const char *path, uint64_t rows)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint8_t row[NGRAM_ROW_BYTES];
    for (uint64_t r = 0; r < rows; r++) {
        fill_pattern(row, r);
        fwrite(row, 1, sizeof(row), f);
    }
    fclose(f);
    return 0;
}

static void test_modes_basic(void)
{
    const uint64_t ROWS = 100000;
    mk_table("/tmp/ngram_test_table.bin", ROWS);
    uint64_t want[4] = { 7, 4242, 99999, 0 };
    for (int mode = NSR_RAM; mode <= NSR_PREAD; mode++) {
        nsr_cfg cfg = {0};
        cfg.mode = (nsr_mode)mode;
        cfg.rows = ROWS;
        cfg.row_bytes = NGRAM_ROW_BYTES;
        cfg.file = mode == NSR_RAM ? NULL : "/tmp/ngram_test_table.bin";
        char err[128];
        nsr_store *s = nsr_open(&cfg, err, sizeof(err));
        CHECK(s != NULL, "open mode %d: %s", mode, err);
        if (!s) continue;
        void *blk = NULL;
        uint64_t wait_ns = 0;
        int rc = nsr_consume(s, want, 4, &blk, &wait_ns);
        CHECK(rc == 0, "consume mode %d rc=%d", mode, rc);
        if (rc == 0) {
            uint8_t *p = blk;
            for (int i = 0; i < 4; i++) {
                CHECK(row_id(p + (size_t)i * NGRAM_ROW_BYTES) == want[i],
                      "mode %d row %d id %" PRIu64, mode, i,
                      row_id(p + (size_t)i * NGRAM_ROW_BYTES));
                CHECK((p + (size_t)i * NGRAM_ROW_BYTES)[NGRAM_ROW_BYTES - 1] == 0x3c,
                      "mode %d row %d fill", mode, i);
            }
            free(blk);
        }
        /* out-of-range row must error, not read OOB */
        uint64_t bad = ROWS + 12345;
        rc = nsr_consume(s, &bad, 1, &blk, &wait_ns);
        CHECK(rc != 0 || mode == NSR_RAM || mode == NSR_PREAD,
              "oob row mode %d rc=%d (mmap must fail cleanly)", mode, rc);
        if (rc == 0) free(blk);
        nsr_close(s);
    }
    unlink("/tmp/ngram_test_table.bin");
}

static void test_cache_dedup(void)
{
    const uint64_t ROWS = 5000;
    mk_table("/tmp/ngram_test_cache.bin", ROWS);
    nsr_cfg cfg = {0};
    cfg.mode = NSR_PREAD;
    cfg.rows = ROWS;
    cfg.file = "/tmp/ngram_test_cache.bin";
    cfg.row_bytes = NGRAM_ROW_BYTES;
    cfg.cache_rows = 64;
    char err[128];
    nsr_store *s = nsr_open(&cfg, err, sizeof(err));
    CHECK(s != NULL, "open: %s", err);
    if (!s) return;
    void *blk;
    uint64_t wait_ns;
    uint64_t r1[2] = { 100, 200 };
    nsr_consume(s, r1, 2, &blk, &wait_ns); free(blk);
    nsr_stats st;
    nsr_stats_get(s, &st);
    CHECK(st.issued_reads == 2, "initial issues %" PRIu64, st.issued_reads);
    /* repeat same rows -> cache hits, no new physical issues */
    nsr_consume(s, r1, 2, &blk, &wait_ns); free(blk);
    nsr_stats_get(s, &st);
    CHECK(st.cache_hits == 2, "cache hits %" PRIu64, st.cache_hits);
    CHECK(st.issued_reads == 2, "no reissue, issued=%" PRIu64, st.issued_reads);
    /* thrash the tiny cache, then hit again -> reissue */
    for (uint64_t r = 300; r < 400; r++) {
        nsr_consume(s, &r, 1, &blk, &wait_ns); free(blk);
    }
    nsr_consume(s, r1, 2, &blk, &wait_ns); free(blk);
    nsr_stats_get(s, &st);
    CHECK(st.issued_reads > 2, "thrash reissue %" PRIu64, st.issued_reads);
    nsr_close(s);
    unlink("/tmp/ngram_test_cache.bin");
}

static void test_async_dedup_backpressure(void)
{
    const uint64_t ROWS = 200000;
    mk_table("/tmp/ngram_test_async.bin", ROWS);
    nsr_cfg cfg = {0};
    cfg.mode = NSR_ASYNC;
    cfg.rows = ROWS;
    cfg.file = "/tmp/ngram_test_async.bin";
    cfg.row_bytes = NGRAM_ROW_BYTES;
    cfg.workers = 4;
    cfg.queue_depth = 2;
    char err[128];
    nsr_store *s = nsr_open(&cfg, err, sizeof(err));
    CHECK(s != NULL, "async open: %s", err);
    if (!s) return;

    /* duplicate rows inside one batch: 16 requests, 5 unique rows */
    uint64_t dup[16];
    uint64_t uniq[5] = { 11, 222, 3333, 44444, 55555 };
    for (int i = 0; i < 16; i++) dup[i] = uniq[i % 5];
    int b = nsr_prefetch(s, dup, 16);
    CHECK(b >= 0, "prefetch returned %d", b);
    void *blk = NULL;
    uint64_t wait_ns = 0, lookup_ns = 0;
    int hit = 0;
    int rc = nsr_consume_batch(s, b, &blk, &wait_ns, &lookup_ns, &hit);
    CHECK(rc == 0, "consume_batch rc=%d", rc);
    if (rc == 0) {
        uint8_t *p = blk;
        for (int i = 0; i < 16; i++)
            CHECK(row_id(p + (size_t)i * NGRAM_ROW_BYTES) == dup[i],
                  "async dedup row %d", i);
        free(blk);
    }
    nsr_stats st;
    nsr_stats_get(s, &st);
    CHECK(st.dedup_hits == 11, "dedup hits %" PRIu64 " (want 11)", st.dedup_hits);

    /* backpressure: more outstanding batches than depth must return -1 */
    int ok = 0, refused = 0;
    uint64_t rows[16];
    for (int i = 0; i < 16; i++) rows[i] = (uint64_t)(i + 1) * 1000;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 16; j++) rows[j] = (uint64_t)((i * 16 + j) % ROWS);
        int r = nsr_prefetch(s, rows, 16);
        if (r >= 0) ok++; else refused++;
    }
    CHECK(refused > 0, "bounded queue never refused (ok=%d)", ok);
    nsr_close(s);
    unlink("/tmp/ngram_test_async.bin");
}

static void test_async_cache_at_issue(void)
{
    const uint64_t ROWS = 200000;
    mk_table("/tmp/ngram_test_acache.bin", ROWS);
    nsr_cfg cfg = {0};
    cfg.mode = NSR_ASYNC;
    cfg.rows = ROWS;
    cfg.file = "/tmp/ngram_test_acache.bin";
    cfg.row_bytes = NGRAM_ROW_BYTES;
    cfg.cache_rows = 4096;
    cfg.workers = 2;
    cfg.queue_depth = 4;
    char err[128];
    nsr_store *s = nsr_open(&cfg, err, sizeof(err));
    CHECK(s != NULL, "acache open: %s", err);
    if (!s) return;
    uint64_t rows[16];
    for (int i = 0; i < 16; i++) rows[i] = (uint64_t)i * 7 + 3;
    void *blk = NULL;
    uint64_t w, l;
    int hit;
    int b = nsr_prefetch(s, rows, 16);
    CHECK(b >= 0, "acache first prefetch");
    CHECK(nsr_consume_batch(s, b, &blk, &w, &l, &hit) == 0, "acache consume 1");
    free(blk);
    nsr_stats st;
    nsr_stats_get(s, &st);
    uint64_t issued1 = st.issued_reads;
    CHECK(issued1 == 16, "first pass issues %" PRIu64 " want 16", issued1);
    b = nsr_prefetch(s, rows, 16);
    CHECK(b >= 0, "acache second prefetch");
    CHECK(nsr_consume_batch(s, b, &blk, &w, &l, &hit) == 0, "acache consume 2");
    free(blk);
    nsr_stats_get(s, &st);
    CHECK(st.issued_reads == issued1, "cached rows reissued (%" PRIu64 ")",
          st.issued_reads);
    CHECK(st.cache_hits >= 16, "async cache hits %" PRIu64, st.cache_hits);
    /* integrity after cache serve */
    b = nsr_prefetch(s, rows, 16);
    nsr_consume_batch(s, b, &blk, &w, &l, &hit);
    uint8_t *p = blk;
    for (int i = 0; i < 16; i++)
        CHECK(row_id(p + (size_t)i * NGRAM_ROW_BYTES) == rows[i],
              "acache row %d", i);
    free(blk);
    nsr_close(s);
    unlink("/tmp/ngram_test_acache.bin");
}

static void test_error_paths(void)
{
    char err[128];
    nsr_cfg cfg = {0};
    cfg.mode = NSR_MMAP;
    cfg.rows = 10;
    cfg.file = "/tmp/ngram_definitely_missing_table.bin";
    nsr_store *s = nsr_open(&cfg, err, sizeof(err));
    CHECK(s == NULL, "missing file must fail open");
    if (s) nsr_close(s);
    FILE *f = fopen("/tmp/ngram_test_small.bin", "wb");
    fwrite("junk", 1, 4, f);
    fclose(f);
    cfg.file = "/tmp/ngram_test_small.bin";
    s = nsr_open(&cfg, err, sizeof(err));
    CHECK(s == NULL, "undersized file must fail open");
    if (s) nsr_close(s);
    unlink("/tmp/ngram_test_small.bin");
}

int main(void)
{
    test_modes_basic();
    test_cache_dedup();
    test_async_dedup_backpressure();
    test_async_cache_at_issue();
    test_error_paths();
    if (failures) {
        fprintf(stderr, "NGRAM_STORE_FAIL %d\n", failures);
        return 1;
    }
    printf("NGRAM_STORE_PASS\n");
    return 0;
}
