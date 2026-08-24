/* Forge F1 Phase 5 — telemetry byte-identity pins.
 *
 * The v3 oracle TSV and the v4 request stream are load-bearing formats
 * (oracle fixtures + replay tooling). These tests pin the EXACT bytes the
 * promoted qwen36.c emitters produced, via open_memstream captures.
 */

#include "../expert_telemetry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_fail++;                                                     \
        }                                                                 \
    } while (0)

static char *capture(char **buf, size_t *len) {
    /* open_memstream: POSIX.1-2008; available on macOS + glibc/musl */
    FILE *f = open_memstream(buf, len);
    if (!f) return NULL;
    return f;
}

int main(void) {
    char *buf = NULL;
    size_t len = 0;
    FILE *f = capture(&buf, &len);
    CHECK(f != NULL);
    if (!f) return 2;

    ColiExpertTraceV3 v3 = {NULL, 0};
    coli_expert_telemetry_v3_attach(&v3, f);

    /* disabled stream is a silent no-op */
    ColiExpertTraceV3 off = {NULL, 0};
    coli_expert_telemetry_v3_emit(&off, "DEMAND", "HIT", 1, 0, 0, 8, 0, 0.0, -1, 0);
    CHECK(off.seq == 0);

    /* rows reproduce the promoted trace_emit byte-for-byte */
    coli_expert_telemetry_v3_emit(&v3, "DEMAND", "INSERT", -1, 3, 17, 4,
                                  1769608, 3.907, 42, 1);
    coli_expert_telemetry_v3_emit(&v3, "PILOT", "EVICT", -1, 3, 42, 4, 0, 0.0,
                                  -1, 0);
    coli_expert_telemetry_v3_emit(&v3, "DEMAND", "HIT", 7777, 11, 256, 3, 0,
                                  0.015, -1, 2);

    ColiExpertTraceV4 v4 = {NULL, 0};
    coli_expert_telemetry_v4_attach(&v4, f);
    ColiExpertTraceV4 off4 = {NULL, 0};
    coli_expert_telemetry_v4_emit(&off4, "R", "DEMAND 5 3 17 0.123456");
    CHECK(off4.seq == 0);

    coli_expert_telemetry_v4_emit(&v4, "#",
                                  "qwen36_req_stream v4 cap=128 ep=1 pilot=1 wide=- omp=- snap=x");
    coli_expert_telemetry_v4_emit(&v4, "E", "3 17 4 1769608");
    coli_expert_telemetry_v4_emit(
        &v4, "R", "DEMAND -1 3 17 0.086538");
    coli_expert_telemetry_v4_emit(
        &v4, "C", "PC -1 4 9 3 3 0.204113 4 1769608");

    fclose(f);

    static const char *expect =
        "1\tDEMAND\tINSERT\t-1\t3\t17\t4\t1769608\t3.907\t42\t1\n"
        "2\tPILOT\tEVICT\t-1\t3\t42\t4\t0\t0.000\t-1\t0\n"
        "3\tDEMAND\tHIT\t7777\t11\t256\t3\t0\t0.015\t-1\t2\n"
        "1 # qwen36_req_stream v4 cap=128 ep=1 pilot=1 wide=- omp=- snap=x\n"
        "2 E 3 17 4 1769608\n"
        "3 R DEMAND -1 3 17 0.086538\n"
        "4 C PC -1 4 9 3 3 0.204113 4 1769608\n";
    if (len != strlen(expect) || memcmp(buf, expect, len) != 0) {
        fprintf(stderr, "telemetry byte drift!\n--- got (%zu) ---\n%.*s\n--- want (%zu) ---\n%s\n",
                len, (int)len, buf, strlen(expect), expect);
        g_fail++;
    }

    /* seq continuity across interleaved streams: each carries its own counter */
    CHECK(v3.seq == 3);
    CHECK(v4.seq == 4);

    free(buf);
    if (g_fail) {
        fprintf(stderr, "expert telemetry tests: %d FAILURES\n", g_fail);
        return 1;
    }
    puts("expert telemetry tests: ok");
    return 0;
}
