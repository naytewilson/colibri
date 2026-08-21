#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

#define HIDDEN 2048
#define INTER  512
#define GS     64

#define I3_GROUP 64
#define I3_GBYTES 24

static inline int64_t i3_groups(int64_t n) { return (n + I3_GROUP - 1) / I3_GROUP; }
static inline int64_t i3_rowbytes(int64_t n) { return i3_groups(n) * I3_GBYTES; }

#if defined(__AVX512F__) && defined(__AVX512BW__)
static inline float dot_i3g64_avx512(const uint8_t *lo, const uint8_t *hi, const float *x) {
    const __m128i m2 = _mm_set1_epi8(0x03);
    const __m512i c4 = _mm512_set1_epi8(4);
    __m128i b0 = _mm_loadu_si128((const __m128i*)lo);
    __m128i p0 = _mm_and_si128(b0, m2), p1 = _mm_and_si128(_mm_srli_epi16(b0, 2), m2);
    __m128i p2 = _mm_and_si128(_mm_srli_epi16(b0, 4), m2), p3 = _mm_and_si128(_mm_srli_epi16(b0, 6), m2);
    __m128i l01 = _mm_unpacklo_epi8(p0, p1), h01 = _mm_unpackhi_epi8(p0, p1);
    __m128i l23 = _mm_unpacklo_epi8(p2, p3), h23 = _mm_unpackhi_epi8(p2, p3);
    __m512i lov = _mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(
        _mm512_castsi128_si512(_mm_unpacklo_epi16(l01, l23)),
        _mm_unpackhi_epi16(l01, l23), 1),
        _mm_unpacklo_epi16(h01, h23), 2),
        _mm_unpackhi_epi16(h01, h23), 3);
    uint64_t hb; memcpy(&hb, hi, 8);
    __m512i wq = _mm512_sub_epi8(_mm512_mask_add_epi8(lov, (__mmask64)hb, lov, c4), c4);
    __m512 ac0 = _mm512_setzero_ps(), ac1 = _mm512_setzero_ps();
    ac0 = _mm512_fmadd_ps(_mm512_loadu_ps(x),    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_castsi512_si128(wq))),      ac0);
    ac1 = _mm512_fmadd_ps(_mm512_loadu_ps(x+16), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(wq, 1))), ac1);
    ac0 = _mm512_fmadd_ps(_mm512_loadu_ps(x+32), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(wq, 2))), ac0);
    ac1 = _mm512_fmadd_ps(_mm512_loadu_ps(x+48), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(wq, 3))), ac1);
    return _mm512_reduce_add_ps(_mm512_add_ps(ac0, ac1));
}
#endif

#if defined(__AVX2__)
static inline float dot_i4g64_avx2(const uint8_t *p, const float *x) {
    __m256 ac0 = _mm256_setzero_ps(), ac1 = _mm256_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m256i c8 = _mm256_set1_epi32(8);
    for (int k = 0; k < 64; k += 16) {
        __m128i b = _mm_loadu_si128((const __m128i*)(p + k / 2));
        __m128i lo = _mm_and_si128(b, mask);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), mask);
        __m128i u0 = _mm_unpacklo_epi8(lo, hi);
        __m128i u1 = _mm_unpackhi_epi8(lo, hi);
        __m256i w0 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(u0), c8);
        __m256i w1 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(u0, 8)), c8);
        __m256i w2 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(u1), c8);
        __m256i w3 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(u1, 8)), c8);
        ac0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k), _mm256_cvtepi32_ps(w0), ac0);
        ac1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 8), _mm256_cvtepi32_ps(w1), ac1);
        ac0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 16), _mm256_cvtepi32_ps(w2), ac0);
        ac1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 24), _mm256_cvtepi32_ps(w3), ac1);
        k += 16;
    }
    __m256 s256 = _mm256_add_ps(ac0, ac1);
    __m128 s128 = _mm_add_ps(_mm256_castps256_ps128(s256), _mm256_extractf128_ps(s256, 1));
    s128 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
    s128 = _mm_add_ss(s128, _mm_shuffle_ps(s128, s128, 1));
    return _mm_cvtss_f32(s128);
}
#endif

static void matmul_i3(float *y, const float *x, const uint8_t *q3, const float *scale, int I, int O) {
    int64_t ng = i3_groups(I);
    int64_t rb = i3_rowbytes(I);
    #pragma omp parallel for schedule(static) if(O >= 256)
    for (int o = 0; o < O; o++) {
        const uint8_t *wrow = q3 + (int64_t)o * rb;
        const float *srow = scale + (int64_t)o * ng;
        float acc = 0.f;
        for (int64_t g = 0; g < ng; g++) {
            const uint8_t *lo = wrow + g * I3_GBYTES, *hi = lo + 16;
            int base = (int)(g * I3_GROUP);
#if defined(__AVX512F__) && defined(__AVX512BW__)
            float a = dot_i3g64_avx512(lo, hi, x + base);
#else
            float a = 0.f;
            for (int k = 0; k < 64; k++) {
                uint8_t lbyte = lo[k >> 2], hbyte = hi[k >> 3];
                uint8_t lbits = (lbyte >> ((k & 3) * 2)) & 3, hbit = (hbyte >> (k & 7)) & 1;
                unsigned u = (unsigned)lbits | ((unsigned)hbit << 2);
                a += x[base + k] * (float)((int)u - 4);
            }
#endif
            acc += a * srow[g];
        }
        y[o] = acc;
    }
}

static void matmul_i4(float *y, const float *x, const uint8_t *q4, const float *scale, int I, int O) {
    int64_t ng = I / GS, rb = I / 2;
    #pragma omp parallel for schedule(static) if(O >= 256)
    for (int o = 0; o < O; o++) {
        const uint8_t *wrow = q4 + (int64_t)o * rb;
        const float *srow = scale + (int64_t)o * ng;
        float acc = 0.f;
        for (int64_t g = 0; g < ng; g++) {
            const uint8_t *p = wrow + g * (GS / 2);
            int base = (int)(g * GS);
#if defined(__AVX2__)
            float a = dot_i4g64_avx2(p, x + base);
#else
            float a = 0.f;
            for (int k = 0; k < 64; k++) {
                uint8_t b = p[k >> 1];
                int8_t v = (int8_t)((k & 1) ? ((b >> 4) & 0xF) : (b & 0xF));
                if (v & 8) v -= 16;
                a += x[base + k] * (float)v;
            }
#endif
            acc += a * srow[g];
        }
        y[o] = acc;
    }
}

static inline double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

typedef struct {
    double min, median, p10, p90, mean;
    double gb_s;
} BenchResult;

static BenchResult run_bench_func(void (*fn)(float*, const float*, const uint8_t*, const float*, int, int),
                                  float *y, const float *x, const uint8_t *w, const float *s, int I, int O,
                                  size_t weight_bytes, int n_iters) {
    double *times = malloc(n_iters * sizeof(double));
    for (int i = 0; i < 50; i++) fn(y, x, w, s, I, O);

    for (int i = 0; i < n_iters; i++) {
        double t0 = now_us();
        fn(y, x, w, s, I, O);
        double t1 = now_us();
        times[i] = t1 - t0;
    }
    qsort(times, n_iters, sizeof(double), cmp_double);
    BenchResult r;
    r.min = times[0];
    r.p10 = times[n_iters / 10];
    r.median = times[n_iters / 2];
    r.p90 = times[(n_iters * 9) / 10];
    double sum = 0; for (int i = 0; i < n_iters; i++) sum += times[i];
    r.mean = sum / n_iters;
    r.gb_s = ((double)weight_bytes / (r.median * 1e-6)) / 1e9;
    free(times);
    return r;
}

static void run_expert_ffn_i3(float *h_out, const float *x_in,
                             const uint8_t *w3_g, const uint8_t *w3_u, const uint8_t *w3_d,
                             const float *s_g, const float *s_u, const float *s_d,
                             float *buf_g, float *buf_u) {
    matmul_i3(buf_g, x_in, w3_g, s_g, HIDDEN, INTER);
    matmul_i3(buf_u, x_in, w3_u, s_u, HIDDEN, INTER);
    for (int i = 0; i < INTER; i++) {
        float gv = buf_g[i];
        buf_g[i] = (gv / (1.f + expf(-gv))) * buf_u[i];
    }
    matmul_i3(h_out, buf_g, w3_d, s_d, INTER, HIDDEN);
}

static void run_expert_ffn_i4(float *h_out, const float *x_in,
                             const uint8_t *w4_g, const uint8_t *w4_u, const uint8_t *w4_d,
                             const float *s_g, const float *s_u, const float *s_d,
                             float *buf_g, float *buf_u) {
    matmul_i4(buf_g, x_in, w4_g, s_g, HIDDEN, INTER);
    matmul_i4(buf_u, x_in, w4_u, s_u, HIDDEN, INTER);
    for (int i = 0; i < INTER; i++) {
        float gv = buf_g[i];
        buf_g[i] = (gv / (1.f + expf(-gv))) * buf_u[i];
    }
    matmul_i4(h_out, buf_g, w4_d, s_d, INTER, HIDDEN);
}

int main(int argc, char **argv) {
    int n_iters = argc > 1 ? atoi(argv[1]) : 2000;
    printf("=== Qwen3.6 MoE Kernel Compute-Only Benchmark (INT3 vs INT4) ===\n");
    printf("Threads: %d | Iters: %d\n", omp_get_max_threads(), n_iters);
    printf("Expert Dimensions: Hidden=%d, Inter=%d, GS=%d\n", HIDDEN, INTER, GS);
    printf("Gate/Up: %d -> %d (%d weights) | Down: %d -> %d (%d weights)\n",
           HIDDEN, INTER, HIDDEN * INTER, INTER, HIDDEN, INTER * HIDDEN);

    size_t gu_i3_bytes = i3_rowbytes(HIDDEN) * INTER;
    size_t d_i3_bytes  = i3_rowbytes(INTER) * HIDDEN;
    size_t total_i3_bytes = gu_i3_bytes + gu_i3_bytes + d_i3_bytes;

    size_t gu_i4_bytes = (HIDDEN / 2) * INTER;
    size_t d_i4_bytes  = (INTER / 2) * HIDDEN;
    size_t total_i4_bytes = gu_i4_bytes + gu_i4_bytes + d_i4_bytes;

    size_t gu_scales = (HIDDEN / GS) * INTER;
    size_t d_scales  = (INTER / GS) * HIDDEN;

    uint8_t *w3_g = malloc(gu_i3_bytes), *w3_u = malloc(gu_i3_bytes), *w3_d = malloc(d_i3_bytes);
    uint8_t *w4_g = malloc(gu_i4_bytes), *w4_u = malloc(gu_i4_bytes), *w4_d = malloc(d_i4_bytes);
    float *s_g = malloc(gu_scales * sizeof(float)), *s_u = malloc(gu_scales * sizeof(float)), *s_d = malloc(d_scales * sizeof(float));

    float *x_in = malloc(HIDDEN * sizeof(float));
    float *buf_g = malloc(INTER * sizeof(float)), *buf_u = malloc(INTER * sizeof(float));
    float *h_out = malloc(HIDDEN * sizeof(float));

    srand(42);
    for (size_t i = 0; i < gu_i3_bytes; i++) { w3_g[i] = rand() & 0xFF; w3_u[i] = rand() & 0xFF; }
    for (size_t i = 0; i < d_i3_bytes; i++)  { w3_d[i] = rand() & 0xFF; }
    for (size_t i = 0; i < gu_i4_bytes; i++) { w4_g[i] = rand() & 0xFF; w4_u[i] = rand() & 0xFF; }
    for (size_t i = 0; i < d_i4_bytes; i++)  { w4_d[i] = rand() & 0xFF; }
    for (size_t i = 0; i < gu_scales; i++)   { s_g[i] = 0.02f; s_u[i] = 0.02f; }
    for (size_t i = 0; i < d_scales; i++)    { s_d[i] = 0.02f; }
    for (int i = 0; i < HIDDEN; i++) x_in[i] = (float)((i % 11) - 5) * 0.1f;

    printf("INT3 Total Expert Weight: %zu bytes (%.3f MB)\n", total_i3_bytes, total_i3_bytes / 1048576.0);
    printf("INT4 Total Expert Weight: %zu bytes (%.3f MB)\n", total_i4_bytes, total_i4_bytes / 1048576.0);

    BenchResult r_gate_i3 = run_bench_func(matmul_i3, buf_g, x_in, w3_g, s_g, HIDDEN, INTER, gu_i3_bytes, n_iters);
    BenchResult r_gate_i4 = run_bench_func(matmul_i4, buf_g, x_in, w4_g, s_g, HIDDEN, INTER, gu_i4_bytes, n_iters);

    BenchResult r_up_i3   = run_bench_func(matmul_i3, buf_u, x_in, w3_u, s_u, HIDDEN, INTER, gu_i3_bytes, n_iters);
    BenchResult r_up_i4   = run_bench_func(matmul_i4, buf_u, x_in, w4_u, s_u, HIDDEN, INTER, gu_i4_bytes, n_iters);

    BenchResult r_down_i3 = run_bench_func(matmul_i3, h_out, buf_g, w3_d, s_d, INTER, HIDDEN, d_i3_bytes, n_iters);
    BenchResult r_down_i4 = run_bench_func(matmul_i4, h_out, buf_g, w4_d, s_d, INTER, HIDDEN, d_i4_bytes, n_iters);

    double *ffn_i3_times = malloc(n_iters * sizeof(double));
    double *ffn_i4_times = malloc(n_iters * sizeof(double));

    for (int i = 0; i < 50; i++) {
        run_expert_ffn_i3(h_out, x_in, w3_g, w3_u, w3_d, s_g, s_u, s_d, buf_g, buf_u);
        run_expert_ffn_i4(h_out, x_in, w4_g, w4_u, w4_d, s_g, s_u, s_d, buf_g, buf_u);
    }

    for (int i = 0; i < n_iters; i++) {
        double t0 = now_us();
        run_expert_ffn_i3(h_out, x_in, w3_g, w3_u, w3_d, s_g, s_u, s_d, buf_g, buf_u);
        double t1 = now_us();
        ffn_i3_times[i] = t1 - t0;
    }
    for (int i = 0; i < n_iters; i++) {
        double t0 = now_us();
        run_expert_ffn_i4(h_out, x_in, w4_g, w4_u, w4_d, s_g, s_u, s_d, buf_g, buf_u);
        double t1 = now_us();
        ffn_i4_times[i] = t1 - t0;
    }

    qsort(ffn_i3_times, n_iters, sizeof(double), cmp_double);
    qsort(ffn_i4_times, n_iters, sizeof(double), cmp_double);

    BenchResult r_ffn_i3, r_ffn_i4;
    r_ffn_i3.min = ffn_i3_times[0]; r_ffn_i3.p10 = ffn_i3_times[n_iters/10]; r_ffn_i3.median = ffn_i3_times[n_iters/2]; r_ffn_i3.p90 = ffn_i3_times[(n_iters*9)/10]; r_ffn_i3.gb_s = ((double)total_i3_bytes / (r_ffn_i3.median * 1e-6)) / 1e9;
    r_ffn_i4.min = ffn_i4_times[0]; r_ffn_i4.p10 = ffn_i4_times[n_iters/10]; r_ffn_i4.median = ffn_i4_times[n_iters/2]; r_ffn_i4.p90 = ffn_i4_times[(n_iters*9)/10]; r_ffn_i4.gb_s = ((double)total_i4_bytes / (r_ffn_i4.median * 1e-6)) / 1e9;

    printf("\n========================================================================================\n");
    printf("=== MEASUREMENT RESULTS (Isolated Compute-Only Latency & Bandwidth) ===\n");
    printf("========================================================================================\n");
    printf("%-20s | %-12s | %-12s | %-12s | %-12s | %-10s\n", "Operation", "INT3 Median", "INT3 BW", "INT4 Median", "INT4 BW", "Delta (Speedup)");
    printf("----------------------------------------------------------------------------------------\n");
    printf("%-20s | %8.2f us | %7.2f GB/s | %8.2f us | %7.2f GB/s | %+.1f%%\n",
           "Gate Proj (2048->512)", r_gate_i3.median, r_gate_i3.gb_s, r_gate_i4.median, r_gate_i4.gb_s,
           (r_gate_i4.median - r_gate_i3.median) * 100.0 / r_gate_i4.median);
    printf("%-20s | %8.2f us | %7.2f GB/s | %8.2f us | %7.2f GB/s | %+.1f%%\n",
           "Up Proj (2048->512)", r_up_i3.median, r_up_i3.gb_s, r_up_i4.median, r_up_i4.gb_s,
           (r_up_i4.median - r_up_i3.median) * 100.0 / r_up_i4.median);
    printf("%-20s | %8.2f us | %7.2f GB/s | %8.2f us | %7.2f GB/s | %+.1f%%\n",
           "Down Proj (512->2048)", r_down_i3.median, r_down_i3.gb_s, r_down_i4.median, r_down_i4.gb_s,
           (r_down_i4.median - r_down_i3.median) * 100.0 / r_down_i4.median);
    printf("%-20s | %8.2f us | %7.2f GB/s | %8.2f us | %7.2f GB/s | %+.1f%%\n",
           "COMPLETE EXPERT FFN", r_ffn_i3.median, r_ffn_i3.gb_s, r_ffn_i4.median, r_ffn_i4.gb_s,
           (r_ffn_i4.median - r_ffn_i3.median) * 100.0 / r_ffn_i4.median);
    printf("========================================================================================\n\n");

    printf("Detailed Complete FFN Distributions:\n");
    printf("  INT3: Min=%6.2f us | P10=%6.2f us | Median=%6.2f us | P90=%6.2f us | BW=%.2f GB/s\n",
           r_ffn_i3.min, r_ffn_i3.p10, r_ffn_i3.median, r_ffn_i3.p90, r_ffn_i3.gb_s);
    printf("  INT4: Min=%6.2f us | P10=%6.2f us | Median=%6.2f us | P90=%6.2f us | BW=%.2f GB/s\n",
           r_ffn_i4.min, r_ffn_i4.p10, r_ffn_i4.median, r_ffn_i4.p90, r_ffn_i4.gb_s);

    free(w3_g); free(w3_u); free(w3_d);
    free(w4_g); free(w4_u); free(w4_d);
    free(s_g); free(s_u); free(s_d);
    free(x_in); free(buf_g); free(buf_u); free(h_out);
    free(ffn_i3_times); free(ffn_i4_times);
    return 0;
}
