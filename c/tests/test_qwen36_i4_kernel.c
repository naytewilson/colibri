#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <string.h>

static void matmul_q_gs_ref(float *y, const float *x, const int8_t *q, const float *scale,
                            int I, int O, int gs) {
    int ng = (I + gs - 1) / gs;
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        const float *sc = scale + (int64_t)o * ng;
        float acc = 0.f;
        for (int gi = 0; gi < ng; gi++) {
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            int base = gi * gs, end = base + gs; if (end > I) end = I;
            for (int i = base; i + 16 <= end; i += 16) {
                __m128i b0 = _mm_loadu_si128((const __m128i*)(w + i));
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i),   _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)), a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+8), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b0,8))), a1);
            }
            a0 = _mm256_add_ps(a0, a1);
            __m128 s = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0,1));
            s = _mm_add_ps(s, _mm_movehl_ps(s,s));
            s = _mm_add_ss(s, _mm_shuffle_ps(s,s,1));
            acc += _mm_cvtss_f32(s) * sc[gi];
        }
        y[o] = acc;
    }
}

static void matmul_i4_gs_fast(float *y, const float *x, const uint8_t *q4, const float *scale,
                              int I, int O, int gs) {
    int ng = (I + gs - 1) / gs;
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i s8 = _mm_set1_epi8(0x08);

    for (int o = 0; o < O; o++) {
        const uint8_t *w4 = q4 + (int64_t)o * (I / 2);
        const float *sc = scale + (int64_t)o * ng;
        float acc = 0.f;

        for (int gi = 0; gi < ng; gi++) {
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            int base = gi * gs; // gs = 64

            // Process 64 elements in two 32-element steps
            for (int step = 0; step < 2; step++) {
                int i = base + step * 32;
                __m128i raw16 = _mm_loadu_si128((const __m128i*)(w4 + (i >> 1)));
                __m128i lo = _mm_and_si128(raw16, m4);
                __m128i hi = _mm_and_si128(_mm_srli_epi16(raw16, 4), m4);
                lo = _mm_sub_epi8(_mm_xor_si128(lo, s8), s8);
                hi = _mm_sub_epi8(_mm_xor_si128(hi, s8), s8);

                __m128i w0_15  = _mm_unpacklo_epi8(lo, hi);
                __m128i w16_31 = _mm_unpackhi_epi8(lo, hi);

                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i),      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w0_15)), a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8),  _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w0_15, 8))), a1);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 16), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w16_31)), a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 24), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w16_31, 8))), a1);
            }

            a0 = _mm256_add_ps(a0, a1);
            __m128 s = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0,1));
            s = _mm_add_ps(s, _mm_movehl_ps(s,s));
            s = _mm_add_ss(s, _mm_shuffle_ps(s,s,1));
            acc += _mm_cvtss_f32(s) * sc[gi];
        }
        y[o] = acc;
    }
}

int main() {
    int I = 2048, O = 512, gs = 64;
    float *x = malloc(I * sizeof(float));
    int8_t *q_ref = malloc((int64_t)O * I);
    uint8_t *q4 = malloc((int64_t)O * (I / 2));
    float *scales = malloc((int64_t)O * (I / gs) * sizeof(float));
    float *y_ref = malloc(O * sizeof(float));
    float *y_fast = malloc(O * sizeof(float));

    for (int i = 0; i < I; i++) x[i] = ((float)rand() / RAND_MAX) * 2.f - 1.f;
    for (int i = 0; i < O * (I / gs); i++) scales[i] = ((float)rand() / RAND_MAX) * 0.1f;

    for (int o = 0; o < O; o++) {
        for (int i = 0; i < I; i += 2) {
            int8_t v0 = (rand() % 16) - 8;
            int8_t v1 = (rand() % 16) - 8;
            q_ref[o * I + i] = v0;
            q_ref[o * I + i + 1] = v1;
            uint8_t nib0 = (uint8_t)(v0 & 0xF);
            uint8_t nib1 = (uint8_t)(v1 & 0xF);
            q4[o * (I / 2) + (i / 2)] = (nib1 << 4) | nib0;
        }
    }

    matmul_q_gs_ref(y_ref, x, q_ref, scales, I, O, gs);
    matmul_i4_gs_fast(y_fast, x, q4, scales, I, O, gs);

    float max_diff = 0.f;
    for (int o = 0; o < O; o++) {
        float diff = fabsf(y_ref[o] - y_fast[o]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("KERNEL PARITY TEST: max_diff = %e -> %s\n", max_diff, (max_diff < 1e-6f) ? "PASS (EXACT MATCH)" : "FAIL");
    return (max_diff < 1e-6f) ? 0 : 1;
}
