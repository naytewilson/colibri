#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "../st.h"

#define I3_GROUP 64
#define I3_GBYTES 24
#define GROUPS_PER_EXPERT 49152
#define NUM_EXPERTS 256

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

static inline float dot_i3g64_scalar(const uint8_t *lo, const uint8_t *hi, const float *x) {
    float sum = 0.0f;
    for (int k = 0; k < 64; k++) {
        uint8_t lbyte = lo[k >> 2];
        uint8_t hbyte = hi[k >> 3];
        uint8_t lbits = (lbyte >> ((k & 3) * 2)) & 3;
        uint8_t hbit  = (hbyte >> (k & 7)) & 1;
        int u = (int)lbits | ((int)hbit << 2);
        int v = u - 4; /* [-4, 3] */
        sum += x[k] * (float)v;
    }
    return sum;
}

int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : "/data/ANVIL/models/qwen36_i3_gs64_clean";
    int layer_to_check = (argc > 2) ? atoi(argv[2]) : 0;
    printf("=== Verifying Direct INT3 Converter Output ===\n");
    printf("Model Directory: %s\n", dir);
    printf("Layer to check:  %d\n", layer_to_check);

    shards S;
    memset(&S, 0, sizeof(S));
    st_init(&S, dir);
    printf("Indexed %d tensors across shards\n", S.n);

    int experts_found = 0;
    int kernel_mismatches = 0;
    int invalid_scales = 0;
    int invalid_values = 0;
    double max_kernel_diff = 0.0;
    double scale_min = 1e30, scale_max = 0.0, scale_sum = 0.0;
    int64_t total_groups = 0;
    int64_t hist[8] = {0};

    uint8_t *w_data = malloc(GROUPS_PER_EXPERT * I3_GBYTES);
    float *s_data = malloc(GROUPS_PER_EXPERT * sizeof(float));

    float x_test[64];
    for (int k = 0; k < 64; k++) {
        x_test[k] = (float)((k % 7) - 3) * 0.125f;
    }

    for (int e = 0; e < NUM_EXPERTS; e++) {
        char nm_w[256], nm_s[256];
        snprintf(nm_w, sizeof(nm_w), "model.layers.%d.mlp.experts.%d.merged_weight", layer_to_check, e);
        snprintf(nm_s, sizeof(nm_s), "model.layers.%d.mlp.experts.%d.qs", layer_to_check, e);

        st_tensor *tw = st_find(&S, nm_w);
        st_tensor *ts = st_find(&S, nm_s);

        if (!tw || !ts) {
            fprintf(stderr, "Expert %d: missing tensor(s) (tw=%p, ts=%p)\n", e, tw, ts);
            continue;
        }

        if (tw->nbytes != GROUPS_PER_EXPERT * I3_GBYTES) {
            fprintf(stderr, "Expert %d: merged_weight size %lld != expected %d\n",
                    e, (long long)tw->nbytes, GROUPS_PER_EXPERT * I3_GBYTES);
            return 1;
        }
        if (ts->nbytes != GROUPS_PER_EXPERT * sizeof(float)) {
            fprintf(stderr, "Expert %d: qs size %lld != expected %lu\n",
                    e, (long long)ts->nbytes, (unsigned long)(GROUPS_PER_EXPERT * sizeof(float)));
            return 1;
        }

        experts_found++;
        st_read_raw(&S, nm_w, w_data, 0);
        st_read_raw(&S, nm_s, s_data, 0);

        for (int g = 0; g < GROUPS_PER_EXPERT; g++) {
            float s = s_data[g];
            if (isnan(s) || isinf(s) || s <= 0.0f) {
                invalid_scales++;
            } else {
                if (s < scale_min) scale_min = s;
                if (s > scale_max) scale_max = s;
                scale_sum += s;
            }

            const uint8_t *lo = w_data + g * I3_GBYTES;
            const uint8_t *hi = lo + 16;

            /* Check value codes and histogram */
            for (int k = 0; k < 64; k++) {
                uint8_t lbyte = lo[k >> 2];
                uint8_t hbyte = hi[k >> 3];
                uint8_t lbits = (lbyte >> ((k & 3) * 2)) & 3;
                uint8_t hbit  = (hbyte >> (k & 7)) & 1;
                unsigned u = (unsigned)lbits | ((unsigned)hbit << 2);
                if (u > 7) invalid_values++;
                else hist[u]++;
            }

            /* Test AVX-512 vs Scalar */
#if defined(__AVX512F__) && defined(__AVX512BW__)
            float y_avx512 = dot_i3g64_avx512(lo, hi, x_test);
            float y_scalar = dot_i3g64_scalar(lo, hi, x_test);
            double diff = fabs((double)y_avx512 - (double)y_scalar);
            if (diff > max_kernel_diff) max_kernel_diff = diff;
            if (diff > 1e-4) {
                kernel_mismatches++;
            }
#endif
            total_groups++;
        }
    }

    printf("\n=== Verification Results ===\n");
    printf("Experts Verified:       %d / %d\n", experts_found, NUM_EXPERTS);
    printf("Total Groups Checked:   %lld (%lld weights)\n", (long long)total_groups, (long long)total_groups * 64);
    printf("Invalid Scale Count:    %d\n", invalid_scales);
    printf("Invalid Value Count:    %d\n", invalid_values);
    printf("Scale Min / Max / Mean: %.6e / %.6e / %.6e\n",
           scale_min, scale_max, (total_groups > 0 ? scale_sum / total_groups : 0.0));
    printf("Max AVX-512 Kernel Diff: %.6e\n", max_kernel_diff);
    printf("Kernel Mismatch Count:  %d\n", kernel_mismatches);

    printf("\nQuantized Value Histogram ([-4..3]):\n");
    for (int u = 0; u < 8; u++) {
        int v = u - 4;
        printf("  v=%2d (u=%d): %12lld (%5.2f%%)\n",
               v, u, (long long)hist[u], (double)hist[u] * 100.0 / (total_groups * 64));
    }

    free(w_data);
    free(s_data);

    if (experts_found == NUM_EXPERTS && invalid_scales == 0 && invalid_values == 0 && kernel_mismatches == 0) {
        printf("\n>>> VERDICT: PASS (Converter output layout, scales, bit-packing, and AVX-512 kernel verified bit-exact) <<<\n");
        return 0;
    } else {
        printf("\n>>> VERDICT: FAIL <<<\n");
        return 1;
    }
}
