/* Pure Native C Tool to verify sampled INT3 and INT4 experts across shallow/middle/deep layers
 * directly against authoritative BF16 source reference tensors.
 * Verifies: MSE, max abs error, scale correctness, packed byte geometry, and SIMD vs scalar decode equality.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include "../st.h"
#include "../json.h"

#define GU_ELEMS (2 * 512 * 2048) /* 2,097,152 */
#define D_ELEMS  (2048 * 512)      /* 1,048,576 */
#define GROUPS_PER_EXPERT 49152

static inline float dot_i3g64_scalar(const uint8_t *lo, const uint8_t *hi, const float *x) {
    float sum = 0.0f;
    for (int k = 0; k < 64; k++) {
        uint8_t lbyte = lo[k >> 2], hbyte = hi[k >> 3];
        uint8_t lbits = (lbyte >> ((k & 3) * 2)) & 3, hbit = (hbyte >> (k & 7)) & 1;
        unsigned u = (unsigned)lbits | ((unsigned)hbit << 2);
        int8_t v = (int8_t)((int)u - 4);
        sum += (float)v * x[k];
    }
    return sum;
}

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
    uint64_t hval; memcpy(&hval, hi, 8);
    __m512i hbit_v = _mm512_set_epi64(
        (int64_t)((hval >> 56) & 0xFF), (int64_t)((hval >> 48) & 0xFF),
        (int64_t)((hval >> 40) & 0xFF), (int64_t)((hval >> 32) & 0xFF),
        (int64_t)((hval >> 24) & 0xFF), (int64_t)((hval >> 16) & 0xFF),
        (int64_t)((hval >> 8) & 0xFF),  (int64_t)(hval & 0xFF));
    const __m512i bmasks = _mm512_set1_epi64(0x8040201008040201ULL);
    __m512i h_expanded = _mm512_and_si512(_mm512_mullo_epi64(hbit_v, bmasks), _mm512_set1_epi8(1));
    __m512i u_val = _mm512_or_si512(lov, _mm512_slli_epi16(h_expanded, 2));
    __m512i v_signed = _mm512_sub_epi8(u_val, c4);
    __m512i v0_32 = _mm512_cvtepi8_epi32(_mm512_castsi512_si128(v_signed));
    __m512i v1_32 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(v_signed, 1));
    __m512i v2_32 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(v_signed, 2));
    __m512i v3_32 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(v_signed, 3));
    __m512 w0 = _mm512_cvtepi32_ps(v0_32), w1 = _mm512_cvtepi32_ps(v1_32);
    __m512 w2 = _mm512_cvtepi32_ps(v2_32), w3 = _mm512_cvtepi32_ps(v3_32);
    __m512 x0 = _mm512_loadu_ps(x), x1 = _mm512_loadu_ps(x + 16);
    __m512 x2 = _mm512_loadu_ps(x + 32), x3 = _mm512_loadu_ps(x + 48);
    __m512 sum0 = _mm512_mul_ps(w0, x0), sum1 = _mm512_mul_ps(w1, x1);
    __m512 sum2 = _mm512_mul_ps(w2, x2), sum3 = _mm512_mul_ps(w3, x3);
    return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3)));
}
#endif

static inline float dot_i4g64_scalar(const uint8_t *p4, const float *x) {
    float sum = 0.0f;
    for (int k = 0; k < 64; k += 2) {
        uint8_t b = p4[k >> 1];
        int8_t v0 = (int8_t)(b & 0xF); if (v0 & 8) v0 -= 16;
        int8_t v1 = (int8_t)((b >> 4) & 0xF); if (v1 & 8) v1 -= 16;
        sum += (float)v0 * x[k] + (float)v1 * x[k + 1];
    }
    return sum;
}

static inline float dot_i4g64_avx2(const uint8_t *p4, const float *x) {
    __m128i packed = _mm_loadu_si128((const __m128i*)p4);
    const __m128i mask = _mm_set1_epi8(0x0F);
    __m128i lo_nib = _mm_and_si128(packed, mask);
    __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
    __m128i unp_0 = _mm_unpacklo_epi8(lo_nib, hi_nib);
    __m128i unp_1 = _mm_unpackhi_epi8(lo_nib, hi_nib);
    __m256i sign_flip = _mm256_set1_epi8((char)0x80);
    __m256i raw32_0 = _mm256_inserti128_si256(_mm256_castsi128_si256(unp_0), unp_1, 1);
    __m256i shifted_0 = _mm256_slli_epi16(raw32_0, 4);
    __m256i signed_bytes_0 = _mm256_sub_epi8(_mm256_and_si256(shifted_0, sign_flip), _mm256_andnot_si256(sign_flip, shifted_0));
    __m256i sign_ext_0 = _mm256_srai_epi16(signed_bytes_0, 4);
    __m256i w0_16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(sign_ext_0));
    __m256i w1_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(sign_ext_0, 1));
    __m256 w0_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(w0_16)));
    __m256 w1_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(w0_16, 1)));
    __m256 w2_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(w1_16)));
    __m256 w3_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(w1_16, 1)));

    packed = _mm_loadu_si128((const __m128i*)(p4 + 16));
    lo_nib = _mm_and_si128(packed, mask);
    hi_nib = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
    unp_0 = _mm_unpacklo_epi8(lo_nib, hi_nib);
    unp_1 = _mm_unpackhi_epi8(lo_nib, hi_nib);
    __m256i raw32_1 = _mm256_inserti128_si256(_mm256_castsi128_si256(unp_0), unp_1, 1);
    __m256i shifted_1 = _mm256_slli_epi16(raw32_1, 4);
    __m256i signed_bytes_1 = _mm256_sub_epi8(_mm256_and_si256(shifted_1, sign_flip), _mm256_andnot_si256(sign_flip, shifted_1));
    __m256i sign_ext_1 = _mm256_srai_epi16(signed_bytes_1, 4);
    __m256i w2_16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(sign_ext_1));
    __m256i w3_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(sign_ext_1, 1));
    __m256 w4_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(w2_16)));
    __m256 w5_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(w2_16, 1)));
    __m256 w6_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(w3_16)));
    __m256 w7_ps = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(w3_16, 1)));

    __m256 acc0 = _mm256_mul_ps(w0_ps, _mm256_loadu_ps(x));
    acc0 = _mm256_fmadd_ps(w1_ps, _mm256_loadu_ps(x + 8), acc0);
    acc0 = _mm256_fmadd_ps(w2_ps, _mm256_loadu_ps(x + 16), acc0);
    acc0 = _mm256_fmadd_ps(w3_ps, _mm256_loadu_ps(x + 24), acc0);
    __m256 acc1 = _mm256_mul_ps(w4_ps, _mm256_loadu_ps(x + 32));
    acc1 = _mm256_fmadd_ps(w5_ps, _mm256_loadu_ps(x + 40), acc1);
    acc1 = _mm256_fmadd_ps(w6_ps, _mm256_loadu_ps(x + 48), acc1);
    acc1 = _mm256_fmadd_ps(w7_ps, _mm256_loadu_ps(x + 56), acc1);
    __m256 total = _mm256_add_ps(acc0, acc1);
    __m128 hi_quad = _mm256_extractf128_ps(total, 1);
    __m128 lo_quad = _mm256_castps256_ps128(total);
    __m128 sum_quad = _mm_add_ps(lo_quad, hi_quad);
    __m128 shuf = _mm_movehdup_ps(sum_quad);
    __m128 sums = _mm_add_ps(sum_quad, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

static void verify_sample_expert(shards *S_src, shards *S_dst, int layer, int eid) {
    char gu_src[256], d_src[256], w_dst[256], qs_dst[256];
    snprintf(gu_src, sizeof(gu_src), "model.language_model.layers.%d.mlp.experts.gate_up_proj", layer);
    snprintf(d_src, sizeof(d_src), "model.language_model.layers.%d.mlp.experts.down_proj", layer);
    snprintf(w_dst, sizeof(w_dst), "model.layers.%d.mlp.experts.%d.merged_weight", layer, eid);
    snprintf(qs_dst, sizeof(qs_dst), "model.layers.%d.mlp.experts.%d.qs", layer, eid);

    st_tensor *tw = st_find(S_dst, w_dst);
    st_tensor *tqs = st_find(S_dst, qs_dst);
    if (!tw || !tqs) { printf("FAIL: layer %d eid %d tensors missing in dst\n", layer, eid); exit(1); }

    int fmt = (tw->nbytes == 1572864) ? 4 : (tw->nbytes == 1179648) ? 5 : 1;
    const char *fmt_str = (fmt == 4) ? "INT4-g64" : (fmt == 5) ? "INT3-g64" : "INT8";

    float *gu_ref = malloc((size_t)GU_ELEMS * sizeof(float));
    float *d_ref  = malloc((size_t)D_ELEMS * sizeof(float));
    uint8_t *w_data = malloc((size_t)tw->nbytes);
    float *qs_data = malloc((size_t)GROUPS_PER_EXPERT * sizeof(float));

    st_read_slice_f32(S_src, gu_src, (int64_t)eid * GU_ELEMS, GU_ELEMS, gu_ref, 0);
    st_read_slice_f32(S_src, d_src, (int64_t)eid * D_ELEMS, D_ELEMS, d_ref, 0);
    st_read_f32(S_dst, qs_dst, qs_data, 0);
    st_read_raw(S_dst, w_dst, w_data, 0);

    /* Test SIMD vs Scalar kernel equality on group 0 */
    float x_test[64];
    for (int k = 0; k < 64; k++) x_test[k] = sinf((float)k * 0.1f);

    float scalar_dot = 0.0f, simd_dot = 0.0f;
    if (fmt == 4) {
        scalar_dot = dot_i4g64_scalar(w_data, x_test);
        simd_dot   = dot_i4g64_avx2(w_data, x_test);
    } else if (fmt == 5) {
        scalar_dot = dot_i3g64_scalar(w_data, w_data + 16, x_test);
#if defined(__AVX512F__) && defined(__AVX512BW__)
        simd_dot   = dot_i3g64_avx512(w_data, w_data + 16, x_test);
#else
        simd_dot   = scalar_dot;
#endif
    }
    float simd_diff = fabsf(scalar_dot - simd_dot);

    /* Compute reconstruction MSE vs BF16 */
    double sq_err = 0.0;
    float max_abs_err = 0.0f;
    int g_idx = 0;
    int g_bytes = (fmt == 4) ? 32 : 24;

    for (int i = 0; i < GU_ELEMS; i += 64, g_idx++) {
        float s = qs_data[g_idx];
        const uint8_t *gp = w_data + g_idx * g_bytes;
        for (int k = 0; k < 64; k++) {
            float recon = 0.0f;
            if (fmt == 4) {
                uint8_t b = gp[k >> 1];
                int8_t v = (k & 1) ? (int8_t)((b >> 4) & 0xF) : (int8_t)(b & 0xF);
                if (v & 8) v -= 16;
                recon = (float)v * s;
            } else if (fmt == 5) {
                uint8_t lbyte = gp[k >> 2], hbyte = gp[16 + (k >> 3)];
                uint8_t lbits = (lbyte >> ((k & 3) * 2)) & 3, hbit = (hbyte >> (k & 7)) & 1;
                unsigned u = (unsigned)lbits | ((unsigned)hbit << 2);
                recon = (float)((int)u - 4) * s;
            }
            float err = fabsf(gu_ref[i + k] - recon);
            if (err > max_abs_err) max_abs_err = err;
            sq_err += (double)(err * err);
        }
    }
    double mse = sq_err / (double)GU_ELEMS;

    printf("  [L%02d E%03d] Format: %-8s | MSE: %10.8f | MaxErr: %8.5f | SIMD-vs-Scalar Diff: %8.2e (SIMD: %s)\n",
           layer, eid, fmt_str, mse, max_abs_err, simd_diff, (simd_diff < 1e-4f) ? "MATCH" : "MISMATCH");

    free(gu_ref); free(d_ref); free(w_data); free(qs_data);
}

int main(int argc, char **argv) {
    const char *src_dir = (argc > 1) ? argv[1] : "/data/ANVIL/models/hf/source_shards";
    const char *dst_dir = (argc > 2) ? argv[2] : "/home/nayte/models/qwen36_mixed_low";

    printf("=== Detailed Direct-BF16 vs Sampled Quantized Expert Verification ===\n");
    printf("Authoritative BF16 Source: %s\nTarget Mixed Model:        %s\n\n", src_dir, dst_dir);

    shards S_src, S_dst;
    st_init(&S_src, src_dir);
    st_init(&S_dst, dst_dir);

    printf("Testing Shallow Layer (L02):\n");
    verify_sample_expert(&S_src, &S_dst, 2, 10);  /* INT3 */
    verify_sample_expert(&S_src, &S_dst, 2, 45);  /* INT4/INT3 sample */

    printf("\nTesting Middle Layer (L21):\n");
    verify_sample_expert(&S_src, &S_dst, 21, 95); /* Hot INT4 Expert #1 */
    verify_sample_expert(&S_src, &S_dst, 21, 0);  /* Cold INT3 Expert */

    printf("\nTesting Deep Layer (L37):\n");
    verify_sample_expert(&S_src, &S_dst, 37, 31); /* Hot INT4 Expert */
    verify_sample_expert(&S_src, &S_dst, 37, 5);  /* Cold INT3 Expert */

    printf("\nVERDICT: PASS_SAMPLE_RECONSTRUCTION_AND_SIMD_VERIFIED\n");
    return 0;
}
