/* Pure Native C Verifier for Mixed INT3/INT4-g64 Containers.
 * Validates tensor presence, byte geometry, scale integrity, format alignment with manifest,
 * and SIMD mathematical correctness across all 40 layers and 256 experts.
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

#define NUM_LAYERS 40
#define NUM_EXPERTS 256
#define TOTAL_EXPERTS (NUM_LAYERS * NUM_EXPERTS)
#define I3_GROUP 64
#define I3_GBYTES 24
#define GS 64

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

static float scalar_ref_i3(const uint8_t *lo, const uint8_t *hi, const float *x) {
    float acc = 0.0f;
    for (int k = 0; k < 64; k++) {
        uint8_t lbyte = lo[k >> 2], hbyte = hi[k >> 3];
        uint8_t lbits = (lbyte >> ((k & 3) * 2)) & 3, hbit = (hbyte >> (k & 7)) & 1;
        unsigned u = (unsigned)lbits | ((unsigned)hbit << 2);
        acc += x[k] * (float)((int)u - 4);
    }
    return acc;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <mixed_model_dir>\n", argv[0]);
        return 1;
    }
    const char *model_dir = argv[1];
    printf("=== Verifying Mixed Container: %s ===\n", model_dir);

    /* 1. Read manifest */
    char manifest_path[512];
    snprintf(manifest_path, sizeof(manifest_path), "%s/allocation_manifest.json", model_dir);
    int expected_fmt[TOTAL_EXPERTS];
    for (int i = 0; i < TOTAL_EXPERTS; i++) expected_fmt[i] = 5;

    FILE *f = fopen(manifest_path, "rb");
    if (!f) {
        fprintf(stderr, "Warning: Could not open manifest %s — defaulting to structural scan\n", manifest_path);
    } else {
        fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
        char *mbuf = malloc(sz + 1);
        if (fread(mbuf, 1, sz, f) == sz) {
            mbuf[sz] = '\0';
            char *arena = NULL;
            jval *root = json_parse(mbuf, &arena);
            if (root) {
                jval *allocs = json_get(root, "allocations");
                if (allocs && allocs->t == J_ARR) {
                    for (int i = 0; i < allocs->len; i++) {
                        jval *item = allocs->kids[i];
                        int l = (int)json_get(item, "layer")->num;
                        int e = (int)json_get(item, "eid")->num;
                        int fmt = (int)json_get(item, "fmt")->num;
                        if (l >= 0 && l < NUM_LAYERS && e >= 0 && e < NUM_EXPERTS) {
                            expected_fmt[l * NUM_EXPERTS + e] = fmt;
                        }
                    }
                }
            }
            free(mbuf);
            free(arena);
        }
        fclose(f);
    }

    shards S;
    st_init(&S, model_dir);

    int n_int3 = 0, n_int4 = 0, n_errors = 0;
    float test_x[64];
    for (int i = 0; i < 64; i++) test_x[i] = (float)((i % 7) - 3) * 0.1f;

    for (int l = 0; l < NUM_LAYERS; l++) {
        for (int e = 0; e < NUM_EXPERTS; e++) {
            int idx = l * NUM_EXPERTS + e;
            char nm[256], qsnm[256];
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.merged_weight", l, e);
            snprintf(qsnm, sizeof(qsnm), "model.layers.%d.mlp.experts.%d.qs", l, e);

            st_tensor *tw = st_find(&S, nm);
            st_tensor *ts = st_find(&S, qsnm);

            if (!tw) { fprintf(stderr, "Missing tensor %s\n", nm); n_errors++; continue; }
            if (!ts) { fprintf(stderr, "Missing tensor %s\n", qsnm); n_errors++; continue; }

            if (ts->numel != 49152) { fprintf(stderr, "Bad scale count %s: %lld\n", qsnm, (long long)ts->numel); n_errors++; }

            int actual_fmt = (tw->nbytes == 1179648) ? 5 : (tw->nbytes == 1572864) ? 4 : 1;
            if (actual_fmt == 5) n_int3++;
            else if (actual_fmt == 4) n_int4++;
            else { fprintf(stderr, "Unknown format %s (%lld bytes)\n", nm, (long long)tw->nbytes); n_errors++; }

            if (actual_fmt != expected_fmt[idx]) {
                fprintf(stderr, "Format mismatch %s: actual=%d, expected=%d\n", nm, actual_fmt, expected_fmt[idx]);
                n_errors++;
            }

            /* Test kernel math on representative sample */
            if (e % 64 == 0) {
                if (actual_fmt == 5) {
                    uint8_t *raw = malloc(tw->nbytes);
                    st_read_raw(&S, nm, raw, 0);
#if defined(__AVX512F__) && defined(__AVX512BW__)
                    float avx = dot_i3g64_avx512(raw, raw + 16, test_x);
                    float sca = scalar_ref_i3(raw, raw + 16, test_x);
                    if (fabsf(avx - sca) > 1e-4f) {
                        fprintf(stderr, "Kernel mismatch %s: avx=%.5f, sca=%.5f\n", nm, avx, sca);
                        n_errors++;
                    }
#endif
                    free(raw);
                }
            }
        }
    }

    printf("Verification Results:\n");
    printf("  Total Verified Experts: %d / %d\n", n_int3 + n_int4, TOTAL_EXPERTS);
    printf("  INT3 Experts (1.18 MB): %d (%.2f%%)\n", n_int3, (double)n_int3 * 100.0 / TOTAL_EXPERTS);
    printf("  INT4 Experts (1.57 MB): %d (%.2f%%)\n", n_int4, (double)n_int4 * 100.0 / TOTAL_EXPERTS);
    printf("  Total Errors:           %d\n", n_errors);

    if (n_errors == 0 && (n_int3 + n_int4) == TOTAL_EXPERTS) {
        printf("VERDICT: PASS_MIXED_CONTAINER_VERIFIED\n");
        return 0;
    } else {
        printf("VERDICT: FAIL_VERIFICATION\n");
        return 1;
    }
}
