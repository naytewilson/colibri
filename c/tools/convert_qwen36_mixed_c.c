/* Pure Native C Direct BF16 -> Heterogeneous Mixed INT3/INT4-g64 Container Builder for Qwen3.6-35B-A3B.
 * Converts authoritative BF16 safetensors shards directly to a mixed container based on an allocation manifest.
 *
 * For each expert:
 *   if manifest specifies INT4 (fmt=4): direct BF16 -> INT4-g64 (1,572,864 B)
 *   if manifest specifies INT3 (fmt=5): direct BF16 -> INT3-g64 (1,179,648 B)
 *
 * Emits complete self-contained container with dense weights in FP16, global embeddings, and metadata.
 * NO Python dependency. Compiles with gcc -O3 -fopenmp -mavx512f -mavx512bw -mf16c.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../st.h"
#include "../json.h"

#define I3_GROUP 64
#define I3_GBYTES 24
#define NUM_LAYERS 40
#define NUM_EXPERTS 256
#define TOTAL_EXPERTS (NUM_LAYERS * NUM_EXPERTS)

#define INTER_SIZE 512
#define HIDDEN_SIZE 2048
#define GU_ELEMS_PER_EXPERT (2 * INTER_SIZE * HIDDEN_SIZE) /* 2,097,152 */
#define D_ELEMS_PER_EXPERT (HIDDEN_SIZE * INTER_SIZE)       /* 1,048,576 */
#define WEIGHTS_PER_EXPERT (GU_ELEMS_PER_EXPERT + D_ELEMS_PER_EXPERT) /* 3,145,728 */
#define GROUPS_PER_EXPERT 49152
#define SCALE_BYTES_PER_EXPERT (GROUPS_PER_EXPERT * 4)

typedef struct {
    char name[256];
    char dtype[16];
    int rank;
    int64_t shape[4];
    int64_t nbytes;
    void *src_buf;
} OutTensor;

static inline void f32_to_f16_array(const float *src, uint16_t *dst, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
    for (; i + 8 <= n; i += 8) {
        __m256 f = _mm256_loadu_ps(src + i);
        __m128i h = _mm256_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i*)(dst + i), h);
    }
#endif
    for (; i < n; i++) {
#if defined(__F16C__)
        __m128 f1 = _mm_set_ss(src[i]);
        __m128i h1 = _mm_cvtps_ph(f1, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        dst[i] = (uint16_t)_mm_extract_epi16(h1, 0);
#else
        union { float f; uint32_t u; } v; v.f = src[i];
        uint32_t sign = (v.u >> 16) & 0x8000;
        int32_t exp = ((v.u >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = v.u & 0x7FFFFF;
        if (exp <= 0) dst[i] = sign;
        else if (exp >= 31) dst[i] = sign | 0x7C00;
        else dst[i] = sign | (exp << 10) | (mant >> 13);
#endif
    }
}

static inline void quantize_group_f32_to_i3(const float *w, uint8_t *lo, uint8_t *hi, float *s_out) {
    float max_abs = 0.0f;
    for (int k = 0; k < 64; k++) { float a = fabsf(w[k]); if (a > max_abs) max_abs = a; }
    if (max_abs <= 1e-12f) { *s_out = 1e-8f; memset(lo, 0x55, 16); memset(hi, 0xFF, 8); return; }
    float s = max_abs / 3.0f;
    if (s < 1e-12f) s = 1e-12f;
    *s_out = s;
    float inv_s = 1.0f / s;
    memset(lo, 0, 16); memset(hi, 0, 8);
    for (int k = 0; k < 64; k++) {
        int v = (int)lrintf(w[k] * inv_s);
        if (v > 3) v = 3; if (v < -4) v = -4;
        unsigned u = (unsigned)(v + 4);
        lo[k >> 2] |= (uint8_t)((u & 3) << ((k & 3) * 2));
        hi[k >> 3] |= (uint8_t)(((u >> 2) & 1) << (k & 7));
    }
}

static inline void quantize_group_f32_to_i4(const float *w, uint8_t *p4, float *s_out) {
    float max_abs = 0.0f;
    for (int k = 0; k < 64; k++) { float a = fabsf(w[k]); if (a > max_abs) max_abs = a; }
    if (max_abs <= 1e-12f) { *s_out = 1e-8f; memset(p4, 0x88, 32); return; }
    float s = max_abs / 7.0f;
    if (s < 1e-12f) s = 1e-12f;
    *s_out = s;
    float inv_s = 1.0f / s;
    for (int k = 0; k < 64; k += 2) {
        int v0 = (int)lrintf(w[k] * inv_s);
        if (v0 > 7) v0 = 7; if (v0 < -8) v0 = -8;
        int v1 = (int)lrintf(w[k+1] * inv_s);
        if (v1 > 7) v1 = 7; if (v1 < -8) v1 = -8;
        uint8_t b0 = (uint8_t)(v0 & 0xF);
        uint8_t b1 = (uint8_t)(v1 & 0xF);
        p4[k >> 1] = (b1 << 4) | b0;
    }
}

static void quantize_expert_i3(const float *gu, const float *down, uint8_t *w3, float *s3) {
    const float *gate = gu;
    const float *up   = gu + (INTER_SIZE * HIDDEN_SIZE);
    int g_idx = 0;
    for (int i = 0; i < (INTER_SIZE * HIDDEN_SIZE); i += 64) {
        uint8_t *lo = w3 + g_idx * I3_GBYTES;
        quantize_group_f32_to_i3(gate + i, lo, lo + 16, &s3[g_idx]);
        g_idx++;
    }
    for (int i = 0; i < (INTER_SIZE * HIDDEN_SIZE); i += 64) {
        uint8_t *lo = w3 + g_idx * I3_GBYTES;
        quantize_group_f32_to_i3(up + i, lo, lo + 16, &s3[g_idx]);
        g_idx++;
    }
    for (int i = 0; i < (HIDDEN_SIZE * INTER_SIZE); i += 64) {
        uint8_t *lo = w3 + g_idx * I3_GBYTES;
        quantize_group_f32_to_i3(down + i, lo, lo + 16, &s3[g_idx]);
        g_idx++;
    }
}

static void quantize_expert_i4(const float *gu, const float *down, uint8_t *w4, float *s4) {
    const float *gate = gu;
    const float *up   = gu + (INTER_SIZE * HIDDEN_SIZE);
    int g_idx = 0;
    for (int i = 0; i < (INTER_SIZE * HIDDEN_SIZE); i += 64) {
        quantize_group_f32_to_i4(gate + i, w4 + g_idx * 32, &s4[g_idx]);
        g_idx++;
    }
    for (int i = 0; i < (INTER_SIZE * HIDDEN_SIZE); i += 64) {
        quantize_group_f32_to_i4(up + i, w4 + g_idx * 32, &s4[g_idx]);
        g_idx++;
    }
    for (int i = 0; i < (HIDDEN_SIZE * INTER_SIZE); i += 64) {
        quantize_group_f32_to_i4(down + i, w4 + g_idx * 32, &s4[g_idx]);
        g_idx++;
    }
}

static void write_shard(const char *dst_path, OutTensor *tensors, int n_tensors) {
    size_t hcap = 256 * 1024;
    char *hbuf = malloc(hcap);
    if (!hbuf) { fprintf(stderr, "OOM in write_shard header\n"); exit(1); }

    int hpos = snprintf(hbuf, hcap, "{\"__metadata__\":{\"format\":\"pt\"}");
    uint64_t data_offset = 0;

    for (int i = 0; i < n_tensors; i++) {
        uint64_t start = data_offset;
        uint64_t end   = data_offset + tensors[i].nbytes;
        data_offset += tensors[i].nbytes;

        hpos += snprintf(hbuf + hpos, hcap - hpos,
            ",\"%s\":{\"dtype\":\"%s\",\"shape\":[", tensors[i].name, tensors[i].dtype);
        for (int r = 0; r < tensors[i].rank; r++) {
            hpos += snprintf(hbuf + hpos, hcap - hpos, "%lld%s",
                (long long)tensors[i].shape[r], (r == tensors[i].rank - 1) ? "" : ",");
        }
        hpos += snprintf(hbuf + hpos, hcap - hpos, "],\"data_offsets\":[%llu,%llu]}",
            (unsigned long long)start, (unsigned long long)end);
    }
    hpos += snprintf(hbuf + hpos, hcap - hpos, "}");

    int pad = (8 - (hpos % 8)) % 8;
    for (int p = 0; p < pad; p++) hbuf[hpos++] = ' ';

    uint64_t hlen = (uint64_t)hpos;
    FILE *out = fopen(dst_path, "wb");
    if (!out) { perror(dst_path); exit(1); }

    fwrite(&hlen, 1, 8, out);
    fwrite(hbuf, 1, hlen, out);

    for (int i = 0; i < n_tensors; i++) {
        if (fwrite(tensors[i].src_buf, 1, (size_t)tensors[i].nbytes, out) != (size_t)tensors[i].nbytes) {
            fprintf(stderr, "Write error on %s in %s\n", tensors[i].name, dst_path);
            exit(1);
        }
    }
    fclose(out);
    free(hbuf);
}

static int load_allocation_map(const char *manifest_path, int *fmt_map) {
    for (int i = 0; i < TOTAL_EXPERTS; i++) fmt_map[i] = 5; /* default INT3 */
    FILE *f = fopen(manifest_path, "rb");
    if (!f) { fprintf(stderr, "Error: Could not open manifest %s\n", manifest_path); return 0; }
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz + 1);
    if (!buf) { fclose(f); return 0; }
    if (fread(buf, 1, sz, f) != sz) { free(buf); fclose(f); return 0; }
    buf[sz] = '\0'; fclose(f);

    char *arena = NULL;
    jval *root = json_parse(buf, &arena);
    if (!root) { free(buf); free(arena); return 0; }
    jval *allocs = json_get(root, "allocations");
    if (!allocs || allocs->t != J_ARR) { free(buf); free(arena); return 0; }

    int count_i4 = 0, count_i3 = 0;
    for (int i = 0; i < allocs->len; i++) {
        jval *item = allocs->kids[i];
        int l = (int)json_get(item, "layer")->num;
        int e = (int)json_get(item, "eid")->num;
        int fmt = (int)json_get(item, "fmt")->num;
        if (l >= 0 && l < NUM_LAYERS && e >= 0 && e < NUM_EXPERTS) {
            int idx = l * NUM_EXPERTS + e;
            fmt_map[idx] = fmt;
            if (fmt == 4) count_i4++;
            else count_i3++;
        }
    }
    free(buf);
    free(arena);
    printf("Loaded allocation map: %d INT4 experts (%.2f%%), %d INT3 experts\n",
           count_i4, (double)count_i4 * 100.0 / TOTAL_EXPERTS, count_i3);
    return 1;
}

int convert_globals_direct(shards *S, const char *dst_dir) {
    char dst_path[1024];
    snprintf(dst_path, sizeof(dst_path), "%s/model-globals.safetensors", dst_dir);

    const char *glob_keys[3][2] = {
        {"model.language_model.embed_tokens.weight", "model.embed_tokens.weight"},
        {"model.language_model.lm_head.weight", "lm_head.weight"},
        {"model.language_model.norm.weight", "model.norm.weight"}
    };
    const char *out_names[3] = {
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight"
    };

    OutTensor out_tensors[3];
    uint16_t *glob_blobs[3];
    int n_out = 0;

    for (int k = 0; k < 3; k++) {
        const char *k1 = glob_keys[k][0];
        const char *k2 = glob_keys[k][1];
        const char *src_k = st_find(S, k1) ? k1 : (st_find(S, k2) ? k2 : NULL);
        if (!src_k) {
            fprintf(stderr, "Global tensor %s not found in source shards\n", k1);
            continue;
        }
        st_tensor *st = st_find(S, src_k);
        OutTensor *ot = &out_tensors[n_out++];
        strcpy(ot->name, out_names[k]);
        strcpy(ot->dtype, "F16");
        ot->rank = st->rank;
        for (int r = 0; r < st->rank; r++) ot->shape[r] = st->shape[r];
        ot->nbytes = st->numel * sizeof(uint16_t);

        float *f32_tmp = malloc((size_t)st->numel * sizeof(float));
        uint16_t *f16_buf = malloc((size_t)st->numel * sizeof(uint16_t));
        if (!f32_tmp || !f16_buf) { fprintf(stderr, "OOM reading global tensor %s\n", src_k); exit(1); }

        st_read_f32(S, src_k, f32_tmp, 0);
        f32_to_f16_array(f32_tmp, f16_buf, st->numel);
        free(f32_tmp);

        glob_blobs[k] = f16_buf;
        ot->src_buf = f16_buf;
    }

    if (n_out > 0) {
        write_shard(dst_path, out_tensors, n_out);
        printf("[direct-c] wrote globals shard -> %s (%d tensors)\n", dst_path, n_out);
        for (int k = 0; k < n_out; k++) free(glob_blobs[k]);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <bf16_source_shards_dir> <manifest.json> <output_dir>\n", argv[0]);
        return 1;
    }
    const char *src_dir = argv[1];
    const char *manifest_path = argv[2];
    const char *out_dir = argv[3];
    mkdir(out_dir, 0755);

    printf("=== Pure C Direct BF16 -> Mixed INT3/INT4 Container Builder ===\n");
    printf("Authoritative Source Shards: %s\nManifest: %s\nOutput Dir: %s\n\n",
           src_dir, manifest_path, out_dir);

    int *fmt_map = malloc(TOTAL_EXPERTS * sizeof(int));
    if (!load_allocation_map(manifest_path, fmt_map)) return 1;

    shards S;
    st_init(&S, src_dir);

    convert_globals_direct(&S, out_dir);

    #pragma omp parallel for schedule(dynamic)
    for (int l = 0; l < NUM_LAYERS; l++) {
        char dst_path[1024];
        snprintf(dst_path, sizeof(dst_path), "%s/model-%05d.safetensors", out_dir, l);

        char gu_name1[256], gu_name2[256], d_name1[256], d_name2[256];
        snprintf(gu_name1, sizeof(gu_name1), "model.language_model.layers.%d.mlp.experts.gate_up_proj", l);
        snprintf(gu_name2, sizeof(gu_name2), "model.layers.%d.mlp.experts.gate_up_proj", l);
        snprintf(d_name1, sizeof(d_name1), "model.language_model.layers.%d.mlp.experts.down_proj", l);
        snprintf(d_name2, sizeof(d_name2), "model.layers.%d.mlp.experts.down_proj", l);

        const char *gu_name = st_find(&S, gu_name1) ? gu_name1 : (st_find(&S, gu_name2) ? gu_name2 : NULL);
        const char *d_name  = st_find(&S, d_name1) ? d_name1 : (st_find(&S, d_name2) ? d_name2 : NULL);

        if (!gu_name || !d_name) {
            fprintf(stderr, "Layer %d: routed experts not found in source shards\n", l);
            exit(1);
        }

        OutTensor out_tensors[600];
        int n_out = 0;

        uint8_t *w_all[NUM_EXPERTS];
        float *s_all[NUM_EXPERTS];
        uint16_t *dense_blobs[64];
        int n_dense = 0;

        for (int e = 0; e < NUM_EXPERTS; e++) {
            int idx = l * NUM_EXPERTS + e;
            int fmt = fmt_map[idx];

            float *gu_buf = malloc((size_t)GU_ELEMS_PER_EXPERT * sizeof(float));
            float *d_buf  = malloc((size_t)D_ELEMS_PER_EXPERT * sizeof(float));
            if (!gu_buf || !d_buf) { fprintf(stderr, "OOM allocating expert buffers\n"); exit(1); }

            st_read_slice_f32(&S, gu_name, (int64_t)e * GU_ELEMS_PER_EXPERT, GU_ELEMS_PER_EXPERT, gu_buf, 0);
            st_read_slice_f32(&S, d_name,  (int64_t)e * D_ELEMS_PER_EXPERT,  D_ELEMS_PER_EXPERT,  d_buf,  0);

            size_t w_bytes = (fmt == 4) ? 1572864 : 1179648;
            uint8_t *dst_w = malloc(w_bytes);
            float *dst_s = malloc((size_t)GROUPS_PER_EXPERT * sizeof(float));

            if (fmt == 4) {
                quantize_expert_i4(gu_buf, d_buf, dst_w, dst_s);
            } else {
                quantize_expert_i3(gu_buf, d_buf, dst_w, dst_s);
            }

            free(gu_buf);
            free(d_buf);

            w_all[e] = dst_w;
            s_all[e] = dst_s;

            OutTensor *ot_s = &out_tensors[n_out++];
            snprintf(ot_s->name, sizeof(ot_s->name), "model.layers.%d.mlp.experts.%d.qs", l, e);
            strcpy(ot_s->dtype, "F32");
            ot_s->rank = 1;
            ot_s->shape[0] = GROUPS_PER_EXPERT;
            ot_s->nbytes = SCALE_BYTES_PER_EXPERT;
            ot_s->src_buf = dst_s;

            OutTensor *ot_w = &out_tensors[n_out++];
            snprintf(ot_w->name, sizeof(ot_w->name), "model.layers.%d.mlp.experts.%d.merged_weight", l, e);
            strcpy(ot_w->dtype, "U8");
            ot_w->rank = 1;
            ot_w->shape[0] = w_bytes;
            ot_w->nbytes = w_bytes;
            ot_w->src_buf = dst_w;
        }

        /* Copy and convert all dense weights for layer l */
        char prefix1[64], prefix2[64];
        snprintf(prefix1, sizeof(prefix1), "model.language_model.layers.%d.", l);
        snprintf(prefix2, sizeof(prefix2), "model.layers.%d.", l);
        size_t p1_len = strlen(prefix1), p2_len = strlen(prefix2);

        for (int i = 0; i < S.n; i++) {
            const char *tname = S.t[i].name;
            int match = 0;
            const char *clean_suffix = NULL;
            if (strncmp(tname, prefix1, p1_len) == 0 && strstr(tname, ".mlp.experts.") == NULL) {
                match = 1; clean_suffix = tname + p1_len;
            } else if (strncmp(tname, prefix2, p2_len) == 0 && strstr(tname, ".mlp.experts.") == NULL) {
                match = 1; clean_suffix = tname + p2_len;
            }

            if (match) {
                st_tensor *st = &S.t[i];
                OutTensor *ot = &out_tensors[n_out++];
                snprintf(ot->name, sizeof(ot->name), "model.layers.%d.%s", l, clean_suffix);
                strcpy(ot->dtype, "F16");
                ot->rank = st->rank;
                for (int r = 0; r < st->rank; r++) ot->shape[r] = st->shape[r];
                ot->nbytes = st->numel * sizeof(uint16_t);

                float *f32_tmp = malloc((size_t)st->numel * sizeof(float));
                uint16_t *f16_buf = malloc((size_t)st->numel * sizeof(uint16_t));
                if (!f32_tmp || !f16_buf) { fprintf(stderr, "OOM reading dense tensor %s\n", tname); exit(1); }

                st_read_f32(&S, tname, f32_tmp, 0);
                f32_to_f16_array(f32_tmp, f16_buf, st->numel);
                free(f32_tmp);

                dense_blobs[n_dense++] = f16_buf;
                ot->src_buf = f16_buf;
            }
        }

        write_shard(dst_path, out_tensors, n_out);

        for (int e = 0; e < NUM_EXPERTS; e++) { free(w_all[e]); free(s_all[e]); }
        for (int i = 0; i < n_dense; i++) free(dense_blobs[i]);
        printf("[direct-c] wrote layer shard %02d / %02d -> %s (%d tensors)\n", l + 1, NUM_LAYERS, dst_path, n_out);
    }

    /* Copy metadata files */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "cp %s/config.json %s/tokenizer.json /home/nayte/models/qwen36_i3_gs64_clean/qwen36_meta.json %s/ 2>/dev/null || true", src_dir, src_dir, out_dir);
    int rc = system(cmd); (void)rc;
    snprintf(cmd, sizeof(cmd), "cp %s %s/allocation_manifest.json", manifest_path, out_dir);
    rc = system(cmd); (void)rc;

    printf("\n=== Direct BF16 Mixed Container Build Complete: %s ===\n", out_dir);
    free(fmt_map);
    return 0;
}
