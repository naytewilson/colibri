/* Pure Native C Direct-Source BF16 -> INT3-g64 Converter for Qwen3.6-35B-A3B.
 * Converts official BF16 safetensors shards directly to fmt=5 INT3-g64 container.
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
#define GROUPS_PER_EXPERT 49152
#define WEIGHTS_PER_EXPERT 3145728
#define INT3_BYTES_PER_EXPERT 1179648
#define SCALE_BYTES_PER_EXPERT (GROUPS_PER_EXPERT * 4)

#define INTER_SIZE 512
#define HIDDEN_SIZE 2048
#define GU_ELEMS_PER_EXPERT (2 * INTER_SIZE * HIDDEN_SIZE) /* 2,097,152 */
#define D_ELEMS_PER_EXPERT (HIDDEN_SIZE * INTER_SIZE)       /* 1,048,576 */

static inline void quantize_group_f32_to_i3(const float *w, uint8_t *lo, uint8_t *hi, float *s_out) {
    float max_abs = 0.0f;
    for (int k = 0; k < 64; k++) {
        float a = fabsf(w[k]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs <= 1e-12f) {
        *s_out = 1e-8f;
        memset(lo, 0x55, 16);
        memset(hi, 0xFF, 8);
        return;
    }
    float s = max_abs / 3.0f;
    if (s < 1e-12f) s = 1e-12f;
    *s_out = s;
    float inv_s = 1.0f / s;

    memset(lo, 0, 16);
    memset(hi, 0, 8);
    for (int k = 0; k < 64; k++) {
        int v = (int)lrintf(w[k] * inv_s);
        if (v > 3) v = 3;
        if (v < -4) v = -4;
        unsigned u = (unsigned)(v + 4); /* [0..7] */
        lo[k >> 2] |= (uint8_t)((u & 3) << ((k & 3) * 2));
        hi[k >> 3] |= (uint8_t)(((u >> 2) & 1) << (k & 7));
    }
}

static inline void quantize_expert_direct_f32(const float *gu, const float *down, uint8_t *w3, float *s3) {
    /* gu has [2*INTER_SIZE, HIDDEN_SIZE] = [1024, 2048]
     * first 512*2048 is gate, second 512*2048 is up */
    const float *gate = gu;
    const float *up   = gu + (INTER_SIZE * HIDDEN_SIZE);

    int g_idx = 0;

    /* 1. Gate: 512*2048 = 1,048,576 floats -> 16,384 groups */
    for (int i = 0; i < (INTER_SIZE * HIDDEN_SIZE); i += 64) {
        uint8_t *lo = w3 + g_idx * I3_GBYTES;
        uint8_t *hi = lo + 16;
        quantize_group_f32_to_i3(gate + i, lo, hi, &s3[g_idx]);
        g_idx++;
    }

    /* 2. Up: 512*2048 = 1,048,576 floats -> 16,384 groups */
    for (int i = 0; i < (INTER_SIZE * HIDDEN_SIZE); i += 64) {
        uint8_t *lo = w3 + g_idx * I3_GBYTES;
        uint8_t *hi = lo + 16;
        quantize_group_f32_to_i3(up + i, lo, hi, &s3[g_idx]);
        g_idx++;
    }

    /* 3. Down: 2048*512 = 1,048,576 floats -> 16,384 groups */
    for (int i = 0; i < (HIDDEN_SIZE * INTER_SIZE); i += 64) {
        uint8_t *lo = w3 + g_idx * I3_GBYTES;
        uint8_t *hi = lo + 16;
        quantize_group_f32_to_i3(down + i, lo, hi, &s3[g_idx]);
        g_idx++;
    }
}

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
        /* Fallback via bit conversion or single F16C */
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

typedef struct {
    char name[256];
    char dtype[16];
    int rank;
    int64_t shape[4];
    int64_t nbytes;
    int64_t data_off;
    const void *src_buf;
} OutTensor;

static void write_shard(const char *dst_path, OutTensor *tensors, int n_tensors) {
    size_t cap = 2 * 1024 * 1024;
    char *hbuf = malloc(cap);
    size_t pos = 0;
    pos += snprintf(hbuf + pos, cap - pos, "{");

    int64_t cur_off = 0;
    for (int i = 0; i < n_tensors; i++) {
        tensors[i].data_off = cur_off;
        cur_off += tensors[i].nbytes;

        pos += snprintf(hbuf + pos, cap - pos, "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[",
                        (i > 0 ? "," : ""), tensors[i].name, tensors[i].dtype);
        for (int r = 0; r < tensors[i].rank; r++) {
            pos += snprintf(hbuf + pos, cap - pos, "%lld%s",
                            (long long)tensors[i].shape[r], (r < tensors[i].rank - 1 ? "," : ""));
        }
        pos += snprintf(hbuf + pos, cap - pos, "],\"data_offsets\":[%lld,%lld]}",
                        (long long)tensors[i].data_off, (long long)(tensors[i].data_off + tensors[i].nbytes));
    }
    pos += snprintf(hbuf + pos, cap - pos, "}");

    uint64_t hlen = pos;
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

int convert_layer_direct(shards *S, int l, const char *dst_dir) {
    char dst_path[1024];
    snprintf(dst_path, sizeof(dst_path), "%s/model-%05d.safetensors", dst_dir, l);

    char gu_name1[256], gu_name2[256], d_name1[256], d_name2[256];
    snprintf(gu_name1, sizeof(gu_name1), "model.language_model.layers.%d.mlp.experts.gate_up_proj", l);
    snprintf(gu_name2, sizeof(gu_name2), "model.layers.%d.mlp.experts.gate_up_proj", l);
    snprintf(d_name1, sizeof(d_name1), "model.language_model.layers.%d.mlp.experts.down_proj", l);
    snprintf(d_name2, sizeof(d_name2), "model.layers.%d.mlp.experts.down_proj", l);

    const char *gu_name = st_find(S, gu_name1) ? gu_name1 : (st_find(S, gu_name2) ? gu_name2 : NULL);
    const char *d_name  = st_find(S, d_name1) ? d_name1 : (st_find(S, d_name2) ? d_name2 : NULL);

    if (!gu_name || !d_name) {
        fprintf(stderr, "Layer %d: routed experts not found in source shards (%s / %s)\n", l, gu_name1, d_name1);
        return -1;
    }

    /* Allocate memory for 256 INT3 expert weights and scales */
    uint8_t *w3_all = malloc((size_t)NUM_EXPERTS * INT3_BYTES_PER_EXPERT);
    float *s3_all   = malloc((size_t)NUM_EXPERTS * SCALE_BYTES_PER_EXPERT);
    if (!w3_all || !s3_all) { fprintf(stderr, "OOM allocating INT3 buffers\n"); exit(1); }

    #pragma omp parallel for schedule(dynamic)
    for (int e = 0; e < NUM_EXPERTS; e++) {
        float *gu_buf = malloc((size_t)GU_ELEMS_PER_EXPERT * sizeof(float));
        float *d_buf  = malloc((size_t)D_ELEMS_PER_EXPERT * sizeof(float));
        if (!gu_buf || !d_buf) { fprintf(stderr, "OOM allocating expert decode buffers\n"); exit(1); }

        st_read_slice_f32(S, gu_name, (int64_t)e * GU_ELEMS_PER_EXPERT, GU_ELEMS_PER_EXPERT, gu_buf, 0);
        st_read_slice_f32(S, d_name,  (int64_t)e * D_ELEMS_PER_EXPERT,  D_ELEMS_PER_EXPERT,  d_buf,  0);

        uint8_t *w3_cur = w3_all + (size_t)e * INT3_BYTES_PER_EXPERT;
        float *s3_cur   = s3_all + (size_t)e * GROUPS_PER_EXPERT;

        quantize_expert_direct_f32(gu_buf, d_buf, w3_cur, s3_cur);

        free(gu_buf);
        free(d_buf);
    }

    OutTensor out_tensors[600];
    int n_out = 0;

    /* Add 256 qs */
    for (int e = 0; e < NUM_EXPERTS; e++) {
        OutTensor *ot = &out_tensors[n_out++];
        snprintf(ot->name, sizeof(ot->name), "model.layers.%d.mlp.experts.%d.qs", l, e);
        strcpy(ot->dtype, "F32");
        ot->rank = 1;
        ot->shape[0] = GROUPS_PER_EXPERT;
        ot->nbytes = SCALE_BYTES_PER_EXPERT;
        ot->src_buf = s3_all + (size_t)e * GROUPS_PER_EXPERT;
    }

    /* Add 256 merged_weight */
    for (int e = 0; e < NUM_EXPERTS; e++) {
        OutTensor *ot = &out_tensors[n_out++];
        snprintf(ot->name, sizeof(ot->name), "model.layers.%d.mlp.experts.%d.merged_weight", l, e);
        strcpy(ot->dtype, "U8");
        ot->rank = 1;
        ot->shape[0] = INT3_BYTES_PER_EXPERT;
        ot->nbytes = INT3_BYTES_PER_EXPERT;
        ot->src_buf = w3_all + (size_t)e * INT3_BYTES_PER_EXPERT;
    }

    /* Add all dense tensors belonging to layer l in source shard */
    uint16_t *dense_f16_blobs[64];
    int n_dense = 0;

    char prefix1[64], prefix2[64];
    snprintf(prefix1, sizeof(prefix1), "model.language_model.layers.%d.", l);
    snprintf(prefix2, sizeof(prefix2), "model.layers.%d.", l);
    size_t p1_len = strlen(prefix1), p2_len = strlen(prefix2);

    for (int i = 0; i < S->n; i++) {
        const char *tname = S->t[i].name;
        int match = 0;
        const char *clean_suffix = NULL;
        if (strncmp(tname, prefix1, p1_len) == 0 && strstr(tname, ".mlp.experts.") == NULL) {
            match = 1;
            clean_suffix = tname + p1_len;
        } else if (strncmp(tname, prefix2, p2_len) == 0 && strstr(tname, ".mlp.experts.") == NULL) {
            match = 1;
            clean_suffix = tname + p2_len;
        }

        if (match) {
            st_tensor *st = &S->t[i];
            OutTensor *ot = &out_tensors[n_out++];
            snprintf(ot->name, sizeof(ot->name), "model.layers.%d.%s", l, clean_suffix);
            strcpy(ot->dtype, "F16");
            ot->rank = st->rank;
            for (int r = 0; r < st->rank; r++) ot->shape[r] = st->shape[r];
            ot->nbytes = st->numel * sizeof(uint16_t);

            float *f32_tmp = malloc((size_t)st->numel * sizeof(float));
            uint16_t *f16_buf = malloc((size_t)st->numel * sizeof(uint16_t));
            if (!f32_tmp || !f16_buf) { fprintf(stderr, "OOM reading dense tensor %s\n", tname); exit(1); }

            st_read_f32(S, tname, f32_tmp, 0);
            f32_to_f16_array(f32_tmp, f16_buf, st->numel);
            free(f32_tmp);

            dense_f16_blobs[n_dense++] = f16_buf;
            ot->src_buf = f16_buf;
        }
    }

    write_shard(dst_path, out_tensors, n_out);

    for (int i = 0; i < n_dense; i++) free(dense_f16_blobs[i]);
    free(w3_all);
    free(s3_all);

    printf("[direct-c] wrote layer shard %02d / %02d -> %s (%d tensors)\n", l + 1, NUM_LAYERS, dst_path, n_out);
    return 0;
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
    const char *src_dir = argc > 1 ? argv[1] : "/data/ANVIL/models/hf/source_shards";
    const char *dst_dir = argc > 2 ? argv[2] : "/data/ANVIL/models/qwen36_i3_gs64_clean";
    int single_layer    = argc > 3 ? atoi(argv[3]) : -1;

    printf("=== Qwen3.6-35B-A3B Pure Native C Direct BF16 -> INT3-g64 Converter ===\n");
    printf("Source Shards: %s\n", src_dir);
    printf("Destination:   %s\n\n", dst_dir);

    mkdir(dst_dir, 0777);

    shards S;
    memset(&S, 0, sizeof(S));
    st_init(&S, src_dir);
    printf("Indexed %d tensors from source shards\n", S.n);

    if (single_layer >= 0 && single_layer < NUM_LAYERS) {
        convert_layer_direct(&S, single_layer, dst_dir);
    } else {
        convert_globals_direct(&S, dst_dir);
        for (int l = 0; l < NUM_LAYERS; l++) {
            convert_layer_direct(&S, l, dst_dir);
        }
    }

    printf("\n=== Direct C Conversion Pass Complete ===\n");
    return 0;
}
