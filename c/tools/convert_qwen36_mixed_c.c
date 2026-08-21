/* Pure Native C Direct-Source Mixed INT3/INT4-g64 Container Builder for Qwen3.6-35B-A3B.
 * Converts source container to a heterogeneous GEMQ-style container
 * based on an allocation manifest JSON. Includes all dense weights, global embeddings,
 * metadata, and tokenizer.
 *
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

typedef struct {
    char name[256];
    char dtype[16];
    int rank;
    int64_t shape[4];
    int64_t nbytes;
    void *src_buf;
} OutTensor;

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

/* Load allocation map: layer*256+eid -> fmt (4 or 5) */
static int load_allocation_map(const char *manifest_path, int *fmt_map) {
    for (int i = 0; i < TOTAL_EXPERTS; i++) fmt_map[i] = 5; /* default INT3 */
    char *buf = NULL; size_t sz = 0;
    FILE *f = fopen(manifest_path, "rb");
    if (!f) { fprintf(stderr, "Error: Could not open manifest %s\n", manifest_path); return 0; }
    fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
    buf = malloc(sz + 1);
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
    printf("Loaded allocation map: %d INT4 experts, %d INT3 experts (Total %d)\n",
           count_i4, count_i3, count_i4 + count_i3);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <source_model_dir> <manifest.json> <output_dir>\n", argv[0]);
        return 1;
    }
    const char *src_dir = argv[1];
    const char *manifest_path = argv[2];
    const char *out_dir = argv[3];
    mkdir(out_dir, 0755);

    printf("=== Pure C Mixed INT3/INT4 Container Builder ===\n");
    printf("Source Model: %s\nManifest: %s\nOutput Dir: %s\n", src_dir, manifest_path, out_dir);

    int *fmt_map = malloc(TOTAL_EXPERTS * sizeof(int));
    if (!load_allocation_map(manifest_path, fmt_map)) {
        return 1;
    }

    shards S;
    st_init(&S, src_dir);

    #pragma omp parallel for schedule(dynamic)
    for (int l = 0; l < NUM_LAYERS; l++) {
        char dst_path[1024];
        snprintf(dst_path, sizeof(dst_path), "%s/model-%05d.safetensors", out_dir, l);

        OutTensor out_tensors[600];
        int n_out = 0;

        uint8_t *w_all[NUM_EXPERTS];
        float *s_all[NUM_EXPERTS];
        float *dense_blobs[64];
        int n_dense = 0;

        /* 1. Quantize each expert according to its assigned format */
        float *gu = malloc(GU_ELEMS_PER_EXPERT * sizeof(float));
        float *down = malloc(D_ELEMS_PER_EXPERT * sizeof(float));

        for (int e = 0; e < NUM_EXPERTS; e++) {
            int idx = l * NUM_EXPERTS + e;
            int fmt = fmt_map[idx];

            char nm[256], qsnm[256];
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.merged_weight", l, e);
            snprintf(qsnm, sizeof(qsnm), "model.layers.%d.mlp.experts.%d.qs", l, e);

            st_tensor *tw = st_find(&S, nm);
            st_tensor *ts = st_find(&S, qsnm);
            if (!tw || !ts) {
                fprintf(stderr, "Missing expert %s in source\n", nm); exit(1);
            }

            uint8_t *raw_src = malloc(tw->nbytes);
            st_read_raw(&S, nm, raw_src, 0);
            float *scales_src = malloc(49152 * sizeof(float));
            st_read_f32(&S, qsnm, scales_src, 0);

            /* Dequantize source */
            int g_idx = 0;
            if (tw->nbytes == 1572864) {
                for (int i = 0; i < GU_ELEMS_PER_EXPERT; i += 64) {
                    float sc = scales_src[g_idx++];
                    for (int k = 0; k < 64; k++) {
                        int byte_idx = (i + k) >> 1;
                        uint8_t b = raw_src[byte_idx];
                        int8_t v = (int8_t)(((i + k) & 1) ? ((b >> 4) & 0xF) : (b & 0xF));
                        if (v & 8) v -= 16;
                        gu[i + k] = (float)v * sc;
                    }
                }
                for (int i = 0; i < D_ELEMS_PER_EXPERT; i += 64) {
                    float sc = scales_src[g_idx++];
                    for (int k = 0; k < 64; k++) {
                        int byte_idx = (GU_ELEMS_PER_EXPERT / 2) + ((i + k) >> 1);
                        uint8_t b = raw_src[byte_idx];
                        int8_t v = (int8_t)(((i + k) & 1) ? ((b >> 4) & 0xF) : (b & 0xF));
                        if (v & 8) v -= 16;
                        down[i + k] = (float)v * sc;
                    }
                }
            } else if (tw->nbytes == 1179648) {
                for (int i = 0; i < GU_ELEMS_PER_EXPERT; i += 64) {
                    float sc = scales_src[g_idx];
                    const uint8_t *lo = raw_src + g_idx * I3_GBYTES, *hi = lo + 16;
                    g_idx++;
                    for (int k = 0; k < 64; k++) {
                        uint8_t lbyte = lo[k >> 2], hbyte = hi[k >> 3];
                        uint8_t lbits = (lbyte >> ((k & 3) * 2)) & 3, hbit = (hbyte >> (k & 7)) & 1;
                        unsigned u = (unsigned)lbits | ((unsigned)hbit << 2);
                        gu[i + k] = (float)((int)u - 4) * sc;
                    }
                }
                for (int i = 0; i < D_ELEMS_PER_EXPERT; i += 64) {
                    float sc = scales_src[g_idx];
                    const uint8_t *lo = raw_src + g_idx * I3_GBYTES, *hi = lo + 16;
                    g_idx++;
                    for (int k = 0; k < 64; k++) {
                        uint8_t lbyte = lo[k >> 2], hbyte = hi[k >> 3];
                        uint8_t lbits = (lbyte >> ((k & 3) * 2)) & 3, hbit = (hbyte >> (k & 7)) & 1;
                        unsigned u = (unsigned)lbits | ((unsigned)hbit << 2);
                        down[i + k] = (float)((int)u - 4) * sc;
                    }
                }
            }
            free(raw_src);
            free(scales_src);

            /* Target format quantization */
            size_t w_bytes = (fmt == 4) ? 1572864 : 1179648;
            uint8_t *dst_w = malloc(w_bytes);
            float *dst_s = malloc(49152 * sizeof(float));

            if (fmt == 4) {
                quantize_expert_i4(gu, down, dst_w, dst_s);
            } else {
                quantize_expert_i3(gu, down, dst_w, dst_s);
            }

            w_all[e] = dst_w;
            s_all[e] = dst_s;

            /* Add to OutTensor list */
            OutTensor *ot_s = &out_tensors[n_out++];
            snprintf(ot_s->name, sizeof(ot_s->name), "model.layers.%d.mlp.experts.%d.qs", l, e);
            strcpy(ot_s->dtype, "F32");
            ot_s->rank = 1;
            ot_s->shape[0] = 49152;
            ot_s->nbytes = 49152 * sizeof(float);
            ot_s->src_buf = dst_s;

            OutTensor *ot_w = &out_tensors[n_out++];
            snprintf(ot_w->name, sizeof(ot_w->name), "model.layers.%d.mlp.experts.%d.merged_weight", l, e);
            strcpy(ot_w->dtype, "U8");
            ot_w->rank = 1;
            ot_w->shape[0] = w_bytes;
            ot_w->nbytes = w_bytes;
            ot_w->src_buf = dst_w;
        }
        free(gu);
        free(down);

        /* 2. Copy all dense weights for layer l */
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "model.layers.%d.", l);
        size_t p_len = strlen(prefix);

        for (int i = 0; i < S.n; i++) {
            const char *tname = S.t[i].name;
            if (strncmp(tname, prefix, p_len) == 0 && strstr(tname, ".mlp.experts.") == NULL) {
                st_tensor *st = &S.t[i];
                OutTensor *ot = &out_tensors[n_out++];
                strcpy(ot->name, tname);
                strcpy(ot->dtype, (st->dtype == 2) ? "F32" : (st->dtype == 1) ? "F16" : "U8");
                ot->rank = st->rank;
                for (int r = 0; r < st->rank; r++) ot->shape[r] = st->shape[r];
                ot->nbytes = st->nbytes;

                void *dense_buf = malloc(st->nbytes);
                st_read_raw(&S, tname, dense_buf, 0);
                dense_blobs[n_dense++] = (float*)dense_buf;
                ot->src_buf = dense_buf;
            }
        }

        write_shard(dst_path, out_tensors, n_out);

        for (int e = 0; e < NUM_EXPERTS; e++) { free(w_all[e]); free(s_all[e]); }
        for (int i = 0; i < n_dense; i++) free(dense_blobs[i]);
        printf("[Layer %02d/40] Wrote mixed shard: %s (%d tensors)\n", l, dst_path, n_out);
    }

    /* 3. Copy globals shard */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "cp %s/model-globals.safetensors %s/model-globals.safetensors 2>/dev/null || true", src_dir, out_dir);
    int rc = system(cmd); (void)rc;

    /* 4. Copy config.json, tokenizer.json, qwen36_meta.json, manifest */
    snprintf(cmd, sizeof(cmd), "cp %s/config.json %s/tokenizer.json %s/qwen36_meta.json %s/ 2>/dev/null || true", src_dir, src_dir, src_dir, out_dir);
    rc = system(cmd); (void)rc;
    snprintf(cmd, sizeof(cmd), "cp %s %s/allocation_manifest.json", manifest_path, out_dir);
    rc = system(cmd); (void)rc;

    printf("=== Mixed Container Build Complete: %s ===\n", out_dir);
    free(fmt_map);
    return 0;
}
