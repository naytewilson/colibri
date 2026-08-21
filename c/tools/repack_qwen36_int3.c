/* Native C repack tool: Qwen3.6-35B-A3B INT4 gs=64 -> fmt=5 INT3-g64 container.
 * Converts 40 layer shards model-00000.safetensors .. model-00039.safetensors.
 * Expert weights: 1.572 MB/expert (INT4) -> 1.179 MB/expert (INT3-g64).
 * Expert scales: 196.6 KB/expert (49,152 FP32 group scales, gs=64) preserved/re-estimated.
 * Dense weights: copied bit-identical.
 * NO Python dependency. Compiles with gcc -O3 -fopenmp.
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
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../st.h"
#include "../json.h"

static double jnum(jval *o, const char *k) { jval *v = json_get(o, k); return (v && v->t == J_NUM) ? v->num : 0; }

#define I3_GROUP 64
#define I3_GBYTES 24
#define NUM_LAYERS 40
#define NUM_EXPERTS 256
#define GROUPS_PER_EXPERT 49152
#define WEIGHTS_PER_EXPERT 3145728
#define INT4_BYTES_PER_EXPERT 1572864
#define INT3_BYTES_PER_EXPERT 1179648
#define SCALE_BYTES_PER_EXPERT (GROUPS_PER_EXPERT * 4)

static void quantize_expert_i4_to_i3(const uint8_t *w4, const float *s4, uint8_t *w3, float *s3) {
    for (int g = 0; g < GROUPS_PER_EXPERT; g++) {
        float s_orig = s4[g];
        int8_t q4_vals[64];
        int qmax = 0;
        int elem_base = g * 64;
        for (int k = 0; k < 64; k++) {
            int idx = elem_base + k;
            uint8_t b = w4[idx >> 1];
            int8_t v = (idx & 1) ? (int8_t)((b >> 4) & 0xF) : (int8_t)(b & 0xF);
            if (v >= 8) v -= 16;
            q4_vals[k] = v;
            int ab = abs(v);
            if (ab > qmax) qmax = ab;
        }

        uint8_t *lo = w3 + g * I3_GBYTES;
        uint8_t *hi = lo + 16;
        memset(lo, 0, I3_GBYTES);

        if (qmax == 0) {
            s3[g] = 1e-8f;
        } else {
            float s_new = ((float)qmax * s_orig) / 3.0f;
            if (s_new < 1e-8f) s_new = 1e-8f;
            s3[g] = s_new;
            float inv_s = 1.0f / s_new;
            for (int k = 0; k < 64; k++) {
                float w = (float)q4_vals[k] * s_orig;
                int v = (int)lrintf(w * inv_s);
                if (v > 3) v = 3;
                if (v < -4) v = -4;
                unsigned u = (unsigned)(v + 4); /* [0, 7] */
                lo[k >> 2] |= (uint8_t)((u & 3) << ((k & 3) * 2));
                hi[k >> 3] |= (uint8_t)(((u >> 2) & 1) << (k & 7));
            }
        }
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
    /* Build JSON header */
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

int main(int argc, char **argv) {
    const char *src_dir = argc > 1 ? argv[1] : "/home/nayte/models/qwen36_i4_gs64";
    const char *dst_dir = argc > 2 ? argv[2] : "/home/nayte/models/qwen36_i3_gs64";

    printf("== Qwen3.6-35B-A3B INT4 -> fmt=5 INT3-g64 Native C Repack ==\n");
    printf("Source: %s\nDestination: %s\n\n", src_dir, dst_dir);

    mkdir(dst_dir, 0777);

    shards S;
    memset(&S, 0, sizeof(S));
    st_init(&S, src_dir); /* opens and indexes once */

    /* Process all 40 layer shards */
    for (int l = 0; l < NUM_LAYERS; l++) {
        char src_path[1024], dst_path[1024];
        snprintf(src_path, sizeof(src_path), "%s/model-%05d.safetensors", src_dir, l);
        snprintf(dst_path, sizeof(dst_path), "%s/model-%05d.safetensors", dst_dir, l);

        /* Allocate memory for 256 INT3 expert weights and scales */
        uint8_t *w3_all = malloc((size_t)NUM_EXPERTS * INT3_BYTES_PER_EXPERT);
        float *s3_all   = malloc((size_t)NUM_EXPERTS * SCALE_BYTES_PER_EXPERT);
        if (!w3_all || !s3_all) { fprintf(stderr, "OOM allocating INT3 buffers\n"); exit(1); }

        #pragma omp parallel for schedule(dynamic)
        for (int e = 0; e < NUM_EXPERTS; e++) {
            char wname[256], sname[256];
            snprintf(wname, sizeof(wname), "model.layers.%d.mlp.experts.%d.merged_weight", l, e);
            snprintf(sname, sizeof(sname), "model.layers.%d.mlp.experts.%d.qs", l, e);

            st_tensor *tw = st_find(&S, wname);
            st_tensor *ts = st_find(&S, sname);
            if (!tw || !ts) { fprintf(stderr, "Layer %d expert %d missing in %s\n", l, e, src_path); exit(1); }

            uint8_t *w4 = malloc(tw->nbytes);
            float *s4   = malloc(ts->nbytes);
            st_pread_full(tw->fd, w4, tw->nbytes, tw->off, "pread w4");
            st_pread_full(ts->fd, s4, ts->nbytes, ts->off, "pread s4");

            uint8_t *w3_cur = w3_all + (size_t)e * INT3_BYTES_PER_EXPERT;
            float *s3_cur   = s3_all + (size_t)e * GROUPS_PER_EXPERT;

            quantize_expert_i4_to_i3(w4, s4, w3_cur, s3_cur);

            free(w4);
            free(s4);
        }

        /* Collect all out tensors: 256 qs, 256 merged_weight, + dense tensors for this layer */
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
        uint8_t *dense_blobs[64];
        int n_dense = 0;

        char layer_prefix[64];
        snprintf(layer_prefix, sizeof(layer_prefix), "model.layers.%d.", l);
        size_t lp_len = strlen(layer_prefix);

        for (int i = 0; i < S.n; i++) {
            if (strncmp(S.t[i].name, layer_prefix, lp_len) == 0 &&
                strstr(S.t[i].name, ".mlp.experts.") == NULL) {
                st_tensor *st = &S.t[i];
                OutTensor *ot = &out_tensors[n_out++];
                snprintf(ot->name, sizeof(ot->name), "%s", st->name);
                snprintf(ot->dtype, sizeof(ot->dtype), "%s", st_dtype_name(st->dtype));
                ot->rank = st->rank;
                for (int r = 0; r < st->rank; r++) ot->shape[r] = st->shape[r];
                ot->nbytes = st->nbytes;

                uint8_t *dbuf = malloc(st->nbytes);
                st_pread_full(st->fd, dbuf, st->nbytes, st->off, "read dense");
                dense_blobs[n_dense++] = dbuf;
                ot->src_buf = dbuf;
            }
        }

        write_shard(dst_path, out_tensors, n_out);

        for (int i = 0; i < n_dense; i++) free(dense_blobs[i]);
        free(w3_all);
        free(s3_all);

        printf("[repack] wrote layer shard %d / %d -> %s (%d tensors)\n", l + 1, NUM_LAYERS, dst_path, n_out);
    }

    /* Copy model-globals.safetensors */
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "cp %s/model-globals.safetensors %s/config.json %s/tokenizer.json %s/",
             src_dir, src_dir, src_dir, dst_dir);
    system(cmd);

    /* Write updated qwen36_meta.json with ebits=3, expert_fmt=5, expert_gs=64 */
    char meta_src[1024], meta_dst[1024];
    snprintf(meta_src, sizeof(meta_src), "%s/qwen36_meta.json", src_dir);
    snprintf(meta_dst, sizeof(meta_dst), "%s/qwen36_meta.json", dst_dir);
    FILE *mf = fopen(meta_src, "rb");
    if (mf) {
        fseek(mf, 0, SEEK_END); long sz = ftell(mf); fseek(mf, 0, SEEK_SET);
        char *mbuf = malloc(sz + 1);
        fread(mbuf, 1, sz, mf); mbuf[sz] = 0; fclose(mf);
        char *arena = NULL;
        jval *j = json_parse(mbuf, &arena);
        if (j && j->t == J_OBJ) {
            FILE *mo = fopen(meta_dst, "wb");
            fprintf(mo, "{\n");
            fprintf(mo, "  \"hidden_size\": %d,\n", (int)jnum(j, "hidden_size"));
            fprintf(mo, "  \"num_hidden_layers\": %d,\n", (int)jnum(j, "num_hidden_layers"));
            fprintf(mo, "  \"vocab_size\": %d,\n", (int)jnum(j, "vocab_size"));
            fprintf(mo, "  \"num_experts\": %d,\n", (int)jnum(j, "num_experts"));
            fprintf(mo, "  \"topk\": %d,\n", (int)jnum(j, "topk"));
            fprintf(mo, "  \"moe_inter\": %d,\n", (int)jnum(j, "moe_inter"));
            fprintf(mo, "  \"shared_inter\": %d,\n", (int)jnum(j, "shared_inter"));
            fprintf(mo, "  \"q_heads\": %d,\n", (int)jnum(j, "q_heads"));
            fprintf(mo, "  \"kv_heads\": %d,\n", (int)jnum(j, "kv_heads"));
            fprintf(mo, "  \"head_dim\": %d,\n", (int)jnum(j, "head_dim"));
            fprintf(mo, "  \"q_head_dim\": %d,\n", (int)jnum(j, "q_head_dim"));
            fprintf(mo, "  \"k_head_dim\": %d,\n", (int)jnum(j, "k_head_dim"));
            fprintf(mo, "  \"v_head_dim\": %d,\n", (int)jnum(j, "v_head_dim"));
            fprintf(mo, "  \"o_in\": %d,\n", (int)jnum(j, "o_in"));
            fprintf(mo, "  \"qk_rope_head_dim\": %d,\n", (int)jnum(j, "qk_rope_head_dim"));
            fprintf(mo, "  \"partial_rotary_factor\": %.4f,\n", (float)jnum(j, "partial_rotary_factor"));
            fprintf(mo, "  \"rope_theta\": %.1f,\n", (float)jnum(j, "rope_theta"));
            fprintf(mo, "  \"rms_eps\": %.6f,\n", (float)jnum(j, "rms_eps"));
            fprintf(mo, "  \"attn_output_gate\": 1,\n");
            fprintf(mo, "  \"has_qk_norm\": 1,\n");
            fprintf(mo, "  \"dn_vheads\": %d,\n", (int)jnum(j, "dn_vheads"));
            fprintf(mo, "  \"dn_kheads\": %d,\n", (int)jnum(j, "dn_kheads"));
            fprintf(mo, "  \"dn_kdim\": %d,\n", (int)jnum(j, "dn_kdim"));
            fprintf(mo, "  \"dn_vdim\": %d,\n", (int)jnum(j, "dn_vdim"));
            fprintf(mo, "  \"dn_convk\": %d,\n", (int)jnum(j, "dn_convk"));
            fprintf(mo, "  \"dn_conv_dim\": %d,\n", (int)jnum(j, "dn_conv_dim"));
            fprintf(mo, "  \"ebits\": 3,\n");
            fprintf(mo, "  \"expert_fmt\": 5,\n");
            fprintf(mo, "  \"expert_gs\": 64\n");
            fprintf(mo, "}\n");
            fclose(mo);
        }
        free(mbuf); free(arena);
    }

    printf("\n== Repack to fmt=5 INT3-g64 complete: %s ==\n", dst_dir);
    return 0;
}
