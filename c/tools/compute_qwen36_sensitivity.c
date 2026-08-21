/* Pure Native C Tool to compute direct BF16 expert sensitivity and GEMQ global Pareto allocation for Qwen3.6-35B-A3B.
 * Reads AUTHORITATIVE BF16 routed expert tensors directly from source shards.
 * Computes direct BF16 -> INT3 and BF16 -> INT4 reconstruction MSE and delta-MSE.
 * Combines with 9-domain routing counts, router mass, and domain coverage.
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

#define NUM_LAYERS 40
#define NUM_EXPERTS 256
#define TOTAL_EXPERTS (NUM_LAYERS * NUM_EXPERTS)

#define INTER_SIZE 512
#define HIDDEN_SIZE 2048
#define GU_ELEMS_PER_EXPERT (2 * INTER_SIZE * HIDDEN_SIZE) /* 2,097,152 */
#define D_ELEMS_PER_EXPERT (HIDDEN_SIZE * INTER_SIZE)       /* 1,048,576 */
#define WEIGHTS_PER_EXPERT (GU_ELEMS_PER_EXPERT + D_ELEMS_PER_EXPERT) /* 3,145,728 */

typedef struct {
    int layer;
    int eid;
    uint64_t total_count;
    double total_mass;
    uint16_t domain_mask;
    int domain_count;
    double mse_int3;
    double mse_int4;
    double delta_mse;
    double composite_score;
    int assigned_fmt; /* 4 = INT4, 5 = INT3 */
    int is_uncertain;
} ExpertStats;

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

static double eval_expert_quant_mse_i3(const float *w, int n) {
    double total_sq_err = 0.0;
    for (int i = 0; i < n; i += 64) {
        uint8_t lo[16], hi[8];
        float s;
        quantize_group_f32_to_i3(w + i, lo, hi, &s);
        for (int k = 0; k < 64; k++) {
            uint8_t lbyte = lo[k >> 2], hbyte = hi[k >> 3];
            uint8_t lbits = (lbyte >> ((k & 3) * 2)) & 3, hbit = (hbyte >> (k & 7)) & 1;
            unsigned u = (unsigned)lbits | ((unsigned)hbit << 2);
            float recon = (float)((int)u - 4) * s;
            float diff = w[i + k] - recon;
            total_sq_err += (double)(diff * diff);
        }
    }
    return total_sq_err / (double)n;
}

static double eval_expert_quant_mse_i4(const float *w, int n) {
    double total_sq_err = 0.0;
    for (int i = 0; i < n; i += 64) {
        uint8_t p4[32];
        float s;
        quantize_group_f32_to_i4(w + i, p4, &s);
        for (int k = 0; k < 64; k += 2) {
            uint8_t b = p4[k >> 1];
            int8_t v0 = (int8_t)(b & 0xF); if (v0 & 8) v0 -= 16;
            int8_t v1 = (int8_t)((b >> 4) & 0xF); if (v1 & 8) v1 -= 16;
            float r0 = (float)v0 * s, r1 = (float)v1 * s;
            float d0 = w[i + k] - r0, d1 = w[i + k + 1] - r1;
            total_sq_err += (double)(d0 * d0 + d1 * d1);
        }
    }
    return total_sq_err / (double)n;
}

static int parse_and_accumulate_census(const char *json_path, int domain_id, ExpertStats *stats) {
    FILE *f = fopen(json_path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz + 1);
    if (!buf) { fclose(f); return 0; }
    if (fread(buf, 1, sz, f) != sz) { free(buf); fclose(f); return 0; }
    buf[sz] = '\0';
    fclose(f);

    char *arena = NULL;
    jval *root = json_parse(buf, &arena);
    if (!root || root->t != J_OBJ) { free(buf); free(arena); return 0; }
    jval *layers = json_get(root, "layers");
    if (!layers || layers->t != J_ARR) { free(buf); free(arena); return 0; }

    for (int l = 0; l < layers->len && l < NUM_LAYERS; l++) {
        jval *layer_obj = layers->kids[l];
        jval *counts = json_get(layer_obj, "expert_counts");
        jval *masses = json_get(layer_obj, "expert_mass");

        if (counts && masses && counts->t == J_ARR && masses->t == J_ARR) {
            for (int e = 0; e < NUM_EXPERTS && e < counts->len && e < masses->len; e++) {
                int idx = l * NUM_EXPERTS + e;
                uint64_t c = (uint64_t)counts->kids[e]->num;
                double m = masses->kids[e]->num;

                stats[idx].total_count += c;
                stats[idx].total_mass += m;
                if (c > 0) {
                    if (!(stats[idx].domain_mask & (1 << domain_id))) {
                        stats[idx].domain_mask |= (1 << domain_id);
                        stats[idx].domain_count++;
                    }
                }
            }
        }
    }

    free(buf);
    free(arena);
    return 1;
}

static int compare_expert_scores(const void *a, const void *b) {
    const ExpertStats *ea = (const ExpertStats *)a;
    const ExpertStats *eb = (const ExpertStats *)b;
    if (ea->composite_score > eb->composite_score) return -1;
    if (ea->composite_score < eb->composite_score) return 1;
    return 0;
}

static void export_manifest(const char *out_path, ExpertStats *stats, int n_int4, const char *profile_name) {
    FILE *f = fopen(out_path, "w");
    if (!f) { perror(out_path); exit(1); }

    int *fmt_map = malloc(TOTAL_EXPERTS * sizeof(int));
    for (int i = 0; i < TOTAL_EXPERTS; i++) fmt_map[i] = 5; /* default INT3 */
    for (int i = 0; i < n_int4 && i < TOTAL_EXPERTS; i++) {
        int idx = stats[i].layer * NUM_EXPERTS + stats[i].eid;
        fmt_map[idx] = 4; /* INT4 */
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"profile\": \"%s\",\n", profile_name);
    fprintf(f, "  \"model\": \"Qwen3.6-35B-A3B\",\n");
    fprintf(f, "  \"num_layers\": %d,\n", NUM_LAYERS);
    fprintf(f, "  \"num_experts\": %d,\n", NUM_EXPERTS);
    fprintf(f, "  \"total_experts\": %d,\n", TOTAL_EXPERTS);
    fprintf(f, "  \"int4_experts\": %d,\n", n_int4);
    fprintf(f, "  \"int3_experts\": %d,\n", TOTAL_EXPERTS - n_int4);
    fprintf(f, "  \"int4_pct\": %.2f,\n", (double)n_int4 * 100.0 / TOTAL_EXPERTS);
    fprintf(f, "  \"allocations\": [\n");

    for (int l = 0; l < NUM_LAYERS; l++) {
        for (int e = 0; e < NUM_EXPERTS; e++) {
            int idx = l * NUM_EXPERTS + e;
            int is_last = (l == NUM_LAYERS - 1 && e == NUM_EXPERTS - 1);
            fprintf(f, "    {\"layer\": %d, \"eid\": %d, \"fmt\": %d}%s\n",
                    l, e, fmt_map[idx], is_last ? "" : ",");
        }
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
    free(fmt_map);
    printf("Exported allocation manifest: %s (%d INT4, %d INT3)\n",
           out_path, n_int4, TOTAL_EXPERTS - n_int4);
}

int main(int argc, char **argv) {
    const char *src_dir = (argc > 1) ? argv[1] : "/data/ANVIL/models/hf/source_shards";
    const char *census_dir = (argc > 2) ? argv[2] : "/home/nayte/census_runs";
    const char *out_dir = (argc > 3) ? argv[3] : "/home/nayte/allocations";
    mkdir(out_dir, 0755);

    printf("=== Qwen3.6 Direct-BF16 Sensitivity Analysis & GEMQ Allocation Solver ===\n");
    printf("Authoritative Source Shards: %s\nCensus Directory: %s | Output Directory: %s\n\n",
           src_dir, census_dir, out_dir);

    ExpertStats *stats = calloc(TOTAL_EXPERTS, sizeof(ExpertStats));
    for (int l = 0; l < NUM_LAYERS; l++) {
        for (int e = 0; e < NUM_EXPERTS; e++) {
            int idx = l * NUM_EXPERTS + e;
            stats[idx].layer = l;
            stats[idx].eid = e;
            stats[idx].assigned_fmt = 5;
        }
    }

    /* 1. Ingest Multi-Domain Routing Census Data (all 9 domains) */
    int loaded_census_count = 0;
    for (int d = 1; d <= 9; d++) {
        char pattern[512];
        snprintf(pattern, sizeof(pattern), "census_p%d_", d);
        DIR *dir = opendir(census_dir);
        if (dir) {
            struct dirent *ent;
            while ((ent = readdir(dir)) != NULL) {
                if (strstr(ent->d_name, pattern) && strstr(ent->d_name, ".json")) {
                    char full_path[1024];
                    snprintf(full_path, sizeof(full_path), "%s/%s", census_dir, ent->d_name);
                    if (parse_and_accumulate_census(full_path, d - 1, stats)) {
                        loaded_census_count++;
                        printf("Parsed domain %d census: %s\n", d, ent->d_name);
                    }
                    break;
                }
            }
            closedir(dir);
        }
    }
    printf("Total Domain Census Files Loaded: %d/9\n\n", loaded_census_count);

    /* 2. Compute Direct BF16 -> INT3 and BF16 -> INT4 Reconstruction MSE */
    shards S;
    st_init(&S, src_dir);
    printf("Computing direct BF16->INT3 and BF16->INT4 reconstruction errors across all %d experts...\n", TOTAL_EXPERTS);

    #pragma omp parallel for schedule(dynamic)
    for (int l = 0; l < NUM_LAYERS; l++) {
        char gu_name1[256], gu_name2[256], d_name1[256], d_name2[256];
        snprintf(gu_name1, sizeof(gu_name1), "model.language_model.layers.%d.mlp.experts.gate_up_proj", l);
        snprintf(gu_name2, sizeof(gu_name2), "model.layers.%d.mlp.experts.gate_up_proj", l);
        snprintf(d_name1, sizeof(d_name1), "model.language_model.layers.%d.mlp.experts.down_proj", l);
        snprintf(d_name2, sizeof(d_name2), "model.layers.%d.mlp.experts.down_proj", l);

        const char *gu_name = st_find(&S, gu_name1) ? gu_name1 : (st_find(&S, gu_name2) ? gu_name2 : NULL);
        const char *d_name  = st_find(&S, d_name1) ? d_name1 : (st_find(&S, d_name2) ? d_name2 : NULL);

        if (!gu_name || !d_name) {
            fprintf(stderr, "Layer %d: routed experts not found in BF16 source shards\n", l);
            exit(1);
        }

        float *gu = malloc((size_t)GU_ELEMS_PER_EXPERT * sizeof(float));
        float *down = malloc((size_t)D_ELEMS_PER_EXPERT * sizeof(float));
        if (!gu || !down) { fprintf(stderr, "OOM in sensitivity buffers\n"); exit(1); }

        for (int e = 0; e < NUM_EXPERTS; e++) {
            int idx = l * NUM_EXPERTS + e;

            st_read_slice_f32(&S, gu_name, (int64_t)e * GU_ELEMS_PER_EXPERT, GU_ELEMS_PER_EXPERT, gu, 0);
            st_read_slice_f32(&S, d_name,  (int64_t)e * D_ELEMS_PER_EXPERT,  D_ELEMS_PER_EXPERT,  down,  0);

            double mse3_gu = eval_expert_quant_mse_i3(gu, GU_ELEMS_PER_EXPERT);
            double mse3_d  = eval_expert_quant_mse_i3(down, D_ELEMS_PER_EXPERT);
            double mse3    = (mse3_gu * 2.0 + mse3_d) / 3.0;

            double mse4_gu = eval_expert_quant_mse_i4(gu, GU_ELEMS_PER_EXPERT);
            double mse4_d  = eval_expert_quant_mse_i4(down, D_ELEMS_PER_EXPERT);
            double mse4    = (mse4_gu * 2.0 + mse4_d) / 3.0;

            stats[idx].mse_int3 = mse3;
            stats[idx].mse_int4 = mse4;
            stats[idx].delta_mse = (mse3 > mse4) ? (mse3 - mse4) : 0.0;
        }

        free(gu);
        free(down);
    }

    /* 3. Compute Composite Importance Scores */
    int active_experts = 0, uncertain_experts = 0;
    for (int i = 0; i < TOTAL_EXPERTS; i++) {
        if (stats[i].total_count > 0) {
            active_experts++;
            double domain_multiplier = 1.0 + (0.15 * (double)stats[i].domain_count);
            stats[i].composite_score = stats[i].total_mass * stats[i].delta_mse * domain_multiplier;
            stats[i].is_uncertain = 0;
        } else {
            uncertain_experts++;
            stats[i].is_uncertain = 1;
            stats[i].composite_score = 0.25 * stats[i].delta_mse;
        }
    }

    /* 4. Global Pareto Ranking */
    ExpertStats *sorted_stats = malloc(TOTAL_EXPERTS * sizeof(ExpertStats));
    memcpy(sorted_stats, stats, TOTAL_EXPERTS * sizeof(ExpertStats));
    qsort(sorted_stats, TOTAL_EXPERTS, sizeof(ExpertStats), compare_expert_scores);

    printf("\n========================================================================================\n");
    printf("=== TOP 20 MOST SENSITIVE EXPERTS (GLOBAL GEMQ DIRECT-BF16 RANKING) ===\n");
    printf("========================================================================================\n");
    printf("%-5s | %-6s | %-5s | %-8s | %-8s | %-6s | %-10s | %-10s | %-10s\n",
           "Rank", "Layer", "EID", "Count", "Mass", "Domain", "MSE(INT3)", "MSE(INT4)", "Score");
    printf("----------------------------------------------------------------------------------------\n");
    for (int i = 0; i < 20; i++) {
        printf("#%-4d | L%-5d | E%-4d | %8lu | %8.3f | %6d | %10.6f | %10.6f | %10.6f\n",
               i + 1, sorted_stats[i].layer, sorted_stats[i].eid,
               (unsigned long)sorted_stats[i].total_count,
               sorted_stats[i].total_mass,
               sorted_stats[i].domain_count,
               sorted_stats[i].mse_int3,
               sorted_stats[i].mse_int4,
               sorted_stats[i].composite_score);
    }
    printf("========================================================================================\n\n");

    /* Export Allocation Profiles */
    char out_mixed_low[1024];
    snprintf(out_mixed_low, sizeof(out_mixed_low), "%s/allocation_mixed_low.json", out_dir);
    export_manifest(out_mixed_low, sorted_stats, 1536, "MIXED-LOW-15pct-INT4-DIRECT-BF16");

    free(stats);
    free(sorted_stats);
    return 0;
}
