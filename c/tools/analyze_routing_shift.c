/* Pure Native C Tool to measure Aggregate Expert Count Overlap between Control and Mixed Containers */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../json.h"

#define NUM_LAYERS 40
#define NUM_EXPERTS 256

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <census_1.json> <census_2.json>\n", argv[0]);
        return 1;
    }
    printf("=== Analyzing AGGREGATE_EXPERT_COUNT_OVERLAP ===\n");
    printf("Census 1: %s\nCensus 2: %s\n", argv[1], argv[2]);

    FILE *f1 = fopen(argv[1], "rb");
    FILE *f2 = fopen(argv[2], "rb");
    if (!f1 || !f2) { fprintf(stderr, "Error opening census files\n"); return 1; }

    fseek(f1, 0, SEEK_END); size_t s1 = ftell(f1); fseek(f1, 0, SEEK_SET);
    char *b1 = malloc(s1 + 1); if (fread(b1, 1, s1, f1) != s1) {} b1[s1] = 0; fclose(f1);

    fseek(f2, 0, SEEK_END); size_t s2 = ftell(f2); fseek(f2, 0, SEEK_SET);
    char *b2 = malloc(s2 + 1); if (fread(b2, 1, s2, f2) != s2) {} b2[s2] = 0; fclose(f2);

    char *a1 = NULL, *a2 = NULL;
    jval *r1 = json_parse(b1, &a1);
    jval *r2 = json_parse(b2, &a2);

    if (!r1 || !r2) { fprintf(stderr, "JSON parse error\n"); return 1; }

    jval *l1 = json_get(r1, "layers");
    jval *l2 = json_get(r2, "layers");

    int total_activations_1 = 0, total_activations_2 = 0;
    int common_activations = 0;

    for (int l = 0; l < NUM_LAYERS && l < l1->len && l < l2->len; l++) {
        jval *lay1 = l1->kids[l];
        jval *lay2 = l2->kids[l];
        jval *c1 = json_get(lay1, "expert_counts");
        jval *c2 = json_get(lay2, "expert_counts");
        if (c1 && c2 && c1->t == J_ARR && c2->t == J_ARR) {
            for (int e = 0; e < NUM_EXPERTS && e < c1->len && e < c2->len; e++) {
                int cnt1 = (int)c1->kids[e]->num;
                int cnt2 = (int)c2->kids[e]->num;
                total_activations_1 += cnt1;
                total_activations_2 += cnt2;
                if (cnt1 > 0 && cnt2 > 0) {
                    common_activations += (cnt1 < cnt2) ? cnt1 : cnt2;
                }
            }
        }
    }

    double agreement = (total_activations_1 > 0) ? (double)common_activations * 100.0 / total_activations_1 : 100.0;
    printf("Total Activations C1: %d | C2: %d\n", total_activations_1, total_activations_2);
    printf("AGGREGATE_EXPERT_COUNT_OVERLAP / Overlap: %.2f%%\n", agreement);
    printf("Router Stability Verdict: %s\n", agreement >= 95.0 ? "STABLE_ROUTING (Agreement >= 95%)" : "SHIFT_DETECTED");

    free(b1); free(b2); free(a1); free(a2);
    return 0;
}
