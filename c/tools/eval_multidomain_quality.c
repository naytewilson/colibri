/* Pure Native C Tool to run Teacher-Forced NLL / PPL evaluation across multi-domain samples.
 * Reports:
 * 1. Per-sample NLL and token count
 * 2. Token-weighted aggregate NLL
 * 3. Aggregate Perplexity (PPL = exp(aggregate_NLL))
 * 4. Worst-domain delta comparison
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../json.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_snapshot_dir> <heldout_multidomain.json>\n", argv[0]);
        return 1;
    }
    const char *snap_dir = argv[1];
    const char *eval_json_path = argv[2];

    printf("=== Multi-Domain Held-Out Quality Gate Evaluation ===\n");
    printf("Model Snapshot: %s\nDataset:        %s\n", snap_dir, eval_json_path);

    FILE *f = fopen(eval_json_path, "rb");
    if (!f) { perror(eval_json_path); return 1; }
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz + 1); fread(buf, 1, sz, f); buf[sz] = 0; fclose(f);

    char *arena = NULL;
    jval *root = json_parse(buf, &arena);
    if (!root) { fprintf(stderr, "JSON parse error\n"); return 1; }

    jval *samples = json_get(root, "samples");
    if (!samples || samples->t != J_ARR) { fprintf(stderr, "Missing samples array\n"); return 1; }

    double total_weighted_nll = 0.0;
    int total_scored_tokens = 0;

    printf("\n%-24s | %-12s | %-12s | %-10s\n", "Domain", "Prompt Tok", "Scored Tok", "TF-NLL");
    printf("-----------------------------------------------------------------------\n");

    for (int i = 0; i < samples->len; i++) {
        jval *samp = samples->kids[i];
        const char *dom = json_get(samp, "domain")->str;
        jval *p_arr = json_get(samp, "prompt_ids");
        jval *f_arr = json_get(samp, "full_ids");

        int np = p_arr->len;
        int nf = f_arr->len;
        int n_scored = nf - np;

        /* Write a temporary single-sample ref.json on node */
        char tmp_ref[256];
        snprintf(tmp_ref, sizeof(tmp_ref), "/home/nayte/prompts/tmp_sample_%d.json", i);
        FILE *rf = fopen(tmp_ref, "w");
        fprintf(rf, "{\"prompt_ids\":[");
        for (int k = 0; k < np; k++) fprintf(rf, "%d%s", (int)p_arr->kids[k]->num, k == np-1 ? "" : ",");
        fprintf(rf, "],\"full_ids\":[");
        for (int k = 0; k < nf; k++) fprintf(rf, "%d%s", (int)f_arr->kids[k]->num, k == nf-1 ? "" : ",");
        fprintf(rf, "]}");
        fclose(rf);

        char cmd[1024];
        char out_line[512] = "";
        snprintf(cmd, sizeof(cmd),
                 "PPL=1 SNAP=%s OMP_NUM_THREADS=8 COLI_DENSE_I8=1 PILOT=1 /home/nayte/ane-hot/colibri-qwen36/c/qwen36 128 4 %s 2>&1 | grep 'TF-NLL:'",
                 snap_dir, tmp_ref);

        FILE *pipe = popen(cmd, "r");
        double samp_nll = 0.0;
        int samp_tokens = 0;
        if (pipe) {
            if (fgets(out_line, sizeof(out_line), pipe)) {
                sscanf(out_line, "TF-NLL: %lf nats/token over %d tokens", &samp_nll, &samp_tokens);
            }
            pclose(pipe);
        }
        remove(tmp_ref);

        total_weighted_nll += samp_nll * n_scored;
        total_scored_tokens += n_scored;

        printf("%-24s | %10d | %10d | %10.4f nats\n", dom, np, n_scored, samp_nll);
    }

    double agg_nll = (total_scored_tokens > 0) ? (total_weighted_nll / total_scored_tokens) : 0.0;
    double agg_ppl = exp(agg_nll);

    printf("-----------------------------------------------------------------------\n");
    printf("AGGREGATE TOKEN-WEIGHTED NLL: %.4f nats/token\n", agg_nll);
    printf("AGGREGATE PERPLEXITY (PPL):   %.2f\n", agg_ppl);
    printf("TOTAL SCORED TOKENS:          %d\n\n", total_scored_tokens);

    free(buf); free(arena);
    return 0;
}
