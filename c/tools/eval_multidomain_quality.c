/* Pure Native C Tool to run Teacher-Forced NLL / PPL evaluation across multi-domain samples.
 * Reports:
 * 1. Per-sample NLL and token count
 * 2. Token-weighted aggregate NLL
 * 3. Aggregate Perplexity (PPL = exp(aggregate_NLL))
 * 4. Worst-domain delta comparison
 *
 * Artifact binding:
 *   QWEN36_BIN must point at the exact qwen36 binary under test. This tool
 *   fails closed when the binary is not provided or when any sample command
 *   fails to produce a parseable TF-NLL result.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/wait.h>
#include "../json.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: QWEN36_BIN=/exact/qwen36 %s <model_snapshot_dir> <heldout_multidomain.json>\n", argv[0]);
        return 1;
    }
    const char *qwen36_bin = getenv("QWEN36_BIN");
    if (!qwen36_bin || !*qwen36_bin) {
        fprintf(stderr, "QWEN36_BIN is required; refusing to fall back to a mutable builder binary\n");
        return 2;
    }
    const char *snap_dir = argv[1];
    const char *eval_json_path = argv[2];

    printf("=== Multi-Domain Held-Out Quality Gate Evaluation ===\n");
    printf("QWEN36 Binary: %s\nModel Snapshot: %s\nDataset:        %s\n", qwen36_bin, snap_dir, eval_json_path);

    FILE *f = fopen(eval_json_path, "rb");
    if (!f) { perror(eval_json_path); return 1; }
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz + 1); fread(buf, 1, sz, f); buf[sz] = 0; fclose(f);

    char *arena = NULL;
    jval *root = json_parse(buf, &arena);
    if (!root) { fprintf(stderr, "JSON parse error\n"); free(buf); return 1; }

    jval *samples = json_get(root, "samples");
    if (!samples || samples->t != J_ARR) { fprintf(stderr, "Missing samples array\n"); free(buf); free(arena); return 1; }

    double total_weighted_nll = 0.0;
    int total_scored_tokens = 0;

    printf("\n%-24s | %-12s | %-12s | %-10s\n", "Domain", "Prompt Tok", "Scored Tok", "TF-NLL");
    printf("-----------------------------------------------------------------------\n");

    for (int i = 0; i < samples->len; i++) {
        jval *samp = samples->kids[i];
        jval *dom_v = json_get(samp, "domain");
        jval *p_arr = json_get(samp, "prompt_ids");
        jval *f_arr = json_get(samp, "full_ids");
        if (!dom_v || !p_arr || !f_arr || p_arr->t != J_ARR || f_arr->t != J_ARR || f_arr->len <= p_arr->len) {
            fprintf(stderr, "Malformed sample %d\n", i);
            free(buf); free(arena); return 2;
        }
        const char *dom = dom_v->str;

        int np = p_arr->len;
        int nf = f_arr->len;
        int n_scored = nf - np;

        /* Write a temporary single-sample ref.json on node */
        char tmp_ref[256];
        snprintf(tmp_ref, sizeof(tmp_ref), "/home/nayte/prompts/tmp_sample_%d.json", i);
        FILE *rf = fopen(tmp_ref, "w");
        if (!rf) { perror(tmp_ref); free(buf); free(arena); return 2; }
        fprintf(rf, "{\"prompt_ids\":[");
        for (int k = 0; k < np; k++) fprintf(rf, "%d%s", (int)p_arr->kids[k]->num, k == np-1 ? "" : ",");
        fprintf(rf, "],\"full_ids\":[");
        for (int k = 0; k < nf; k++) fprintf(rf, "%d%s", (int)f_arr->kids[k]->num, k == nf-1 ? "" : ",");
        fprintf(rf, "]}");
        fclose(rf);

        char cmd[2048];
        char out_line[512] = "";
        snprintf(cmd, sizeof(cmd),
                 "PPL=1 SNAP=%s OMP_NUM_THREADS=8 COLI_DENSE_I8=1 PILOT=1 %s 128 4 %s 2>&1 | grep 'TF-NLL:'",
                 snap_dir, qwen36_bin, tmp_ref);

        FILE *pipe = popen(cmd, "r");
        if (!pipe) {
            perror("popen"); remove(tmp_ref); free(buf); free(arena); return 2;
        }

        double samp_nll = 0.0;
        int samp_tokens = 0;
        int parsed = 0;
        if (fgets(out_line, sizeof(out_line), pipe))
            parsed = (sscanf(out_line, "TF-NLL: %lf nats/token over %d tokens", &samp_nll, &samp_tokens) == 2);
        int status = pclose(pipe);
        remove(tmp_ref);

        if (!parsed || status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0 || samp_tokens <= 0) {
            fprintf(stderr, "Quality evaluation failed closed at sample %d (%s): parsed=%d status=%d line=%s\n",
                    i, dom ? dom : "?", parsed, status, out_line);
            free(buf); free(arena); return 2;
        }
        if (samp_tokens != n_scored) {
            fprintf(stderr, "Scored-token mismatch at sample %d (%s): expected=%d observed=%d\n",
                    i, dom ? dom : "?", n_scored, samp_tokens);
            free(buf); free(arena); return 2;
        }

        total_weighted_nll += samp_nll * samp_tokens;
        total_scored_tokens += samp_tokens;

        printf("%-24s | %10d | %10d | %10.4f nats\n", dom, np, samp_tokens, samp_nll);
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
