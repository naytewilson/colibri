/* Native C tool to generate heldout_eval.json for teacher-forced quantitative quality evaluation.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../tok.h"

int main(void) {
    Tok T;
    const char *tokpath = "/home/nayte/models/qwen36_i3_gs64_clean/tokenizer.json";
    tok_load(&T, tokpath);

    const char *prompt_text = "Explain the architectural differences between traditional multi-head attention and gated recurrent units in modern hybrid transformer architectures:";
    const char *full_text = "Explain the architectural differences between traditional multi-head attention and gated recurrent units in modern hybrid transformer architectures:\n"
        "Traditional Multi-Head Attention (MHA) computes pairwise token interactions across all previous positions using full softmax normalization over dynamic Query and Key matrices, requiring quadratic time complexity and linear key-value cache memory growth with sequence length. In contrast, modern Gated Recurrent Units such as DeltaNet maintain a constant-size hidden recurrent state per head, updating linear projection vectors using input-dependent decay rates and gating mechanisms. This reduces memory bandwidth during autoregressive decoding from O(T) cache growth to strictly O(1) constant working state while preserving long-range associative recall capability.";

    int prompt_ids[1024];
    int np = tok_encode(&T, prompt_text, strlen(prompt_text), prompt_ids, 1024);

    int full_ids[4096];
    int nfull = tok_encode(&T, full_text, strlen(full_text), full_ids, 4096);

    printf("Prompt tokens: %d | Full tokens: %d (Scored evaluation tokens: %d)\n", np, nfull, nfull - np);

    const char *out_path = "/home/nayte/prompts/heldout_eval.json";
    FILE *f = fopen(out_path, "w");
    if (!f) { perror(out_path); return 1; }

    fprintf(f, "{\n");
    fprintf(f, "  \"prompt\": \"%s\",\n", prompt_text);
    fprintf(f, "  \"prompt_ids\": [");
    for (int i = 0; i < np; i++) fprintf(f, "%d%s", prompt_ids[i], i == np - 1 ? "" : ", ");
    fprintf(f, "],\n");
    fprintf(f, "  \"full_ids\": [");
    for (int i = 0; i < nfull; i++) fprintf(f, "%d%s", full_ids[i], i == nfull - 1 ? "" : ", ");
    fprintf(f, "],\n");
    fprintf(f, "  \"text\": \"heldout evaluation text\"\n");
    fprintf(f, "}\n");
    fclose(f);

    printf("Generated held-out evaluation dataset: %s\n", out_path);
    tok_free(&T);
    return 0;
}
