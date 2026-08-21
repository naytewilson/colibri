/* Pure Native C Tool to generate a compact 8-domain held-out evaluation corpus for teacher-forced NLL / PPL evaluation.
 * Covers: reasoning, math, code, technical explanation, instruction following, summarization, knowledge, and structured JSON.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../tok.h"

typedef struct {
    const char *domain;
    const char *prompt;
    const char *completion;
} Sample;

static const Sample SAMPLES[] = {
    {
        "reasoning",
        "Analyze the following logical scenario and determine the causal relationship:",
        " When cache hit rates drop below sixty percent during memory-constrained inference, the processor spends more cycles waiting for DRAM memory transfers than performing arithmetic operations, shifting the primary bottleneck from ALU throughput to memory bandwidth."
    },
    {
        "math",
        "State the closed-form solution and convergence criteria for the linear dynamical system:",
        " The continuous-time system x_dot equals A times x with initial condition x_zero has the unique solution x of t equals matrix exponential of A times t multiplied by x_zero, which is asymptotically stable if and only if all eigenvalues of A have strictly negative real parts."
    },
    {
        "code",
        "Write the core loop of an efficient binary search algorithm in C:",
        "\nint binary_search(const int *arr, int n, int target) {\n    int low = 0, high = n - 1;\n    while (low <= high) {\n        int mid = low + (high - low) / 2;\n        if (arr[mid] == target) return mid;\n        if (arr[mid] < target) low = mid + 1;\n        else high = mid - 1;\n    }\n    return -1;\n}"
    },
    {
        "technical_explanation",
        "Explain the architectural trade-offs between traditional full attention and gated recurrence:",
        " Traditional full multi-head attention computes quadratic pairwise token correlations requiring linear KV-cache memory growth with sequence length, whereas gated recurrence maintains constant O(1) state updates per head using input-dependent decay gates."
    },
    {
        "instruction_following",
        "Provide a concise three-step protocol for server health validation:",
        " First, inspect CPU temperature and thermal throttling states. Second, verify memory residency and swap utilization under load. Third, execute diagnostic network ping sweeps to guarantee packet delivery."
    },
    {
        "summarization",
        "Summarize the key finding of the mixed-precision expert quantization experiment:",
        " Allocating four-bit representation to the top fifteen percent of frequently activated experts while compressing the cold long tail to three bits maximizes arithmetic throughput, accelerates inference by over thirty percent, and preserves model quality."
    },
    {
        "knowledge",
        "Describe the geological process of continental drift and plate tectonics:",
        " The lithosphere is broken into rigid tectonic plates that float upon the ductile asthenosphere, moving relative to each other through seafloor spreading at divergent boundaries, subduction at convergent boundaries, and lateral sliding at transform faults."
    },
    {
        "structured_json",
        "Format the execution status of a task as a structured JSON object:",
        " {\n  \"task_id\": \"exp-402\",\n  \"status\": \"SUCCESS\",\n  \"metrics\": {\n    \"tokens_per_second\": 5.19,\n    \"memory_rss_gb\": 10.77,\n    \"cache_hit_rate\": 0.915\n  }\n}"
    }
};

#define NUM_SAMPLES (sizeof(SAMPLES) / sizeof(SAMPLES[0]))

int main(void) {
    Tok T;
    const char *tokpath = "/home/nayte/models/qwen36_i3_gs64_clean/tokenizer.json";
    tok_load(&T, tokpath);

    const char *out_path = "/home/nayte/prompts/heldout_multidomain.json";
    FILE *f = fopen(out_path, "w");
    if (!f) { perror(out_path); return 1; }

    fprintf(f, "{\n  \"samples\": [\n");

    int total_scored_tokens = 0;

    for (size_t i = 0; i < NUM_SAMPLES; i++) {
        char full_text[4096];
        snprintf(full_text, sizeof(full_text), "%s%s", SAMPLES[i].prompt, SAMPLES[i].completion);

        int p_ids[512], f_ids[2048];
        int np = tok_encode(&T, SAMPLES[i].prompt, strlen(SAMPLES[i].prompt), p_ids, 512);
        int nf = tok_encode(&T, full_text, strlen(full_text), f_ids, 2048);
        int n_scored = nf - np;
        total_scored_tokens += n_scored;

        printf("[Sample %zu/8] Domain: %-22s | Prompt tok: %2d | Full tok: %3d | Scored: %2d\n",
               i + 1, SAMPLES[i].domain, np, nf, n_scored);

        fprintf(f, "    {\n");
        fprintf(f, "      \"domain\": \"%s\",\n", SAMPLES[i].domain);
        fprintf(f, "      \"prompt_ids\": [");
        for (int k = 0; k < np; k++) fprintf(f, "%d%s", p_ids[k], k == np - 1 ? "" : ", ");
        fprintf(f, "],\n");
        fprintf(f, "      \"full_ids\": [");
        for (int k = 0; k < nf; k++) fprintf(f, "%d%s", f_ids[k], k == nf - 1 ? "" : ", ");
        fprintf(f, "]\n");
        fprintf(f, "    }%s\n", (i == NUM_SAMPLES - 1) ? "" : ",");
    }

    fprintf(f, "  ],\n  \"total_samples\": %zu,\n  \"total_scored_tokens\": %d\n}\n",
            NUM_SAMPLES, total_scored_tokens);
    fclose(f);

    printf("=== Created Multi-Domain Held-Out Evaluation Dataset: %s (8 samples, %d scored tokens) ===\n",
           out_path, total_scored_tokens);

    tok_free(&T);
    return 0;
}
