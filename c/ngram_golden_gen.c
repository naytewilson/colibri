/* ngram_golden_gen.c - emit tests/ngram_golden_rows.tsv from the
 * production ngram_addr implementation on fixed short sequences.
 * This pins addressing against later regressions; it is NOT the parity
 * evidence (parity evidence is the triple transcription test).
 */
#include "ngram_forge.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t fx_next(uint64_t *st)
{
    *st = (*st * 6364136223846793005ull + 1442695040888963407ull);
    return (uint64_t)((*st >> 33) % 300000);
}

int main(void)
{
    ngram_params p;
    ngram_params_flash_next(&p);
    printf("# NGRAM-FORGE golden row-ID fixture (regression pin)\n");
    printf("# generator: c/ngram_golden_gen.c @ ngram-forge/n0-n1\n");
    printf("# oracle constants sha256: multipliers=37349930c3c77bf4927b2afb6b50f30c404f42f00f5790bda96e7d9589c26477\n");
    printf("#   head_vocab_sizes=81879b658d9974287ceecc77463e5fdd183cb4cd66692b061ea6af24ef414d39\n");
    printf("#   head_offsets=af1566f7a3be2f4a9ee8e2384aa161b8d45cfcbd92376827662a028679b7d8eb\n");
    printf("# source: Qwen/Qwen3.8-Flash-Next safetensors byte-ranges 2026-08-29;\n");
    printf("#   formula: transformers qwen4_exp/modeling_qwen4_exp.py:1046-1188\n");
    printf("# each line: seq_id len tok0..tok4 \\t row0..row15 (rows at FINAL position)\n");

    /* boundary cases first (fixed) */
    static const int64_t b[][5] = {
        { 1, -1, -1, -1, -1 },
        { 5, 7, -1, -1, -1 },
        { 0, 0, 0, -1, -1 },
        { NGRAM_EOS_TOKEN_ID, 42, -1, -1, -1 },
        { 42, NGRAM_EOS_TOKEN_ID, 43, -1, -1 },
        { NGRAM_EOS_TOKEN_ID, NGRAM_EOS_TOKEN_ID, 44, 45, -1 },
        { 248319, 248318, 248317, 248316, 248315 },
        { 248043, NGRAM_EOS_TOKEN_ID, 248045, 1, 2 },
    };
    int id = 0;
    for (size_t i = 0; i < sizeof(b) / sizeof(b[0]); i++) {
        int64_t seq[5];
        size_t n = 0;
        while (n < 5 && b[i][n] != -1) { seq[n] = b[i][n]; n++; }
        if (!n) continue;
        uint64_t rows[5 * 16];
        ngram_rows_seq(&p, seq, n, rows);
        uint64_t *last = rows + (n - 1) * 16;
        printf("%d %zu", id++, n);
        for (size_t j = 0; j < 5; j++) printf(" %lld", j < n ? (long long)seq[j] : -1LL);
        printf("\t");
        for (int h = 0; h < 16; h++) printf(" %" PRIu64, last[h]);
        printf("\n");
    }
    /* 512 short deterministic fuzz cases */
    uint64_t st = 0xC0FFEE1234ull;
    for (int c = 0; c < 512; c++) {
        int64_t seq[5];
        size_t n = 1 + (size_t)(st % 5u);
        for (size_t j = 0; j < n; j++) {
            int64_t t = (int64_t)fx_next(&st);
            seq[j] = (t % 17 == 0) ? p.eos : (t % (int64_t)p.unigram_vocab);
        }
        uint64_t rows[5 * 16];
        ngram_rows_seq(&p, seq, n, rows);
        uint64_t *last = rows + (n - 1) * 16;
        printf("%d %zu", id++, n);
        for (size_t j = 0; j < 5; j++) printf(" %lld", j < n ? (long long)seq[j] : -1LL);
        printf("\t");
        for (int h = 0; h < 16; h++) printf(" %" PRIu64, last[h]);
        printf("\n");
    }
    fprintf(stderr, "golden cases: %d\n", id);
    return 0;
}
