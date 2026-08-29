/* test_ngram_addr.c - NGRAM-FORGE N0 exit gate.
 *
 * Check 1: derived constants == checkpoint-PROVEN constants (bit-exact).
 * Check 2: production address generator vs verbatim transcription of the
 *          official HF torch path (_shift_right_ignore_eos + torch-style
 *          signed remainder) vs independent llama.cpp set_input()
 *          transcription (unsigned mod + window cut) over boundary
 *          fixtures + deterministic fuzz. Triple parity required.
 * Check 3: positivity invariant (mixed < 2^63) => signed/unsigned mod
 *          semantics provably coincide.
 * Check 4: committed golden row-ID fixture (regression pin).
 *
 * Python-deny note: the runtime torch reference itself is NOT executed on
 * this node. The oracle legs are verbatim transcriptions of
 * modeling_qwen4_exp.py (commit-time content fetched 2026-08-29) and
 * llama.cpp qwen4exp.cpp; constants are pinned from checkpoint artifact
 * bytes. Absolute torch-runtime parity is recorded separately in the
 * receipt (see ngram_forge.h provenance).
 */
#include "../ngram_forge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); failures++; } } while (0)

/* ---- oracle leg 1: verbatim transcription of the HF torch path ------- */

typedef int64_t i64;
#define NGRAM_ORDER_MAX 3

static i64 hf_shift_right_ignore_eos(const i64 *tok, size_t n, size_t t,
                                     int shift, i64 eos)
{
    /* token_ids[t - shift] if same segment else eos, per
     * modeling_qwen4_exp.py::_shift_right_ignore_eos (window over `tok`). */
    if (shift == 0) return tok[t];
    /* previous_eos = last index < t with tok[idx]==eos (cummax strictly pre) */
    long long prev_eos = -1;
    for (size_t i = 0; i < t; i++)
        if (tok[i] == eos) prev_eos = (long long)i;
    long long segment_start = prev_eos + 1;
    long long position_in_segment = (long long)t - segment_start;
    long long source = (long long)t - shift;
    int valid = (position_in_segment >= shift) && (source >= 0);
    if (!valid) return eos;
    return tok[(size_t)source];
}

static i64 torch_remainder(i64 a, uint64_t b_)
{
    /* torch.remainder: sign follows divisor (b>0) => [0,b) */
    i64 b = (i64)b_;
    i64 r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}

static void rows_ref_hf(const ngram_params *p, const i64 *seq, size_t n,
                        size_t t, uint64_t *rows)
{
    i64 sh[NGRAM_ORDER_MAX];
    for (int s = 0; s < p->ngram_size; s++)
        sh[s] = hf_shift_right_ignore_eos(seq, n, t, s, p->eos);
    for (int order = 2; order <= p->ngram_size; order++) {
        i64 mixed = (i64)((uint64_t)sh[0] * p->mult[0]);
        for (int pos = 1; pos < order; pos++)
            mixed ^= (i64)((uint64_t)sh[pos] * p->mult[pos]);
        int start = (order - 2) * p->heads_per_ngram;
        for (int g = 0; g < p->heads_per_ngram; g++) {
            int h = start + g;
            rows[h] = (uint64_t)torch_remainder(mixed, p->head_vocab[h])
                    + p->head_offset[h];
        }
    }
}

/* ---- oracle leg 2: independent llama.cpp set_input() transcription ---- */

static void rows_ref_llama(const ngram_params *p, const i64 *seq, size_t n,
                           size_t t, uint64_t *rows)
{
    /* prev predecessors with cut semantics; missing cell reads as EOS */
    i64 ctx[NGRAM_ORDER_MAX];
    ctx[0] = seq[t];
    int cut = 0;
    for (int s = 1; s < p->ngram_size; s++) {
        long long idx = (long long)t - s;
        i64 tok = (idx < 0) ? -1 : seq[idx];
        if (cut || tok < 0 || tok == p->eos) { ctx[s] = p->eos; cut = 1; }
        else ctx[s] = tok;
        if (tok == p->eos) cut = 1; /* this predecessor is an EOS itself */
    }
    for (int order = 2; order <= p->ngram_size; order++) {
        uint64_t mixed = (uint64_t)ctx[0] * p->mult[0];
        for (int j = 1; j < order; j++)
            mixed ^= (uint64_t)ctx[j] * p->mult[j];
        int base = (order - 2) * p->heads_per_ngram;
        for (int g = 0; g < p->heads_per_ngram; g++) {
            int h = base + g;
            rows[h] = mixed % p->head_vocab[h] + p->head_offset[h];
        }
    }
}

/* ---- check 1: derived vs checkpoint-proven constants ------------------ */

static void check_constants(void)
{
    ngram_params d, g;
    ngram_params_flash_next(&g);
    int rc = ngram_params_derive(1234, 248320, 3, 8, 20000000, 128, 0,
                                NGRAM_EOS_TOKEN_ID, &d);
    CHECK(rc == 0, "derive rc=%d", rc);
    for (int i = 0; i < 3; i++)
        CHECK(d.mult[i] == g.mult[i], "mult[%d] %" PRIu64 " != %" PRIu64,
              i, d.mult[i], g.mult[i]);
    for (int h = 0; h < 16; h++) {
        CHECK(d.head_vocab[h] == g.head_vocab[h], "vocab[%d] %" PRIu64,
              h, d.head_vocab[h]);
        CHECK(d.head_offset[h] == g.head_offset[h], "offset[%d] %" PRIu64,
              h, d.head_offset[h]);
    }
    CHECK(d.padded_rows == 320001536ull, "padded_rows %" PRIu64,
          d.padded_rows);
}

/* ---- checks 2+3: triple parity + positivity invariant ----------------- */

static const int64_t boundary_seqs[][12] = {
    { 1, -1, -1, -1, -1, -1, -1, -1 },                      /* first token */
    { 5, 7, -1, -1, -1, -1, -1, -1 },                       /* two tokens  */
    { 5, 7, 9, -1, -1, -1, -1, -1 },                        /* three       */
    { NGRAM_EOS_TOKEN_ID, 42, -1, -1, -1, -1, -1, -1 },     /* post-EOS    */
    { 42, NGRAM_EOS_TOKEN_ID, 43, -1, -1, -1, -1, -1 },     /* mid EOS     */
    { NGRAM_EOS_TOKEN_ID, NGRAM_EOS_TOKEN_ID, 44, -1, -1, -1, -1, -1 }, /* EOS run */
    { 45, 46, NGRAM_EOS_TOKEN_ID, -1, -1, -1, -1, -1 },     /* EOS tail    */
    { 0, 0, 0, -1, -1, -1, -1, -1 },                        /* zero ids    */
    { 248319, 248318, 248317, -1, -1, -1, -1, -1 },         /* max ids     */
    { 248319, 248319, 248319, 248319, -1, -1, -1, -1 },     /* saturated   */
};

static uint64_t state_mix(uint64_t x) { return ngram_splitmix64(x); }

static void parity_on(const ngram_params *p, const int64_t *seq, size_t n,
                      uint64_t *bad)
{
    uint64_t b[NGRAM_MAX_HEADS], c[NGRAM_MAX_HEADS];
    static uint64_t prod[4096 * NGRAM_MAX_HEADS]; /* full-stream pass */
    if (n > 4096 / (size_t)p->n_heads) return;
    ngram_rows_seq(p, seq, n, prod);
    for (size_t t = 0; t < n; t++) {
        rows_ref_hf(p, seq, n, t, b);
        rows_ref_llama(p, seq, n, t, c);
        uint64_t *pa = prod + t * (size_t)p->n_heads;
        for (int h = 0; h < p->n_heads; h++) {
            if (pa[h] != b[h] || pa[h] != c[h]) {
                if (bad) (*bad)++;
                CHECK(0, "t=%zu h=%d prod=%" PRIu64 " hf=%" PRIu64 " llama=%" PRIu64
                      " n=%zu seq=%lld,%lld,%lld,%lld",
                      t, h, pa[h], b[h], c[h], n,
                      n > 0 ? (long long)seq[0] : -99LL,
                      n > 1 ? (long long)seq[1] : -99LL,
                      n > 2 ? (long long)seq[2] : -99LL,
                      n > 3 ? (long long)seq[3] : -99LL);
                return;
            }
            CHECK(pa[h] < p->padded_rows, "row %zu out of table", (size_t)pa[h]);
        }
    }
}

static void check_parity(void)
{
    ngram_params p;
    ngram_params_flash_next(&p);

    for (size_t i = 0; i < sizeof(boundary_seqs) / sizeof(boundary_seqs[0]); i++) {
        size_t n = 0;
        int64_t seq[12];
        while (boundary_seqs[i][n] != -1 && n < 12) { seq[n] = boundary_seqs[i][n]; n++; }
        parity_on(&p, seq, n, NULL);
    }

    /* deterministic fuzz: 4096 sequences over mixed hot/cold vocab + EOS */
    uint64_t s = 0x12345678ull;
    int64_t seq[96];
    for (int iter = 0; iter < 4096; iter++) {
        size_t n = 1 + (size_t)(state_mix(s += 7) % 90);
        for (size_t t = 0; t < n; t++) {
            uint64_t r = state_mix(s + t * 13 + iter) ;
            uint64_t bucket = r % 100;
            if (bucket < 6) seq[t] = p.eos;
            else if (bucket < 40) seq[t] = (int64_t)(r % 512);          /* hot   */
            else seq[t] = (int64_t)(r % p.unigram_vocab);               /* wide  */
        }
        parity_on(&p, seq, n, NULL);
    }

    /* decode-window equivalence: ngram_rows_window on the ref window must
     * equal the streaming path for every position of every boundary seq */
    for (size_t i = 0; i < sizeof(boundary_seqs) / sizeof(boundary_seqs[0]); i++) {
        size_t n = 0;
        int64_t seq[12];
        while (boundary_seqs[i][n] != -1 && n < 12) { seq[n] = boundary_seqs[i][n]; n++; }
        uint64_t prod[12 * NGRAM_MAX_HEADS], wrows[NGRAM_MAX_HEADS];
        ngram_rows_seq(&p, seq, n, prod);
        for (size_t t = 0; t < n; t++) {
            int64_t w[NGRAM_MAX_NGRAM];
            ngram_window_ref(&p, seq, n, t, w);
            ngram_rows_window(&p, w, wrows);
            for (int h = 0; h < p.n_heads; h++)
                CHECK(wrows[h] == prod[t * p.n_heads + h],
                      "window vs seq t=%zu h=%d", t, h);
        }
    }
}

static void check_positivity_invariant(void)
{
    ngram_params p;
    ngram_params_flash_next(&p);
    uint64_t max_id = p.unigram_vocab - 1;
    for (int j = 0; j < p.ngram_size; j++) {
        /* product must stay < 2^63: guarantees XOR never sets bit 63 */
        CHECK(max_id * p.mult[j] < (1ull << 63),
              "mult[%d] can overflow sign bit", j);
    }
    uint64_t s = 0xABCDEF01ull;
    for (int i = 0; i < 200000; i++) {
        uint64_t a = state_mix(s += 3) % p.unigram_vocab;
        uint64_t b = state_mix(s + 1) % p.unigram_vocab;
        uint64_t c = state_mix(s + 2) % p.unigram_vocab;
        uint64_t m2 = a * p.mult[0] ^ b * p.mult[1];
        uint64_t m3 = m2 ^ c * p.mult[2];
        CHECK(!(m2 >> 63) && !(m3 >> 63), "sign bit set in mixed");
    }
}

/* ---- check 4: golden row fixture -------------------------------------- */

#ifndef NGRAM_GOLDEN_PATH
#define NGRAM_GOLDEN_PATH "tests/ngram_golden_rows.tsv"
#endif

static void check_golden(void)
{
    FILE *f = fopen(NGRAM_GOLDEN_PATH, "r");
    if (!f) {
        fprintf(stderr, "FAIL golden fixture missing: %s\n", NGRAM_GOLDEN_PATH);
        failures++;
        return;
    }
    ngram_params p;
    ngram_params_flash_next(&p);
    char line[512];
    int cases = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        /* format: seq_id len tok0..tok4 \t row0..row15 (rows at final pos) */
        unsigned long len;
        long long toks_ll[5];
        unsigned long long rows_ll[16];
        char *tab = strchr(line, '\t');
        CHECK(tab != NULL, "golden line %d malformed", cases);
        if (!tab) continue;
        *tab = 0;
        int id_read = -1;
        int nt = sscanf(line, "%d %lu %lld %lld %lld %lld %lld",
                        &id_read, &len, toks_ll, toks_ll + 1, toks_ll + 2,
                        toks_ll + 3, toks_ll + 4);
        (void)nt; (void)id_read;
        int nr = sscanf(tab + 1,
                        "%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu",
                        rows_ll, rows_ll+1, rows_ll+2, rows_ll+3, rows_ll+4,
                        rows_ll+5, rows_ll+6, rows_ll+7, rows_ll+8, rows_ll+9,
                        rows_ll+10, rows_ll+11, rows_ll+12, rows_ll+13,
                        rows_ll+14, rows_ll+15);
        CHECK(nr == 16, "golden row count %d", nr);
        cases++;
        if (nr != 16 || len == 0 || len > 5) continue;
        int64_t seq[5];
        uint64_t want[16];
        for (int j = 0; j < 5; j++) seq[j] = (int64_t)toks_ll[j];
        for (int h = 0; h < 16; h++) want[h] = (uint64_t)rows_ll[h];
        uint64_t got[5 * NGRAM_MAX_HEADS];
        ngram_rows_seq(&p, seq, (size_t)len, got);
        /* fixture stores the LAST position of the sequence */
        uint64_t *g = got + (len - 1) * 16;
        for (int h = 0; h < 16; h++)
            CHECK(g[h] == want[h], "golden seq %d h %d", cases, h);
    }
    fclose(f);
    fprintf(stderr, "golden fixture cases checked: %d\n", cases);
}

int main(void)
{
    check_constants();
    check_parity();
    check_positivity_invariant();
    check_golden();
    if (failures) {
        fprintf(stderr, "NGRAM_ADDR_PARITY_FAIL %d\n", failures);
        return 1;
    }
    printf("NGRAM_ADDR_PARITY_PASS\n");
    return 0;
}
