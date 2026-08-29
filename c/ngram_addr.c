/* ngram_addr.c - NGRAM-FORGE N0 exact n-gram address generation.
 * See ngram_forge.h for the authoritative oracle provenance.
 * Native C, no Python. Bit-exact parity with the reference is gated by
 * tests/test_ngram_addr.c.
 */
#include "ngram_forge.h"

#include <stdlib.h>
#include <string.h>

#define M64        0xFFFFFFFFFFFFFFFFull
#define SPLIT_G    0x9E3779B97F4A7C15ull
#define SPLIT_M1   0xBF58476D1CE4E5B9ull
#define SPLIT_M2   0x94D049BB133111EBull
#define PRIME_1    10007ull

uint64_t ngram_splitmix64(uint64_t v)
{
    v = (v + SPLIT_G) & M64;
    v = ((v ^ (v >> 30)) * SPLIT_M1) & M64;
    v = ((v ^ (v >> 27)) * SPLIT_M2) & M64;
    return (v ^ (v >> 31)) & M64;
}

static int is_prime64(uint64_t v)
{
    if (v < 2) return 0;
    if (v % 2 == 0) return v == 2;
    for (uint64_t d = 3; d * d <= v; d += 2)
        if (v % d == 0) return 0;
    return 1;
}

/* _find_nth_prime_after(start, count): count-th prime strictly after start */
static uint64_t nth_prime_after(uint64_t start, uint64_t count)
{
    uint64_t v = start;
    for (uint64_t i = 0; i < count; i++) {
        do { v++; } while (!is_prime64(v));
    }
    return v;
}

int ngram_params_derive(uint64_t seed, uint64_t unigram_vocab,
                        int ngram_size, int heads_per_ngram,
                        uint64_t ngram_vocab_base, uint64_t pad_divisor,
                        uint64_t ple_layer_index,
                        int64_t eos, ngram_params *p)
{
    if (!p || ngram_size < 2 || ngram_size > NGRAM_MAX_NGRAM) return -1;
    int heads = (ngram_size - 1) * heads_per_ngram;
    if (heads <= 0 || heads > NGRAM_MAX_HEADS) return -1;
    memset(p, 0, sizeof(*p));
    p->ngram_size = ngram_size;
    p->heads_per_ngram = heads_per_ngram;
    p->n_heads = heads;
    p->unigram_vocab = unigram_vocab;
    p->eos = eos;

    /* _build_layer_multipliers: signed-long bounds, odd multipliers that
     * keep every (id * mult) product below 2^63 so XOR never sets bit 63. */
    uint64_t max_long = (1ull << 63) - 1;
    uint64_t mult_max = max_long / (unigram_vocab > 0 ? unigram_vocab : 1);
    uint64_t half_bound = mult_max / 2;
    if (half_bound < 1) half_bound = 1;
    uint64_t base_seed = (seed + PRIME_1 * ple_layer_index) & M64;
    for (int i = 0; i < ngram_size; i++) {
        uint64_t v = (base_seed + SPLIT_G * (uint64_t)(i + 1)) & M64;
        p->mult[i] = 2 * (ngram_splitmix64(v) % half_bound) + 1;
    }

    /* head tables: global head stream across ple layers */
    uint64_t total = 0;
    for (int h = 0; h < heads; h++) {
        uint64_t global_h = ple_layer_index * (uint64_t)heads + (uint64_t)h;
        uint64_t size = nth_prime_after(ngram_vocab_base - 1, global_h + 1);
        p->head_vocab[h] = size;
        p->head_offset[h] = total;
        total += size;
    }
    p->padded_rows = ((total + pad_divisor - 1) / pad_divisor) * pad_divisor;
    return 0;
}

/* Constants PROVEN from the Qwen3.8-Flash-Next checkpoint safetensors
 * buffers (raw byte-range sha256 recorded in tests/ngram_golden_rows.tsv):
 *   layer_multipliers          I64[3]  model-00005-of-00131 @2400..2424
 *   ngram_heads_offsets        I64[16] model-00037-of-00131 @2776..2904
 *   ngram_heads_vocab_sizes    I64[16] model-00037-of-00131 @2904..3032
 *   padded table = 128 shards x [2500012,160] = [320001536,160] rows
 */
void ngram_params_flash_next(ngram_params *p)
{
    static const uint64_t mv[3] = {
        23703573157769ull, 20109073645365ull, 8052911324071ull };
    static const uint64_t hv[16] = {
        20000003ull, 20000023ull, 20000033ull, 20000047ull,
        20000059ull, 20000063ull, 20000069ull, 20000077ull,
        20000081ull, 20000093ull, 20000107ull, 20000147ull,
        20000153ull, 20000159ull, 20000161ull, 20000171ull };
    memset(p, 0, sizeof(*p));
    p->ngram_size = 3;
    p->heads_per_ngram = 8;
    p->n_heads = 16;
    p->unigram_vocab = 248320;
    p->eos = NGRAM_EOS_TOKEN_ID;
    memcpy(p->mult, mv, sizeof(mv));
    uint64_t total = 0;
    for (int h = 0; h < 16; h++) {
        p->head_vocab[h] = hv[h];
        p->head_offset[h] = total;
        total += hv[h];
    }
    p->padded_rows = ((total + 127) / 128) * 128; /* 320001536 */
}

void ngram_rows_window(const ngram_params *p, const int64_t *window,
                       uint64_t *rows)
{
    const int per = p->heads_per_ngram;
    for (int n = 2; n <= p->ngram_size; n++) {
        uint64_t mixed = window[0] * p->mult[0];
        for (int j = 1; j < n; j++)
            mixed ^= (uint64_t)window[j] * p->mult[j];
        int base = (n - 2) * per;
        for (int g = 0; g < per; g++) {
            int h = base + g;
            rows[h] = mixed % p->head_vocab[h] + p->head_offset[h];
        }
    }
}

void ngram_window_ref(const ngram_params *p, const int64_t *seq, size_t n,
                      size_t t, int64_t *window)
{
    /* e(t): last EOS strictly before t (segment rule of the oracle) */
    size_t e = (size_t)-1;
    int has_e = 0;
    for (size_t i = 0; i < t; i++) {
        if (seq[i] == p->eos) { e = i; has_e = 1; }
    }
    for (int j = 0; j < p->ngram_size; j++) {
        if ((int64_t)t - j < 0) { window[j] = p->eos; continue; }
        size_t src = t - (size_t)j;
        if (has_e && src <= e) window[j] = p->eos;
        else window[j] = seq[src];
    }
}

void ngram_rows_seq(const ngram_params *p, const int64_t *seq, size_t n,
                    uint64_t *rows_out)
{
    int64_t w[NGRAM_MAX_NGRAM];
    size_t e = (size_t)-1; /* last EOS strictly before t */
    for (size_t t = 0; t < n; t++) {
        for (int j = 0; j < p->ngram_size; j++) {
            if ((int64_t)t - j < 0 || (e != (size_t)-1 && t - (size_t)j <= e))
                w[j] = p->eos;
            else
                w[j] = seq[t - (size_t)j];
        }
        ngram_rows_window(p, w, rows_out + t * (size_t)p->n_heads);
        if (seq[t] == p->eos) e = t;
    }
}
