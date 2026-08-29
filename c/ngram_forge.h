/* ngram_forge.h - NGRAM-FORGE N0/N1 experimental surface (Colibri)
 *
 * Exact deterministic n-gram (Engram/PLE) address generation for the
 * Qwen3.8-Flash-Next architecture, plus a read-only sparse-row store for
 * storage tomography. Isolated from ExpertStore/admission by design
 * (NGRAM_FORGE_PLAN.toon: do not force sparse rows into MoE lease
 * semantics).
 *
 * AUTHORITATIVE ORACLE (recovered 2026-08-29, anvil decision #56289):
 *   huggingface/transformers src/transformers/models/qwen4_exp/
 *     modeling_qwen4_exp.py lines 1046-1188 (splitmix64 constants,
 *     _build_layer_multipliers, prime head tables, _shift_right_ignore_eos,
 *     forward hashing) -- official reference at transformers 5.8.0.dev0,
 *     the version recorded in Qwen/Qwen3.8-Flash-Next config.json.
 *   ggml-org/llama.cpp src/models/qwen4exp.cpp llm_graph_input_ple::
 *     set_input() -- independent transcription of the same semantics.
 *   Constants PROVEN from the official checkpoint safetensors buffers
 *     model.language_model.layers.1.ple.ple_embedding.{layer_multipliers,
 *     ngram_heads_vocab_sizes,ngram_heads_offsets} via HTTP byte-ranges;
 *     sha256 of raw buffers pinned in tests/ngram_golden_rows.tsv header.
 *
 * Semantics (must match the oracle bit-exactly):
 *   ngram_size=3, heads_per_ngram=8, 16 heads: h<8 bigram, h>=8 trigram.
 *   For position t:  ctx[j] = x_{t-j} if t-j > e(t) and t-j >= 0 else EOS,
 *   where e(t) = largest index < t with x[index]==EOS (a window-internal EOS
 *   cuts everything at-or-before it; the token's own EOS does not cut its
 *   own context; a missing predecessor reads as EOS).
 *   mixed_n = XOR over j in [0,n) of uint64(ctx[j]) * mult[j]   (n=2,3)
 *   row_h   = mixed_n % head_vocab[h] + head_offset[h]
 *   All products are < 2^63 by multiplier construction, so uint64 `%` and
 *   torch.remainder (signed, positive divisor) agree; invariant is checked
 *   by tests (NGRAM_ASSERT_POSIX_MIXED).
 */
#ifndef NGRAM_FORGE_H
#define NGRAM_FORGE_H

#include <stddef.h>
#include <stdint.h>

#define NGRAM_MAX_NGRAM   4
#define NGRAM_MAX_HEADS   32
#define NGRAM_EOS_TOKEN_ID 248044LL
#define NGRAM_ROW_DIM_BF16 160
#define NGRAM_ROW_BYTES    (NGRAM_ROW_DIM_BF16 * 2) /* bf16 rows */

typedef struct {
    uint64_t mult[NGRAM_MAX_NGRAM];     /* per-position layer multipliers */
    uint64_t head_vocab[NGRAM_MAX_HEADS];
    uint64_t head_offset[NGRAM_MAX_HEADS];
    uint64_t padded_rows;               /* table rows incl. alignment pad */
    int64_t  eos;
    int      ngram_size;                /* max order (3) */
    int      heads_per_ngram;           /* 8 */
    int      n_heads;                   /* (ngram_size-1)*heads_per_ngram */
    uint64_t unigram_vocab;             /* 248320 */
} ngram_params;

/* --- N0: address generation ------------------------------------------- */

/* Derive the full constant set from HF-style generation parameters
 * (seed, vocab, ngram_size, heads_per_ngram, ngram_vocab_size_base,
 * make_ngram_vocab_size_divisible_by, ple_layer_index). Bit-exact with
 * Qwen4ExpTextNGramEmbedding.__init__ + _build_layer_multipliers. */
int ngram_params_derive(uint64_t seed, uint64_t unigram_vocab,
                        int ngram_size, int heads_per_ngram,
                        uint64_t ngram_vocab_base, uint64_t pad_divisor,
                        uint64_t ple_layer_index,
                        int64_t eos, ngram_params *p);

/* Fill p with the constants PROVEN from the Qwen3.8-Flash-Next checkpoint. */
void ngram_params_flash_next(ngram_params *p);

/* One decode-step token: window = [x_t, x_{t-1}, ..., x_{t-(n-1)}] with the
 * EOS window-reset ALREADY applied (pass EOS entries explicitly).
 * Produces p->n_heads row ids, bigram heads first. */
void ngram_rows_window(const ngram_params *p, const int64_t *window,
                       uint64_t *rows /*[n_heads]*/);

/* Streaming prefill: seq of n token ids (absolute positions). rows_out is
 * n * p->n_heads. Applies the exact _shift_right_ignore_eos segment rule. */
void ngram_rows_seq(const ngram_params *p, const int64_t *seq, size_t n,
                    uint64_t *rows_out /*[n][n_heads]*/);

/* Reference oracle implementing _shift_right_ignore_eos verbatim on the
 * concatenated [previous_context || input_ids] window, used only by tests
 * to cross-check ngram_rows_seq / ngram_rows_window. Returns the window for
 * position t (ctx[j] = x_{t-j} after EOS reset). */
void ngram_window_ref(const ngram_params *p, const int64_t *seq, size_t n,
                      size_t t, int64_t *window /*[ngram_size]*/);

/* uint64 splitmix64 step as defined by the oracle (_splitmix64). */
uint64_t ngram_splitmix64(uint64_t v);

/* --- N1: sparse row store (read-only experimental surface) ------------- */

typedef enum {
    NSR_RAM = 0,    /* synthetic in-RAM table, no backing file           */
    NSR_MMAP = 1,   /* mmap file, row copy at consume                    */
    NSR_PREAD = 2,  /* synchronous explicit pread per row                */
    NSR_ASYNC = 3   /* bounded worker pool pread + batch handles         */
} nsr_mode;

typedef struct {
    nsr_mode mode;
    const char *file;        /* NULL => synthetic rows (content = f(row)) */
    uint64_t  rows;          /* table row count (sparse addressing space) */
    uint32_t  row_bytes;     /* bytes per row (NGRAM_ROW_BYTES)           */
    uint32_t  cache_rows;    /* direct-mapped hot-row cache, 0 disables   */
    int       workers;       /* NSR_ASYNC only                            */
    int       queue_depth;   /* max outstanding batches (NSR_ASYNC)       */
} nsr_cfg;

typedef struct nsr_store nsr_store;

/* telemetry (plan measurements) */
typedef struct {
    uint64_t lookups;          /* row requests                        */
    uint64_t unique_lookups;   /* dedup survivor requests             */
    uint64_t dedup_hits;
    uint64_t cache_hits;
    uint64_t ram_hits;         /* NSR_RAM counts as ram_hits          */
    uint64_t issued_reads;     /* physical-path read attempts         */
    uint64_t bytes_read;       /* bytes fetched from backing store    */
    uint64_t bytes_consumed;   /* bytes handed to consumer            */
    uint64_t faults_minor;
    uint64_t faults_major;     /* proxy for physical page reads       */
    uint64_t wait_ns_total;    /* exposed consumer wait               */
    uint64_t consume_ns_total; /* consume incl. wait                  */
    uint64_t addr_ns_total;    /* address generation                  */
    uint64_t batches, batch_hits; /* prefetch hit accounting          */
} nsr_stats;

nsr_store *nsr_open(const nsr_cfg *cfg, char *err, size_t errlen);
void       nsr_close(nsr_store *s);
/* Copy the given rows for one token into a consumer-visible scratch block
 * (gather). Returns 0 on success. wait_ns: time the consumer was blocked. */
int  nsr_consume(nsr_store *s, const uint64_t *rows, int n_rows,
                 void **out_block /*malloc'd n_rows*row_bytes*/,
                 uint64_t *wait_ns);
/* NSR_ASYNC: begin fetching rows ahead of consumption. Returns a batch id
 * (>=0) or -1 if the bounded queue refuses (backpressure, counted). */
int  nsr_prefetch(nsr_store *s, const uint64_t *rows, int n_rows);
/* Consumer-side: wait for batch and gather. Same out contract as consume.
 * fallback != 0 if the batch was never issued/failed (sync re-fetch). */
int  nsr_consume_batch(nsr_store *s, int batch, void **out_block,
                       uint64_t *wait_ns, uint64_t *lookup_ns, int *hit);
void nsr_stats_get(nsr_store *s, nsr_stats *st);
void nsr_stats_reset(nsr_store *s);
/* drop page cache for the backing file (fadvise DONTNEED; also madvise for
 * mmap). RAM mode: no-op. */
void nsr_evict(nsr_store *s);
int  nsr_cached_fraction(nsr_store *s, const uint64_t *rows, int n_rows);

#endif /* NGRAM_FORGE_H */
