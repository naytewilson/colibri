# FORGE F1 — Mechanism Census (Phase 0, task 3)

Date: 2026-08-23 · Baseline: `f04359a` · Method: direct source inspection of
`c/qwen36.c` (4252 lines) and `c/olmoe.c` (1530 lines) in worktree
`~/ANVIL-worktrees/colibri-forge-f1`. Line numbers refer to that revision.
This census drives Phases 1–7; no abstraction boundary was chosen before it existed.

---

## 1. qwen36.c expert-cache/admission machinery

### 1.1 Core state structures

| Symbol | Lines | Purpose |
|---|---|---|
| `Slot` | 586–599 | One resident expert: `eid`, `pinned`, `is_int4/is_int3`, `allocated_weight_bytes`, weight pointers (`w4`,`w3`,`g/u/d`,`g4/u4/d4`,`g3/u3/d3`), scale pointers (`gs/us/ds`), LRU `used` clock |
| `LCache` | 600–604 | Per-layer cache: `slots[]`, `n`, `cap`, **`loading[eid]` int16 in-flight registry** (-1 = none; set at reservation under `g_pilot_mx`, cleared at publish). Makes duplicate-load coalescing visible to demand AND pilot paths |
| `Model` cache fields | 613–632 | `cache[n_layers]`, `active_of[]` (original→container layer idx), `clock/hits/miss`, `freq[n_layers*n_experts]`, `router_mass[]`, `momentum_logits` (EMA), `pilot_smooth`, `pilot_conf_limit`, `is_pinned[]`, `is_queued[]`, `seen[]`, `resident_mode/collecting`, `hot_n/warmup_tokens/hot_pinned/freq_token_count` |
| `AcqKind/AcqRes` | 1993–1997 | `{ACQ_HIT, ACQ_LOAD}`; result of acquire phase: `{kind, Slot *s, victim_eid, mode}` |
| `ACQ_MODE_DEMAND/PRELOAD` | 1996–1997 | Demand counts hit/miss + emits DEMAND trace; PRELOAD is silent (W9) |
| `ExpertLoadResult` | 768–774 | Per-load outcome: `ms, bytes, fmt(3/4/8/0), wbytes, sbytes` — delivered to the triggering caller only, never via shared globals |

### 1.2 Mechanism groups

**G1 Slot table / per-layer LRU / victim scan**
- Hit scan + LRU touch inside `expert_acquire` (2200–2210).
- Victim scan skipping pinned and in-flight (`eid<0`) slots (2276–2287); fallback oldest non-in-flight; final spin-wait when EVERY slot is in flight (2288–2309) — comment documents this was taken verbatim from olmoe.c after a silent-corruption bug (last-resort steal raced an unlocked pread).
- Capacity init: `model_init` sets `cache[i].cap = cap`, allocates slots + `loading` array all -1 (1680–1686). Cap from CLI; dynamic budget from `COLI_EXPERT_CACHE_GB` (3936–3945).
- Hot pinning: `pin_hot_experts` (2401–2453) top-N per layer by freq (`HOT>=100` → dynamic threshold mode), per-layer pin budget `cap-8` (min 4), enqueues missing hot experts to pilot.
- Prompt residency pins: `apply_resident` (2460–2488), modes via `COLIBRI_RESIDENT`.

**G2 Pinned arenas** — `Slot.pinned` flag + `Model.is_pinned[l*E+e]`; eviction skips pinned unless only-pinned fallback (which may displace a pinned but never an in-flight slot).

**G3 Pilot prefetch worker**
- Globals: `g_pilot_mx` (635), `pilot_q[4096]` ring (636), `pilot_r/pilot_w` atomics (637), `pilot_m` (638), `g_pilot` (639), `g_wide` (640).
- `ensure_pilot_worker_started` (648–663): spawns `COLI_PILOT_W` workers (1..8, default 1).
- `pilot_worker` (3220–3236): CAS-claim dequeue (multi-worker safe).
- `pilot_realload` (3180–3218): re-checks queued/resident/coalesce (`loading>=0` → skip + `g_pilot_coalesce_skips++`), reserves slot (same victim scan), loads with `load_expert_merged(..., is_pilot=1)`, publishes.
- `pilot_prefetch` (3238–3314): next-layer router probe — rmsnorm(post_ln)+gate matmul, optional EMA blend (`SMOOTH`/`momentum_logits`), softmax candidates up to `topk*g_wide` capped 128, cumulative-prob cutoff `CONF_LIMIT` (min candidates = topk), deterministic eid-sorted enqueue carrying original confidence rank, `is_queued` gating, v4 `C PC` intent events emitted unconditionally BEFORE residency/queue decisions under the same lock.

**G4 In-flight reservation / duplicate coalescing**
- Registry: `LCache.loading[eid]`.
- Reserve at 2316–2317 (`s->eid=-1; loading[eid]=slot_idx`) under lock; publish at 2337–2338 clears.
- Demand coalesce wait loop 2216–2248: releases lock, spins 1 ms, rescans; coalesced acquisition counts exactly ONCE as HIT (miss counted only post-coalesce-decision at 2271–2272).
- Counters: `g_demand_coalesce_waits` (747), `g_pilot_coalesce_skips` (748).

**G5 Async acquire/finish split (decode)**
- `expert_acquire` (2187–2320) locked phase; `expert_finish` (2322–2345) does the UNLOCKED load then publishes under lock; wrapper `expert_get` (2347–2351).
- Decode batch path in `moe()` (2759–2798): gated `S==1 && COLI_EXPERT_ASYNC && K>=2 && K<=8 && cap>=2*K`; acquires all K up front in kk order (identical intent stream), finishes misses concurrently with `omp parallel for` W=min(nload, `COLI_ASYNC_W` default 4).

**G6 Prefill batch acquisition (COLI_BATCH_ACQ)**
- `moe()` 2607–2685: gated `S>1 && COLI_BATCH_ACQ=='1' && S<=512 && K<=8`.
- Routing pre-pass on scratch copy (bit-identical selection); unique-missing experts reserved in first-touch order via `ACQ_MODE_PRELOAD` (silent: zero DEMAND emission/counters — W9 canonical-pass-owns-events repair); concurrent finish W=min(nl,8).
- Timers: `g_pb_wall_ms` (window), `g_pb_io_us` atomic sum of preload load durations (2044).
- Hard-coded geometry: `int(*bidx)[8]`, `S<=512`, `K<=8`.

**G7 FUSED container read (COLI_FUSED_LOAD)**
- Inside `load_expert_merged` (1857–1922): enabled when env=1 AND `ts->fd==tw->fd && ts->off+ts->nbytes==tw->off` (zero-gap [qs][merged_weight] interleave verified per-tensor). One `st_pread_full` covering both; memcpy-split after transfer; identical bytes. Disabled automatically for unpack-int8 path.

**G8 Weights/scales admission decomposition + byte accounting**
- Same function: `_io_w` weights segment vs `_io_s` scales segment timers (1921–1926); `total_loaded_bytes = tw->nbytes + ts->nbytes`; format-aware admitted-bytes counters (int3/int4) updated under `g_io_stats_mx` (1928–1951).
- Format sizing: `expert_fmt` detects fmt by tensor nbytes (fmt5 INT3-g64 = `(ng+ng+nd)/64*24`; fmt4 = `/2`; else INT8) (1722–1736); `slot_ensure_format` (1758–1830) allocates/reuses buffers across format changes (realloc paths); `scale_count_gu/d` for gs64 grouped scales (1719–1720).
- Tensor addressing: `st_find(&m->S, "model.layers.%d.mlp.experts.%d.merged_weight"|"...qs")` via `active_of[layer]` remap; reads through st.h (`st_read_raw`, `st_read_f32`, `st_pread_full`), `posix_fadvise(DONTNEED)` when drop flag on.

**G9 Speculative shadow ring (default OFF, out of extraction scope)**
- `spec_ring/spec_loader/spec_issue/spec_serve` (2044–2185) + `spec_predict` (2357–2399): COLI_SPEC_DEPTH/BUDGET/DEBUG/I3ONLY, LEADFILE harvest. Stays engine-local behind its env knobs; noted as future scheduler-policy consumer.

**G10 Trace/telemetry streams**
- v3 oracle stream: `trace_emit` (823–831), file `COLI_MOE_TRACE` (3399); rows `seq class event tok layer eid fmt bytes adm_ms victim_eid slot`, class DEMAND|PILOT, event HIT|EVICT|INSERT, seq assigned under `g_pilot_mx` = total mutation order.
- v4 request stream: `req_emit` (817–821), file `COLI_TRACE_REQ` (3414); intents E/B/R/C only (policy-independent replay input).
- Lead-time harvest: `COLI_LEADFILE` (2024, 2333–2334).

### 1.3 Counters inventory (Phase 5 must cover)

requests/hits/misses: `m->hits/m->miss`, `g_acq_hits/g_acq_miss`;
by-format: `g_cache_hit_int3/int4`, `g_cache_miss_int3/int4`, `g_routed_int3/int4_count`,
`g_admitted_bytes_int3/int4`;
prefetch: `g_pilot_loads`, `g_pilot_bytes`, prefetch-hit = demand hit on pilot-published slot (not separately bucketed today);
bytes: `g_demand_bytes`, `admitted bytes` above;
admission latency: `g_demand_expert_admission_ms`, `g_pilot_expert_admission_ms` with weight/scale split `g_{demand,pilot}_{weight,scale}_ms`;
expert_get internals: `g_eg_lock_wait_ms`, `g_eg_lookup_ms`, `g_eg_victim_ms`;
coalesce: waits/skips (above); batch: `g_pb_io_us`, `g_pb_wall_ms`;
conservation (W9): acquisitions/token expected==actual check via window baselines;
decode-window isolation: `g_win_*` baselines armed post-prefill, folded at prefill boundaries and in `tm_report` (874–887).

### 1.4 Env knobs (qwen36.c, cache-relevant)

| Env | Line(s) | Default | Meaning |
|---|---|---|---|
| `PILOT` | 3929 | 0 | pilot prefetch enable |
| `COLI_PILOT_W` | 652 | 1 (1..8) | pilot worker count |
| `WIDE` | 3930 | 1 | candidate multiplier (topk×WIDE, ≤128) |
| `HOT` | 1699/3934 | 0 | hot-pin count; ≥100 = dynamic %-threshold |
| `WARMUP` | 1700 | 5 | warmup tokens before pinning |
| `SMOOTH` | 1703/3957 | 0.3 | routing EMA coefficient |
| `CONF_LIMIT` | 1712/3958 | 0.92 | cumulative prob cutoff for candidates |
| `COLIBRI_RESIDENT` | 1708 | 0 | prompt-resident pin modes |
| `COLI_FUSED_LOAD` | 1863 | OFF | fused [qs][weights] pread |
| `COLI_BATCH_ACQ` | 2615 | OFF | prefill batch acquisition (S≤512, K≤8) |
| `COLI_EXPERT_ASYNC` | 2002 | OFF | decode async acquisition (K≥2, K≤8, cap≥2K) |
| `COLI_ASYNC_W` | 2003 | 4 | async finish threads |
| `COLI_EXPERT_PAGECACHE` | 2008 | ON | fadvise DONTNEED after expert reads (`expert_drop_flag`) |
| `COLI_EXPERT_PARALLEL` | 856 | 0 | expert-parallel GEMV topology (compute-side) |
| `COLI_EXPERT_CACHE_GB` | 3936 | – | pool budget → cap/layer |
| `COLI_UNPACK_INT8` | 1754 | OFF | unpack int4→int8 in slot |
| `COLI_TIMERS` | 839 | off | M-prof timers |
| `COLI_MOE_TRACE` | 3399 | – | v3 TSV oracle stream path |
| `COLI_TRACE_REQ` | 3414 | – | v4 request stream path |
| `COLI_LEADFILE` | 2024 | – | lead-time harvest file |
| `COLI_SPEC_DEPTH/BUDGET/DEBUG/I3ONLY` | 2038–2126 | OFF | speculation ring |

Promoted production config on node-02: `EXPERT_PARALLEL=1 PILOT=1 DENSE_I8=1 OMP=8`
(+ campaign arms `FUSED_LOAD=1 BATCH_ACQ=1` measured −19.5% total wall, default-OFF).

### 1.5 Qwen-specific items that must NOT enter the generic contract

- `s->eid=-1` in-flight sentinel (contract uses explicit reservation identity instead).
- `bidx[8]` width, `K<=8`, `K>=2`, `cap>=2*K`, `S<=512` batch geometry → scheduler config.
- PILOT constants: ring 4096, worker cap 8, queue-full watermark `<4096`.
- `active_of[]` container-layer remap; tensor-name template strings; `expert_gs` grouping.
- Format detection by nbytes (fmt 1/4/5) → descriptor-declared formats.
- Router-coupled policy state: `momentum_logits`, `pilot_smooth`, `pilot_conf_limit`, freq heatmap — scheduler inputs fed BY the engine, not store state.
- `router_mass` parameter threading through acquire.

## 2. olmoe.c duplicated machinery (~500-line class) and deltas

| Mechanism | olmoe.c location | Delta vs qwen36 |
|---|---|---|
| `Slot/LCache` | 73–74 | INT8-only slot (no int3/int4/unpack variants, no allocated_weight_bytes); **NO `loading[]` in-flight registry** |
| `expert_get` | 486–548 | Same hit scan / victim scan / pinned+inflight skip / while-all-inflight spin (shared origin verbatim), but monolithic load inline (no acquire/finish split, no coalesce-on-loading, no shadow serve) → duplicate demand-vs-pilot loads still possible here |
| Pilot globals/ring | 102–113 | Identical shape; single worker only (no CAS claim, no `COLI_PILOT_W`) |
| `lfru_score` + evict guard | 115–119, 846–857 | **OLMOE-SPECIFIC**: guard compares candidate vs victim LFRU score `heat<<8|recent`; drops speculation if `cs <= vs+(vs>>2)+(4<<8)`; needs `last_access[l*E+e]` uint64 table (Model field, 99) |
| `pilot_realload` | 812–873 | No coalesce registry; on all-pinned/inflight → skip; guard check before steal |
| `pilot_worker` | 875–890 | plain r++ (single consumer) |
| `pilot_prefetch` | 892–1010 | Same EMA/WIDE/CONF_LIMIT/candidate-eid-sort; no C-intent trace events; queue-full rolls back `is_queued` |
| `pin_hot_experts` | 551–625 | Same logic; `freq` is `uint32_t*[layers]` pointer rows owned by route_trace.h |
| `load_expert_merged` | 458–483 | INT8-only + exact-size security validation (reject untrusted containers); `st_read_raw(drop=g_expert_drop)`, `st_read_f32` scales; no fused read, no formats |
| Telemetry | – | Only `hits/miss` + RSS; no decomposition, no trace streams |

OLMoE-specific env knobs: `PILOT_EVICT_GUARD` (default 1), `EXPERT_DROP` (default 0),
`FUSED3` (default 0 — AVX2 activation quant + paired matmul, COMPUTE not storage),
`COLI_USAGE` (freq seeding history), shared `PILOT/WIDE/HOT/WARMUP/SMOOTH/CONF_LIMIT/SNAP/CTX`.

## 3. Existing reusable surfaces (verified present)

- `c/expert_store.h`: generic lease API (`lookup/release/prefetch/stats/destroy`),
  `ColiExpertKey{layer,expert}`, `ColiExpertView{gate,down,up,lease}`,
  `ColiExpertStoreStats{requests,hits,misses,prefetched,prefetch_hits,bytes_read,resident_bytes,capacity_bytes}`.
  **No reservation/pin/batch ops today; no engine except DeepSeek V4 consumes it; qwen36.c/olmoe.c do NOT use it.**
- `c/expert_store_registry.{c,h}`: name→open-fn registry, constructor registration,
  `COLI_EXPERT_STORE` env selection, "auto" builtin = `coli_v4_expert_store_open_planned`.
  **Open contract is DeepSeek-coupled: takes `ColiV4Engine*` + `ColiDeepSeekV4Config*` + options.**
- `c/route_trace.h`: engine-agnostic routing telemetry precedent (freq heatmap owner for olmoe).
- `c/st.h` shard/tensor table (name→fd/off/nbytes), `c/quant.h`, `native_quant*.h`, `uring.h`, `iobench.c`.
- Harnesses: `tests/test_expert_store_ops.c`, `test_expert_store_registry.c`,
  `test_route_trace.c`, `test_eslot_inflight.c`, `test_pilot_ring.c`,
  `test_qwen36_ctx/i4_kernel/mixed_cache`, `test_olmoe_matmul_q`, `test_olmoe_serve_framing`
  (in `c/tests/`), `qwen36_policy_sim.c`, `qwen36_trace_replay.c`,
  `tools/make_qwen36_tiny.py`, `make_qwen36_oracle.py`, `residency_sim.py`.
- Makefile: `test-c` target builds `$(TEST_BINS)` under `c/tests/`.

## 4. Extraction boundary implied by the census (feeds Phase 2–4 design)

- **Store v1 owns**: slot table, LRU/victim scan, pins, in-flight reservation identity
  (replaces `loading[]` + `eid==-1` sentinel), lease/release, publish/abort, batch lookup,
  authoritative resident/in-flight queries, capacity enforcement, byte accounting.
- **pread backend implements**: format-aware admission (descriptor-declared formats),
  weights/scales decomposition, fused contiguous-read optimization (geometry-checked),
  fadvise-drop policy, pread I/O. Keyed internally `(layer, role=EXPERT, index)`.
- **Scheduler owns (policy)**: dedupe, coalescing windows, bounded parallelism semaphore,
  batch windows + first-touch ordering, prefetch ordering/candidate selection inputs,
  pilot worker pool sizing, LFRU-style eviction-guard scoring hooks (OLMoE guard becomes
  a policy callback/config, not a store fork), demand/preload accounting classification.
- **Engine keeps**: router math, EMA/momentum state, freq collection, tensor names,
  `active_of` remap (adapter maps engine layer ids → store keys), spec ring (for now).
- **Registry**: descriptor carries geometry (layers/experts/expert-width provider or
  tensor-name template), formats list, storage backend (shard table ref), capacity bytes,
  scheduler config; DeepSeek adapter builds descriptor from V4 engine/config.

## 5. Abstraction-leak checklist (review before Phase 3 lands)

Generic headers must NOT contain: `eid=-1` sentinel semantics · PILOT ring/worker
constants (4096/8) · `K<=8`/`S<=512`/`bidx[8]` batch geometry · fmt literals
1/4/5 detection-by-size · tensor-name strings · `active_of` remap · router EMA/
CONF_LIMIT/WIDE/SMOOTH concepts · LFRU score formula (policy-side) · `expert_gs`
grouping math (backend descriptor data, not contract code).

## 6. Validation surfaces (from plan §3, confirmed against tree)

Mac: `make test-c` (+ new unit files), tiny-model oracle parity via
`tools/make_qwen36_tiny.py` + `make_qwen36_oracle.py`, `qwen36_trace_replay`,
`qwen36_policy_sim`, Python unit tests where applicable.
Node-02 (no Python): real-container parity + A/B on pinned container
`/home/nayte/models/qwen36_mixed_low` (see `forge_f1_container_pinning_20260823.md`),
protocol = corpus_persistent_serving cap128 ebits4 EP=1 PILOT=1 OMP=8 TEMP=0 N_NEW=64,
comparable anchor **4.98 tok/s warm median**, promgate factorial layout
(20 runs = 5 counterbalanced blocks × {A,B,C,D}, dispersion <1.5%).
