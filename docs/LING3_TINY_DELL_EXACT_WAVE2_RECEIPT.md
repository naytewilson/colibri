# LING 3.0 TINY — DELL EXACT WAVE 2 RECEIPT

Branch: `perf/ling3-exact-wave2-decode-20260824`
Head at receipt: `12551e8` (parent chain from `6b0dd8f` → `2cef53a` → `d55ae67` → `c39eec3` → `ad4a76f` → `12551e8`)
Status: **IN PROGRESS — Strikes 1, 2, 2.5, 3(R1) complete; 4, 5 REJECTED with evidence; 6 partially promoted; 7, 8 remaining.**

## Environment / identity
- Target: anvil-node-02 (i7-11700, 8C/16T, 14 GiB, AVX2+AVX-512, Ubuntu, gcc 15.2)
- Model: official INT4 artifact `/data/ANVIL/ling3/hf_int4` (rev `65a6d1d7`);
  BF16 reference `/data/ANVIL/ling3/hf_tiny` (rev `b61f4338`)
- Prompt: 256 deterministic ids, generator `1000+(i*7919)%150000`; greedy; ngen=64; T16

## Strike 1 — split attention telemetry (PROMOTED)
`t_attn` split into `t_kda`/`t_mla` + sub-stage fields.

## Strike 2 — two-pass KDA state kernel (PROMOTED)
Bit-faithful to the four-sweep reference (standalone discriminator:
max|dS|=max|dvt|=max|doh|=0 at -O2/-O3 after fixing a hoisted-SIMD-accumulator
bug the harness exposed). Integrated scalar-vs-two-pass trace delta 1.75e-04
(scalar path is FMA-contracted by gcc); routes 5888/5888; tokens 64/64.
`L3_KDA_SCALAR=1` retains the reference path.

## STRIKE 2.5 — telemetry semantics repair (PROMOTED)
Racy shared-double writes inside OMP head loops replaced by per-thread work
counters merged via OMP reduction into file-scope globals, printed as WORK
seconds (`WORK kda_rec/mla_abs/mla_lat/mla_val`). `t_mla_ctx` now banks.

## STRIKE 3R1 — persistent workspace (PROMOTED, ASan-clean)
Persistent ws covers step_chunk buffers, KDA/MLA chunk temps, MoE route
arrays + U, union bookkeeping (uid/pcnt/pfirst/poslist/wlist/cur), expert
gate/up/hz (+8 per-slot sets for future top-8), AVX2 xev/xod split, shared
sg/su/sd, dense dg/du, and logits. LM head writes persistent `ws.logits`.
Bugs found and fixed en route: stale `free()` of workspace pointers in
step_chunk tail and dense_forward (ASan-proven heap-use-after-free).
**Allocation proof**: with `L3_ALLOC_COUNT=1`, post-init heap allocations
during a full prefill+decode run: **0** (counter accrues only when phase>0;
remaining init-time allocations are excluded by design).

## STRIKE 4 — prepared input + top-8 outer parallelism (REJECTED as default)
Serial prepared-input kernel changes the decode arithmetic lineage (scalar
order vs the AVX2 row kernel that produced the Wave-1 stream): greedy stream
diverged at token 6. Also measured slower (experts wall 2.17→2.61s).
Kept strictly opt-in via `L3_TOP8=1`.

## STRIKE 5 — serial C==1 matvec (REJECTED, control-proven)
Universal serial regressed decode 9.6→1.4 tok/s; threshold refinement still
regressed KDA to ~27s. Control run (serial disabled, same binary) restored
kda 8.16s. Premise falsified: OpenMP team launches do not dominate small
matvecs on this node; threaded wins down to ~800K MACs. A stale second env-
wiring line kept re-enabling the path — found by audit after two "fixed"
builds. Retained behind `L3_SERIAL_C1=1`.

## STRIKE 6 — MLA SIMD (PARTIALLY PROMOTED)
`w_addrow` AVX2 retained (elementwise, bit-exact). `w_rowdot` SIMD REJECTED:
reduction-order change flips 2/5888 knife-edge router selections; any MLA
arithmetic-lineage change breaks Wave-1 stream identity downstream.
Measured effect of full SIMD variant before rejection: MLA wall −7%.

## Correctness gates at final checkpoint (quiet window, 3 reps)
trace diff vs pre-wave binary: **0** · router selections **5888/5888** ·
greedy tokens **64/64 identical** · RSS **8.07 GB** · decode **9.4–9.6 tok/s** ·
prefill **28.4–28.9 tok/s** · TTFT ~1.10 s · layer-0 envelope untouched ·
post-init model-weight reads: zero by construction (all resident; no new
storage paths).

## Phase decomposition (final, rep2 clean)
attn 9.32s [kda 8.10 mla 1.22 | WORK rec 1.18 abs 1.35 lat 0.41 val 0.52]
norms 0.03 · router 0.14 · experts 2.16 · shared 1.26 · head 2.15 ·
total_accounted 15.05s

## Falsified hypotheses
1. KDA recurrence sweeps dominate the attention wall (rec work ≈1.2s of 8.1s).
2. OpenMP team launches dominate small matvecs on Dell.
3. `L3_ALLOC_COUNT` as originally written proves zero allocation (it only saw
   falloc/fcalloc; raw malloc sites were invisible).

## Remaining exact frontier (next boss ranking)
1. KDA/MLA projection matmuls (weight-bandwidth-bound at f32; exact options:
   better GEMM blocking in quant.h — shared-surface change requiring isolation,
   or Ling-local fused projection execution).
2. LM head f32 traversal (2.15s).
3. Expert row-kernel vectorization inside exp_matvec AVX2 (2.16s).
Quality-changing lane (non-expert load-time int8/int4, LM-head INT8):
explicitly deferred.

## Raw reps
Wave-2 final: prefill 28.5/28.6/28.8 tok/s · decode 9.6/9.6/9.4 tok/s ·
TTFT 1.14/1.14/1.11 s (rep1 of some runs contended by transient node load;
contended reps discarded rather than averaged).
