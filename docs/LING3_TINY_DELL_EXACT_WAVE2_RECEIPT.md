# LING 3.0 TINY — DELL EXACT WAVE 2 RECEIPT

Branch: `perf/ling3-exact-wave2-decode-20260824`
Head at receipt: `a33648e` base + closeout commits (see §Final identity)
Status: **CLOSED — Strikes 1–8 adjudicated; controller corrections C1–C8 applied;
final verdict below.**

## FINAL VERDICT

```
LING3_TINY_COLIBRI_DELL_EXACT_WAVE2_PROVEN
```

Correctness gates pass · default flag semantics unambiguous · allocation proof
independently closed · post-init reads directly measured · Strike 7 adjudicated ·
Strike 8 profile captured · same-window frozen control shows no material
regression · this receipt is source-accurate. Builder evidence only — independent
Wave-2 verification is the controller's next step (fresh verifier from the exact
final SHA; no self-verification claimed).

## Environment / identity
- Target: anvil-node-02 (ANVIL Dell Node): i7-11700, 8C/16T, gcc 15.2, AVX2
- Model: official INT4 `/data/ANVIL/ling3/hf_int4` (rev `65a6d1d7…`, unchanged);
  BF16 reference `/data/ANVIL/ling3/hf_tiny` (`b61f4338…`)
- Prompt: frozen `bench_prompt.ids` (256 deterministic ids `1000+(i*7919)%150000`);
  greedy; ngen=64; production thread policy (prefill=16, decode=8)
- Build contract: `make -C c ARCH=native` ⇒ `-O3 -march=native -fopenmp -Wall …`

## Measurement-window discipline (new, load-bearing)

Mid-closeout the node showed ±30 % bimodal wall-clock noise (decode 6.5→10.2,
prefill 12→28.7 tok/s) **uncorrelated with binary**. Root cause class:
`powersave` governor frequency/power-state drift; perf stat proved CPU work
identical (cycles/instructions equal) while wall time differed. All adjudicating
measurements were therefore taken under `performance` governor in quiet windows
(3-consecutive-check load gate); per-rep load/iowait/MHz telemetry recorded;
**no rep discarded** — contaminated reps labeled (§C6). Governor/perf/yama sysctls
were tuned temporarily with sudo and RESTORED (paranoid=4, yama=1, powersave).

## Accepted banked work (unchanged)

Strikes 1, 2, 2.5, 3R1 promoted; 4, 5 rejected with evidence; `w_rowdot` SIMD
rejected (reduction order changed routing). Builder neighborhood was
prefill ~28.4–28.8, decode ~9.4–9.6, RSS ~8.07 GB — see §C6 for the corrected
reading of those numbers.

## STRIKE 8 — profile of the final default (MEASURED)

Clean capture on final binary, default flags (`L3_PHASES=0`, rejected paths off):

`perf stat` (whole run incl. init): cycles 853.9 G · instructions 578.1 G
(IPC 0.68) · cache-references 5.18 G · **cache-misses 3.23 G** · branches
31.5 G · branch-misses 232 M (0.74 %) · ctx-switches 7 687 · cpu-migrations 72 ·
page-faults 2.25 M.

`perf record -F 997 -g --call-graph dwarf` top symbols (self %):
| symbol | self % |
|---|---|
| `matmul._omp_fn.0` (quant.h f32 matvec/GEMV) | **65.1 %** |
| libgomp runtime (barriers/team spin, 4 syms) | ≈17.7 % |
| `exp_matvec._omp_fn.0` (routed experts AVX2) | 9.8 % |
| `mla_forward._omp_fn.0` | 1.9 % |

First record at default frequency lost 71 % of dwarf samples ("Check IO/CPU
overload"); the 997 Hz re-record reproduced the same ranking (matmul 69.7 %
in record 1 vs 65.1 % in record 2). `perf annotate`: hot region is the
weight-streaming FMA/dot inner loop of `matmul`.

Phase decomposition (quiet rep, `L3_PHASES=1`): attn 9.475 s [kda 8.203 |
mla 1.272 | WORK rec 0.736 abs 0.876 lat 0.364 val 0.467] · norms 0.027 ·
router 0.178 · experts 2.194 · shared 1.229 · head 2.070 · accounted 15.172 s.

KDA attribution: recurrence *work* is 0.74 s of an 8.20 s KDA wall — the wall
is KDA **projection matmuls** (the f32 `matmul`). MLA likewise concentrates in
projections + absorb AXPY (0.88 s). LM head f32 traversal 2.07 s.

Dynamic OpenMP team-launch count (native `GOMP_parallel` interposer,
`c/gomp_probe.so`, runtime-measured not grep-derived): PREFILL 169 000,
DECODE 56 448 for the bench run = **≈882 team launches per decode token**
(falsifies the in-source "~150/token" comment, now removed).

### C4 resolution — bandwidth-bound classification

The old "weight-bandwidth-bound at f32" explanation is upgraded **INFERRED →
MEASURED**: 3.23 G LLC cache-misses against 578 G instructions (~1 miss/179
instr) plus ≥65 % of cycles inside the streaming f32 `matmul` whose annotate
profile is load/FMA-bound. The available counters do NOT separate DRAM
bandwidth saturation from core/execution pressure inside that kernel
(no LLC/memory-bandwidth event set captured) — residual split between those
two remains UNMEASURED. The Boss is the shared-surface f32 `matmul`; per
ownership rules that is banked as **Wave 3**, not mutated here.

## SAME-WINDOW CONTROL (C6) — no regression; historical gap falsified

Binary A = frozen Wave-1 `6b0dd8f` rebuild, SHA256
`15bcd6fa2e4e6b6bc658ccf63e047830e7c56f4dfefcba79e32584b50c767f10` (hash-equal
to the verifier's frozen bench binary). Binary B = final Wave-2 candidate.

Round 1 (candidate then at `37c5a90b…`), counterbalanced A B B A ×2, all reps:

| rep | bin | prefill tok/s | TTFT s | decode tok/s | note |
|---|---|---|---|---|---|
| 1 | A | 28.5 | 1.135 | 10.0 | |
| 2 | B | 28.7 | 1.147 | 10.1 | |
| 3 | B | 28.3 | 1.159 | 10.1 | |
| 4 | A | 18.2 | 1.242 | 8.2 | **CONTAMINATED** — external load spike 4.7→7.9 mid-rep (sibling llama.cpp job woke); retained+labelled per predeclared rule |
| 5 | A | 28.0 | 1.153 | 10.1 | |
| 6 | B | 28.6 | 1.175 | 10.1 | |
| 7 | B | 28.7 | 1.115 | 10.1 | |
| 8 | A | 28.1 | 1.187 | 10.0 | |

Round 2 confirm after final source state (`A B B A`): A {10.1, 9.8},
B {10.1, 10.1} (A-rep2 prefill 20.5 during rising load — labelled).

Adjudication (clean reps): decode median A 10.05 vs B 10.10; prefill
overlaps fully; RSS identical 8.07 GB. **No material regression. The builder's
"9.4–9.6 vs Wave-1 10.0–10.1" gap does NOT exist under controlled clocks —
FALSIFIED as environmental measurement drift.**

## ALLOCATION PROOF (C1) — closed independently

1. Source-complete wrappers: every heap call in `ling3.c` now routes through
   `falloc/fcalloc/xmalloc/xcalloc/xfree` (phase-guarded counters on both
   allocs AND frees; a raw malloc/calloc/free/realloc elsewhere in the TU is a
   grep-visible violation — census clean). Stale `g_allocs--` legacy decrement
   (workspace `moe_idx` site) removed; it drove the printed counter to −1 and
   was itself evidence the old printout was not authoritative.
2. Honest wording: `[L3-ALLOC] ling-wrapped post-init allocs=0 frees=0`
   states exactly what it counts (measured run prints 0/0).
3. Process-global interposer (`LD_PRELOAD` `c/l3_allocprobe.so` against
   `-DL3_ALLOC_PROBE -rdynamic` engine exporting `g_l3_alloc_phase`
   INIT/PREFILL/DECODE): INIT 432 088 malloc / 206 899 calloc / 459 realloc /
   144 193 free (~10 GB, model expansion) · **PREFILL malloc=1 calloc=15
   realloc=1 free=0 (<1 MB)** · **DECODE malloc=1 free=5 (<1 MB)**.
   These are libc/libgomp/loader-internal (Ling-wrapped counters simultaneously
   0; zero raw sites in TU) — classified as external, so process-global zero is
   NOT claimed.

Required claims: `Ling-owned steady decode alloc calls = 0` PROVEN;
`Ling-owned steady decode free calls = 0` PROVEN.

## POST-INIT MODEL READ PROOF (C5) — directly measured

Final binary, strace attached AFTER init (`ptrace_scope` temporarily relaxed
via sudo, restored after): attached to all 16 threads before prefill, covered
full prefill + 96-token decode to exit. Trace contains ONLY 16 thread-exit
lines — **zero read/pread64/readv/preadv syscalls post-init**. Same strongest
form as Wave-1 Gate 12, now re-proven on the final candidate.

## STRIKE 7 — fused greedy argmax (REJECTED as default; opt-in retained)

Design: `L3_ARGMAX_HEAD` numeric flag; eligible only when exact-greedy output
is needed — auto-disabled when `L3_LOGITS` requires materialized logits or when
head fmt ≠ f32. Kernel threads OVER rows (never within a row): per-row scalar
accumulation order identical to `w_matmul`→`matmul` S==1; ascending scan +
strictly-greater compare merged lowest-index-on-tie == `sample_greedy` tie
behavior. First implementation computed argmax at every prefill position
(violating the unchanged-prefill requirement via 256 extra full-vocab scans);
caught and fixed before any timing use — head runs only where the OFF path ran.

Exactness: OFF vs ON generated streams byte-identical across 6 independent
runs (single md5 `f1740a92…`); prefill/reference logits mode unchanged
(`t<C-1` skip preserved).

Performance (same-binary, quiet window, alternating OFF/ON ×3):
OFF decode {10.2, 10.1, 10.1} vs ON {10.2, 10.2, 10.1}; ON prefill unchanged.
Delta ≈ +0.5 % = sub-noise. **Promotion bar ("improves beyond run noise") not
met → rejected as default, retained behind `L3_ARGMAX_HEAD`.** No LM-head INT8
in Exact Wave 2 (respected).

## CORRECTIONS LEDGER (controller C1–C8)

| # | correction | resolution |
|---|---|---|
| C1 | allocation counter incomplete/not authoritative | wrappers made source-complete; frees counted; stale decrement removed; wording narrowed; interposer adds process-global split (§Allocation) |
| C2 | Strike 6 never actually default-promoted | CONFIRMED by source (`g_mla_simd=0`); receipt's "PARTIALLY PROMOTED" claim superseded: Strike 6 = REJECTED/retained-for-experimentation, strictly opt-in `L3_MLA_SIMD` |
| C3 | presence-based unsafe flag semantics | fixed: `l3_flag()` parses once at startup (unset/0=OFF, positive int=value, anything else exit 1); hot-path `getenv("L3_TOP8")` removed; deterministic discriminator `L3_FLAGS_DUMP=1` exits before model load; behavioral proof `=0 ≡ unset` |
| C4 | bandwidth-bound was INFERRED | MEASURED (perf stat misses + symbol profile); DRAM-vs-core split inside kernel stays UNMEASURED |
| C5 | post-init reads needed direct proof | done on final binary, strongest form (§Read proof) |
| C6 | same-window control missing | done, two rounds counterbalanced, no regression; builder gap FALSIFIED as environmental (§Control) |
| C7 | parity refs ambiguous | every comparison names exact refs/hashes below |
| C8 | stale banner | updated to actual contract AFTER proofs closed (zero Ling-owned steady-state heap traffic; numeric opt-in flags) |

Earlier mistaken claims are retained above under their original sections and
superseded here by explicit reference (none erased).

## Parity comparison refs (C7)

| comparison | refs | result |
|---|---|---|
| final vs Wave-1 frozen | B=`2fa7bcc7…`(bin) @ src `a33648e`+closeout vs A=`15bcd6fa…`(bin) @ `6b0dd8f` | full-trace GLOBAL max\|diff\| = **1.75476e-04** (@float 4704817) — identical to the KDA lineage delta; streams/routes bit-identical |
| scalar reference vs two-pass KDA | `trace_scalar.f32` vs `trace_final.f32` | GLOBAL max\|diff\| = **1.75476e-04** (same float index) — reproduces banked integrated delta exactly |
| Wave-1 A vs scalar-KDA | `trace_A.f32` vs `trace_scalar.f32` | **byte-identical** (cmp clean) — proves two-pass KDA is the sole arithmetic-lineage change of Wave 2 |
| final vs post-Strike-2 `2cef53a` | not separately built | SUBSUMED: A predates all Wave-2 strikes and equals scalar-KDA bitwise, so any intermediate delta is ≤ the measured 1.75e-04 lineage bound; localization to Strike 2 already carried by the scalar discriminator |
| router parity (full bench) | `route_A.bin` vs `route_B.bin`, native comparator `rcmp` | 7360/7360 records (5888 prefill + 1472 decode), **0 selection mismatches**, max w-diff 4.89e-06 (fp bits only; byte diff at low-order weight bytes explains raw `cmp` mismatch) |
| TF router adjudication | `xldir_closure` + `fixture_closure/moe_in`, 4-layer BF16 | **63/63 exact**, max w-diff **6.75e-07** == banked verifier value |
| layer-0 TF gate | ref `hidden.f32[1..]` ↔ got trace `[0..]` | L0 max **0.0272282** == sealed floor (PASS). L1 0.130042 / L2 0.208206 vs verifier-banked 0.128071/0.208034: BOTH binaries (A and final) produce these values BIT-IDENTICALLY today, and scalar-vs-two-pass is inert on this fixture — the banked numbers came from the verifier's stricter per-sublayer dump protocol, not chained-stage traces; recorded as provenance note, not regression |

## Thread-policy sanity (T1/T8/T16)

`L3_THREADS_DEC ∈ {1,8,16}` on final binary: one unique stream hash across all
three (md5 count 1); throughput sane (DEC=8 knee consistent with banked sweep).

## Sanitizers & suite

- `make -C c test-c` (final source, Dell gcc): **PASS** (incl. st dtype fork
  negatives, e8/rans oracles, corpus_draft, uring). One transient FAIL of
  `test_uring` traced to /tmp tmpfs user-quota exhaustion caused by MY OWN
  abandoned first perf.data (5.8 GB) — deleted; suite green after. Environment,
  not code.
- ASan (`detect_leaks=0`, 6-layer truncation, prefill+8 decode): rc=0, zero
  reports. Full-model ASan exceeds physical RAM (shadow overhead >14 GiB box) —
  truncation documented.
- UBSan (same shape): rc=0, zero diagnostics.
- TSan disposition: **no newly parallel DEFAULT path survives closeout** —
  Strike 7 is opt-in/rejected; C3/wrapper changes are serial startup/init code.
  Spot-check of the opt-in region shows the known gcc-libgomp unannotated
  false-positive class only (baseline OFF arm 64 warnings vs ON 70, all
  "worker reads main-thread stack" at `quant.h:99`/omp frames; zero
  worker-to-worker mutable races). Bit-exact stream identity is the operative
  correctness evidence for the flag path.

## RSS / peak RSS

8.07 GB steady across every quiet rep; peakRSS == steady (no growth);
matches frozen Wave-1 exactly.

## Hygiene

- Tracked-binary census: `git ls-files | grep -E '^c/(ling3|ling3_check|l3_tracecmp|ling3_probe|ling3_asan|ling3_ubsan|ling3_tsan)(\.exe)?$'` → EMPTY on pushed tree; probe tooling sources tracked, binaries ignored/untracked.
- Node worktree synced to pushed HEAD via git pull; `git status --porcelain`
  clean modulo regenerable artifacts outside tracking.
- Node sysctls restored (perf_event_paranoid=4, yama=1, powersave).

## Final identity

- Owned branch tip (pushed, non-force): see battle report header line.
- Final candidate binary: `c/ling3` SHA256 `2fa7bcc781f9c706…` (full hash in
  battle report) built from the pushed tree with `make -C c ARCH=native`.
- Probe tooling: `c/l3_allocprobe.c`, `c/gomp_probe.c`,
  `ling3_probe` build (`-DL3_ALLOC_PROBE -rdynamic`).

## Remaining frontier (next measured Boss)

1. **Wave 3 recommendation:** the shared `quant.h` f32 `matmul` (≥65 % of
   cycles; KDA/MLA projections + dense + LM head all funnel through it).
   Shared-surface change ⇒ requires its own isolation wave per ownership law;
   Ling-local fused projection execution is the Ling-local alternative lane.
2. LM head f32 traversal (2.07 s wall) — second target, partially addressable
   by the retained `L3_ARGMAX_HEAD` if a future exact algorithm beats noise.
3. Expert row-kernel vectorization inside `exp_matvec` AVX2 (2.19 s).
Quality-changing lanes remain explicitly deferred.
