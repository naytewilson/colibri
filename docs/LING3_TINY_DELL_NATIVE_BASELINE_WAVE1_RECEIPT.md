# LING 3.0 TINY — DELL NATIVE BASELINE + EXACT THROUGHPUT WAVE 1 — CLOSURE RECEIPT

Campaign: Ling-3.0-tiny native Colibri port (`feat/ling3-tiny-colibri`)
Wave: Rapid-port baseline (Waves 0–3) + first exact throughput wave (Wave 4, item 6/7)
Closeout: this receipt banks the frozen state and freezes the fresh-verifier target.
Milestones: `LING3_TINY_COLIBRI_DELL_NATIVE_BASELINE_PROVEN`,
`LING3_TINY_COLIBRI_DELL_EXACT_THROUGHPUT_WAVE1_PROVEN` (both pending independent
verification; see §Parity for a correction issued at closeout).

---

## 1. Source provenance

| Item | Value |
|---|---|
| Repository | naytewilson/colibri |
| Branch | feat/ling3-tiny-colibri |
| Baseline-wave HEAD | `bee024bc444ed75d829ad31631b83e1d415051d1` |
| Closeout HEAD | `65d1bd2bd7a48bc47f2f8b180495758061680c10` (+ receipt commit) |
| Handoff base | `4e86b1e7e8712da7aaafd3fcbdd7b402043cd893` |
| Campaign commits | ae21285 engine · 0815a58 correctness ladder · bee024b throughput wave 1 · ad016ce closeout core · 65d1bd2 portable XLDIR dumper |
| Official BF16 revision | `inclusionAI/Ling-3.0-tiny` @ `b61f4338de3e68ffc9c0bc1ed5e902981a4a929e` |
| Official INT4 revision | `inclusionAI/Ling-3.0-tiny-int4` @ `65a6d1d71e01f73ba01e572992bbd69ea92c865f` |

## 2. Target census — anvil-node-02 (ANVIL Dell Node)

i7-11700, 8C/16T, 14 GiB usable RAM (NOT 16 GB), Ubuntu, gcc 15.2,
AVX2 + AVX-512 family, NVMe + 4 TB `/data`. Model store `/data/ANVIL/ling3/`.

## 3. Artifacts

| Artifact | Identity |
|---|---|
| INT4 container | 32 shards, 5.5 GiB on disk, aggregate sha256-of-manifest `34864b047a6055006355f1e7f4d4f90cdf94e76611ae1bd8a40061be3590c902` |
| Frozen-bench binary (closure re-run) | ling3 @ ad016ce, gcc `-O3 -march=native`, SHA256 `15bcd6fa2e4e6b6bc658ccf63e047830e7c56f4dfefcba79e32584b50c767f10` |
| Bench prompt | `/data/ANVIL/ling3/bench_prompt.ids`, 256 ids `1000+(i*7919)%150000`, SHA256 `1e0ec996ec7ccd7c47756d5a6e5b88f36172c724ec99f02cc9f093712bc65e17` |
| Parity prompt | `/data/ANVIL/ling3/prompt_a.ids`, 21 chat-template ids, SHA256 `b15fc2340181ba3cda52883c82e46f28078822f21d8fd3831e6a7e0c7618a1cd` |
| INT4 format | compressed-tensors pack-quantized int4-g32 symmetric: `weight_packed` I32 [O, I/8] dense LSB-first bitstream, unsigned nibbles (q+8); `weight_scale` BF16 [O, I/32]; verified vs BF16 checkpoint (cosine 0.9953 = g32 noise) |

## 4. Frozen benchmark protocol

```
cd /data/ANVIL/ling3/colibri/c
L3_PHASES=1 ./ling3 --model /data/ANVIL/ling3/hf_int4 \
  --ids-file /data/ANVIL/ling3/bench_prompt.ids --ngen 64
```
Greedy (COLI_TEMP unset ⇒ 0). 3 repetitions per arm.

### Baseline v1 (scalar expert kernel, T16) — raw reps
| rep | prefill tok/s | TTFT s | decode tok/s | RSS GB |
|---|---|---|---|---|
| 1 | 18.9 | 1.694 | 8.1 | 8.07 |
| 2 | 18.9 | 1.691 | 8.1 | 8.07 |
| 3 | 18.9 | 1.689 | 8.0 | 8.07 |
median decode **8.1** (spread ±0.2 tok/s ≈ ±2%), experts wall 7.77 s of ~8 s decode (**98% of token wall**).

### Wave-1 final (AVX2 int4-g32 pair-split kernel + hybrid threads) — raw reps
| rep | prefill tok/s | TTFT s | decode tok/s | RSS GB | experts s |
|---|---|---|---|---|---|
| 1 | 27.8 | 1.213 | 10.0 | 8.07 | 2.162 |
| 2 | 28.3 | 1.123 | 10.1 | 8.07 | 2.075 |
| 3 | 28.5 | 1.120 | 10.1 | 8.07 | 2.076 |
median decode **10.1** (+26% vs baseline), prefill **28.3** (+50%), TTFT 1.12 s (−34%).

### Phase decomposition (baseline → wave-1, medians)
attn 9.61→9.44 s · norms 0.027→0.026 · router 0.140→0.137 · **experts 7.77→2.08 (3.7×)** · shared 1.27→1.22 · head 2.16→2.08 · accounted 20.98→14.87 s.

## 5. Residency

Fully resident: RSS 8.07 GB steady on 14 GiB host; resident INT4 weights 8.13 GB
accounted at load; every expert behind direct [layer][expert][matrix] pointers.

**Physical model-weight reads: `ZERO_MODEL_WEIGHT_READS_PROVEN`.**
Method: `strace -p <pid> -f -e trace=pread64,read` attached AFTER `init done`
(fd map captured: all 32 shard fds open), run to process exit across prefill +
96-token decode. Result: **0 read syscalls, 0 bytes read from ANY fd**
(`/data/ANVIL/ling3/readproof/syscalls.txt`, 0 lines). Strongest userspace-provable
form: not merely zero model-weight reads, zero file reads of any kind after startup.

## 6. Kernel discriminators (re-run at closeout)

| Arm | Result |
|---|---|
| Scalar vs AVX2, full-depth bench traces (256+4 tok) | max abs drift **4.27e-04** over 9,603,072 f32 values — fp32 reduction-order noise; no structural divergence (`l3_tracecmp`) |
| Thread invariance | T1 vs T16 teacher-forced traces **bit-identical** (row-parallel determinism) |
| L3_NO_AVX2 fallback build | compiles clean, runs (prefill 18.6 / decode 8.4 tok/s ≈ baseline scalar ✓) |
| Instrument | `c/l3_tracecmp.c` committed (native comparator; per-stage grouping) |

## 7. Thread policy (closeout semantic fix, FINDING 5)

Old behavior: `L3_THREADS=N` constrained only prefill; decode hardcoded 8.
New contract (committed):
```
th_pref = L3_THREADS_PREF > L3_THREADS > omp_get_max_threads()
th_dec  = L3_THREADS_DEC  > L3_THREADS  > min(8, max_threads)
```
Set-but-invalid (<=0, non-numeric) rejected explicitly with `[L3] ... rejected`.
Discriminated on Dell (all 7 cases): default resolves `prefill=16 decode=8`;
generic var binds both; specific overrides generic; 0/-3/abc rejected.
**Dell default preservation:** closure re-run of the frozen protocol with NO env:
decode 6.349–6.380 s vs frozen 6.336–6.385 s (within noise); r3 prefill 28.3 ==
frozen median (r1–r2 prefill elevated by post-rebuild page-cache warmup only).
Documented: **8 is the measured Dell i7-11700 decode knee, not a universal constant.**

Knee evidence (Wave-1 sweep {1,4,8,16}): decode peaks T8 (10.1 tok/s), regresses
T16 (9.4); prefill scales to T16.

## 8. Correctness / parity — HONEST CLAIM BOUNDARY (FINDING 2 applied)

Fixture A: official modeling code (transformers 4.57.6 + fla CPU shim), fp32,
explicit 4D causal mask (the direct-constructed model runs bidirectionally
without it — prefix-invariance-proven during the campaign), 21-token frozen
prompt, layers 0–3. Native adjudication via `c/ling3_check.c`.

**PROVEN at closeout (reproduced on commit ad016ce against freshly regenerated
coherent fixture round `fixture_closure` + `xldir_closure`):**
- Router exactness: **63/63 selections exact, max w-diff 6.75e-07** (all sparse-layer × position decisions).
- Layer-0 output equivalence at the historical sealed noise floor: **max 2.72e-02** == the torch-replica noise floor banked when layer 0 was sealed.
- Tokenizer/raw-id path, embedding lookup, conv layouts, KDA recurrence scalars: sealed during the correctness ladder.

**OBSERVED (quantified, not classified as parity-passing):**
- Raw-stage drift grows through the sparse layers: outL1 0.128, outL2 0.208 (teacher-forced inputs).
- Post-final-norm state max diff 0.567; logits max 1.05, mean 0.131, greedy argmax 18/21 positions.
- Consistent mechanism: official reference computes experts in BF16-upcast fp32;
  the engine consumes the OFFICIAL INT4 artifact natively (per-weight g32 relerr
  ~10% by design). Final RMSNorm divides by a small residual RMS (~25×) and
  amplifies accumulated expert-quantization drift. Router selection is immune
  because MoE sublayer inputs are teacher-injected.

**FALSIFIED (as unreproducible from any commit):**
The previously reported ultra-tight chain numbers ("post-final-norm 4.5e-05,
logits 3.15e-05 max, argmax 21/21") could NOT be reproduced from `bee024b` or
`ad016ce` under any env combination tested (scalar, AVX2, NO_AVX2, T1/T16).
Preserved artifacts (`run_iso.trace/.logits`, which DO match the current fixture
at 4.51e-05 / 3.147e-05) were produced by an unrecoverable working-tree state
and must not be cited as properties of the committed engine. They remain frozen
on the node as historical evidence.

**Claim boundary now in force:**
- `ROUTER_EXACTNESS_FIRST_4_LAYERS_PROVEN`
- `LAYER_0_DENSE_KDA_NOISE_FLOOR_SEALED`
- `UNIQUE_LAYER_TYPES_AND_ROUTER_SEMANTICS_COVERED` (dense MLP, KDA, sparse MoE, grouped router, shared expert, gated MLA)
- Full-chain first-4-layer parity: NOT CLAIMED beyond the above.
- `FULL_24_LAYER_OFFICIAL_PARITY`: **UNMEASURED** — mandatory fresh-verifier gate.

## 9. Mission-A note (full-depth validation)

A 24-layer BF16 official reference cannot fit simultaneously in 14 GiB
(snapshot is 15 GB); the bounded route requires a streamed per-layer capture
harness that does not yet exist in the repo. Rather than fake depth, the
verifier gate includes building it. The verifier must also settle the
REFERENCE-SOURCE discriminator this closeout surfaced: whether INT4-engine
parity should be adjudicated against the BF16 snapshot (quality reference,
current fixture convention) or against an INT4-dequantized snapshot
(representation reference — would isolate engine math from by-design
quantization noise). Toolchain committed for either: portable
`ling3_refgen_official.py` (script-relative shim import, explicit causal mask),
portable `ling3_dump_layerinputs.py` (XLDIR companion), native `ling3_check`.

## 10. Shared-surface hardening (MISSION B, FINDINGs 1/3/4/6)

- st.h raw-dtype identity FAIL-CLOSED: distinct ST_U8/I8/I16/I32/I64 codes (float codes 0/1/2 unchanged); float readers accept ONLY BF16/F16/F32; `st_read_raw` accepts ONLY raw integer dtypes; malformed dtype/width fails closed. New test suite `tests/test_st_dtypes.c` (incl. fork-based negative tests). Repo-wide `make -C c test-c` green on Mac (clang) AND Dell (gcc), exit 0 both.
- ling3 packed-expert loader requires exact `ST_I32`; scale loader requires exact `ST_BF16`.
- Tracked binaries removed (`c/ling3`, `c/ling3_check`); `.gitignore` covers ling3/ling3_check(.exe); make regenerates them (verified on both hosts).
- `tools/ling3_refgen_official.py`: script-relative shim import (old hardcoded `/data/ANVIL/ling3` broke non-Dell checkouts); validated end-to-end on Dell from arbitrary cwd.
- Engine header corrected: steady-state decode is NOT zero-allocation (AVX2 x-even/x-odd scratch per exp_matvec call + moe/attention temporaries); persistent scratch is the next exact wave.

## 11. QWEN36 transfer matrix status

Frozen control untouched (QWEN36 production source/deployment never opened).
Classifications per `docs/LING3_TINY_QWEN_TRANSFER_OPTIMIZATION_DOCTRINE.md`;
wave-1 dispositions: DIRECT RESIDENT INDEXING=APPLY(promoted), LOW-BIT
NATIVE KERNEL=APPLY(promoted AVX2 pair-split), THREADING KNEE=APPLY(measured
8/16 hybrid), PERSISTENT SCRATCH=DISCRIMINATE(next wave), FUSED I/O=
FALSIFIED_FOR_THIS_REGIME(resident model needs no load path), BATCH
PREPARATION=DEFER, quality-changing lines (expert pruning, mixed precision,
KD, lm_head int8)=DEFERRED until exact frontier exhausted.

## 12. Rejected / falsified paths (cumulative)

"16 GB node" memory claim (14 GiB measured) · QWEN-style streaming for this
regime · naive nibble-order hypotheses for packed int4 (settled via
compressed-tensors source) · transformers>4.57 for official code · bidirectional
fixture generation without explicit 4D causal mask · ultra-tight 4-layer chain
numbers as committed-tree property (this closeout) · lm_head int8 as "exact"
work (QUALITY_CHANGING / DISCRIMINATE LATER).

## 13. Next exact-runtime frontier (Exact Wave 2, in order)

1. persistent scratch/workspace pool (~32 ms/tok alloc churn + OMP overhead)
2. eliminate exp_matvec xsplit malloc/free; reuse split input for gate/up where mathematically valid
3. exact f32/BF16 LM-head vectorization + cache traversal
4. prefill chunk-size sweep at T16
5. re-profile phase decomposition

LM_HEAD INT8 stays OUT of the exact lane until bit-exact representation proof.

## 14. Fresh-verifier target (frozen)

Reproduce independently, in order:
1. build at closeout HEAD on Dell; `make -C c test-c` exit 0 (Linux gates incl. uring);
2. tracked-binary absence: `git ls-files | grep -E "^c/(ling3|ling3_check|l3_tracecmp)(\.exe)?$"` produces NO output; make regenerates each;
3. regenerate coherent fixture round via committed tools from repo checkout (no host-path coupling);
4. adjudicate: router 63/63 @ ≤1e-6 w-diff; layer-0 ≤3e-2; quantify deeper stages per §8 (do NOT expect ultra-tight numbers);
5. thread-policy matrix (§7, all 7 cases) + default resolution print;
6. scalar vs AVX2 discriminator via `l3_tracecmp` (expect ~1e-4-scale drift, no structural divergence); L3_NO_AVX2 compile/run;
7. frozen benchmark 3 reps default policy: decode median 10.1 tok/s ±2%, prefill ~28.3, RSS 8.07 GB;
8. strace physical-read window: ZERO read syscalls post-init;
9. THEN full-depth Mission-A harness (streamed 24-layer official reference) + reference-source discriminator;
10. only after all above: open Exact Wave 2.

## 15. Anomaly log

One unidentified one-time copy event delivered the Mac-built arm64
`l3_tracecmp` binary onto the Dell clone during this session (sha256-identical,
timestamp inside the Dell make window). Impact contained: wrong-arch file
deleted, rebuilt natively (ELF), ALL Dell evidence re-derived from ELF binaries.
Mechanism unresolved — flagged for Nayte; no sync daemon found on either host.

## 16. SOURCE-TRUTH CORRECTION R1 — tracked comparator binary (post-receipt)

Classification: `TRACKED_BUILD_ARTIFACT_CLEANUP = FALSIFIED_AT_B802F76`.

The closeout report and §10 above claimed the tracked Ling binaries were
removed. That claim was INCOMPLETE at the closeout HEAD `b802f76`: the compiled
comparator `c/l3_tracecmp` (Mach-O, blob magic `cffaedfe`) was still tracked.
Root cause: the closeout commit staged it with `git add -A` BEFORE .gitignore
covered the new tool, and the closeout census grep only patterned
ling3/ling3_check. The controller caught this via canonical GitHub fetch before
independent verification. Correction commit: removes `c/l3_tracecmp` from
tracking (source `c/l3_tracecmp.c` remains authoritative; the Makefile
on-demand target `l3_tracecmp$(EXE)` rebuilds it) and extends `.gitignore` with
`c/l3_tracecmp` / `c/l3_tracecmp.exe`. Post-fix census proof:
`git ls-files | grep -E '^c/(ling3|ling3_check|l3_tracecmp)(\.exe)?$'`
produces no output. Regeneration proven clean (`make -C c l3_tracecmp`);
rebuilt binary untracked/ignored and smoke-validated against the banked
scalar-vs-AVX2 traces (reproduces 4.27246e-04 @ float 1131842 exactly).

Commit-count correction to the closeout battle report: the pre-correction count
is **3 commits after bee024b** (`ad016ce`, `65d1bd2`, `b802f76`) — a "4 commits"
wording in that report was wrong; this hygiene correction makes it 4.

This correction is repository hygiene ONLY. No runtime source, kernel math,
thread policy, dtype contract, fixture data, benchmark protocol, or model
artifact changed; all runtime/performance/parity evidence in §§4–8 is unchanged.
