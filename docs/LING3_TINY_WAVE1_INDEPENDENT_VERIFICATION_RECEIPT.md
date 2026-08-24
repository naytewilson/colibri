# LING 3.0 TINY — WAVE 1 INDEPENDENT VERIFICATION RECEIPT

Campaign: Ling-3.0-tiny native Colibri (`feat/ling3-tiny-colibri`)
Verifier: fresh independent session; no builder state inherited
Candidate under verification: `6b0dd8f29c0a588087595e74fffb88799c294c4e`
Verifier branch: `verify/ling3-wave1-20260824` (started at candidate SHA, additive commits only)

## FINAL VERDICT

```
LING3_TINY_COLIBRI_WAVE1_INDEPENDENTLY_VERIFIED
```

Every claim in the frozen Wave-1 claim boundary survived independent
verification. Two closeout-report *observations* did not survive and are
corrected below (§C); neither is part of the frozen claim boundary. New source
truth for full-depth parity is banked (§8) — it was previously UNMEASURED.

---

## 1. Verifier checkout

| Item | Value |
|---|---|
| Mac verifier worktree | `/Users/nayte/Projects/colibri-verifier-wave1` (fresh worktree at candidate SHA) |
| Dell verifier clone | `/data/ANVIL/ling3/verify/colibri` (fresh `git clone`, checkout of candidate SHA) |
| Candidate SHA | `6b0dd8f29c0a588087595e74fffb88799c294c4e` |
| `origin/feat/ling3-tiny-colibri` | same SHA (verified after `git fetch --prune`) |
| `origin/verify/ling3-wave1-20260824` | same SHA before verifier evidence was added |
| Builder binaries inherited | NONE (all builds re-derived from source) |

## 2. GATE 0 — repository hygiene: PASS

- `git ls-files | grep -E '^c/(ling3|ling3_check|l3_tracecmp)(\.exe)?$'`
  produces NO output on both hosts. PROVEN.
- Rebuild from source clean on both hosts; rebuilt binaries untracked and
  ignored; `git status --porcelain` empty after builds.
- Cross-check: verifier-rebuilt `c/ling3` on Dell (gcc `-O3 -march=native
  -fopenmp -pthread`) has SHA256
  `15bcd6fa2e4e6b6bc658ccf63e047830e7c56f4dfefcba79e32584b50c767f10` ==
  the builder's frozen-bench binary hash from the Wave-1 receipt §3.
  The hygiene-only delta (ad016ce → 6b0dd8f) is confirmed by a bit-identical
  rebuild. PROVEN.

## 3. GATE 1 — source / model identity: PASS

| Item | Value |
|---|---|
| Candidate SHA | `6b0dd8f29c0a588087595e74fffb88799c294c4e` |
| Official BF16 revision | `inclusionAI/Ling-3.0-tiny` @ `b61f4338de3e68ffc9c0bc1ed5e902981a4a929e` — PROVEN against HF Hub API (exact sha match; 32 shards + index; usedStorage 15.8 GB) |
| Official INT4 revision | `inclusionAI/Ling-3.0-tiny-int4` @ `65a6d1d71e01f73ba01e572992bbd69ea92c865f` — PROVEN against HF Hub API (exact sha match; compressed-tensors config identical to local snapshot; storage 5.82 GB ≈ receipt's 5.5 GiB) |
| INT4 config hash | `78296932c0a82af7490095d1962c15fb5e206ac162874516039c7f49280fc3d6` |
| Tokenizer hash | `40fb9d7d7795b8bd305aeff39ce9963f3f450915b9553f2938e009be9a1fed60`; chat template `eb6226c94ae38058f875d159f86a206b3a165828c0e7d6bda664ae14667f798a` |
| Tensor census | 26,947 tensors / 32 shards; dtypes I64 8832, I32 8832, BF16 9224, F32 59 |
| Routed experts | layers 1–23 (layer 0 dense), 128 experts each, gate/up/down per expert; packed geometry gate/up `[512,192]` I32 + `[512,48]` BF16 scale; down `[1536,64]` + `[1536,16]`; router gate BF16 `[128,1536]`, expert_bias F32 `[128]` |
| Dell target | anvil-node-02 (ANVIL Dell Node): i7-11700, 16 threads, gcc 15.2.0, AVX2 |
| Native binary | `c/ling3` SHA256 `15bcd6fa…c767f10` (== builder frozen hash); flags `-O3 -march=native -fopenmp -pthread -Wall -Wextra …` |

Architecture re-derived from LIVE config + code (not campaign prose): 24
layers = 18 KDA + 6 gated MLA (`(i+1)%layer_group_size==0`, layer_group_size=4
→ MLA at 3,7,11,15,19,23); hidden 1536; 128 routed experts, top-8,
n_group=8/topk_group=4 grouped sigmoid noaux_tc, norm_topk_prob,
routed_scaling_factor 2.5, expert_bias enabled, router fp32;
first_k_dense_replace=1; MLA partial RoPE (partial_rotary_factor 0.5,
rotary_dim/qk_rope_head_dim 64, theta 6e6, qk_nope 128, v_head 128, kv_lora
512, q_lora 256); KDA short_conv_kernel_size=4 conv4 recurrence, kda_lower_bound −5;
official compressed-tensors pack-quantized int4-g32 symmetric with ignore list
that leaves ONLY routed experts quantized. All claims CONFIRMED.

## 4. GATE 2 — shared `st.h` dtype contract: PASS

- `tests/test_st`: ok on Mac clang 21.0.0 AND Dell gcc 15.2.0.
- `tests/test_st_dtypes`: ok both hosts. The suite pins distinct ST_U8/I8/
  I16/I32/I64 codes, float readers accepting only BF16/F16/F32, raw readers
  accepting only integer dtypes, byte-exact payloads, and FORK-BASED negative
  tests that require exit(1) with the right message (float reader on every raw
  dtype; raw reader on float tensors). A harness that can fail. PROVEN.
- Engine-level enforcement inspected at source: packed-expert load requires
  exact ST_I32 + nbytes (`c/ling3.c:294`), scale requires exact ST_BF16
  (`c/ling3.c:307`). Fail-closed contract CONFIRMED.

## 5. GATE 3 — coherent fixture regeneration: PASS

Regenerated twice from the verifier's own fresh clone using the committed
portable tool (`tools/ling3_refgen_official.py`, script-relative shim):

| artifact | sha256 |
|---|---|
| `hidden.f32` [5,21,1536] | `ed356aafac74b39d738ce15f708d1538f3d0760fbbec6823bd59943c26bfea7e` |
| `logits.f32` [21,157184] | `24514ef0fa8241314115fe76a9e85bfb484d8df3b23ef72e952d5bfd6f56607c` |
| `prompt_ids.json` | `baf61261ea96728d524911a080e0dbf8418a7f5fdd9c120d26158e88d95f1e2b` |
| `router.json` | `511d2b55417dad0e2850eb15c4a28354d5c8d01de82d2070a46631ca08396c69` |

- Determinism: two independent regenerations BYTE-IDENTICAL. PROVEN.
- Banked comparison: BYTE-IDENTICAL to `/data/ANVIL/ling3/fixture_closure`
  (hidden/logits/prompt/router). No source/tool/model drift. PROVEN.
- XLDIR teacher-forcing inputs regenerated via committed
  `tools/ling3_dump_layerinputs.py`: byte-identical to banked `xldir_closure`.
- Model path recorded: `/data/ANVIL/ling3/hf_tiny` @ b61f4338… (BF16),
  fp32 fixture, explicit causal semantics per the committed tools.
- Stage-map note established during verification: the official model appends
  each layer's INPUT plus the post-final-norm state, so `hidden.npy` stages
  are `[embeddings, outL0, outL1, outL2, post_final_norm]` — there is NO raw
  outL3 stage in a 4-layer fixture.

## 6. GATES 4–6 — router exactness and teacher-forced parity

Adjudication regime matters and was established empirically: router decisions
are exactly comparable only when BOTH sublayer inputs are reference-injected
(`L3_XLDIR` + `L3_MOE_IN_DIR`). Chained-arm routing carries accumulated drift
and is NOT the adjudication surface (measured 2/63 there — consistent with the
receipt's own note that routing parity holds because MoE inputs are
teacher-injected).

| check | expected (frozen) | observed (verifier) | verdict |
|---|---|---|---|
| Router selections | 63/63 exact | **63/63 exact** | PASS — reproduced exactly |
| Router max w-diff | ≤ ~1e-6 | **6.75e-07** | PASS — identical to builder value |
| Layer-0 hidden max | ~≤3e-2 envelope | **0.0272282** | PASS — matches sealed floor |
| outL1 max (TF) | quantified, larger | **0.128071** | PASS — matches receipt 0.128 |
| outL2 max (TF) | quantified, larger | **0.208034** | PASS — matches receipt 0.208 |
| TRUE outL3 max (TF) | never quoted by builder | **2.2084** (mean 0.0344) | NEW SOURCE TRUTH (first gated-MLA layer; ~10× step vs L2 but continuous, mean stays small) |
| MoE sublayer deltas (TF, identical inputs) | — | attn ≤0.014 mean / mlp ≤0.0027 mean at L0–L2 | localization data (§7) |

Raw route records: `g4.route.bin` `a31238c3…` (chained arm),
`g4b.route.bin` `d1b9c882…` (adjudication arm). Near-tie analysis: fixture
router margins were captured by the committed refgen; selection sets matched
63/63 so no near-tie flips occur in the adjudication regime.

## 7. GATE 7 — reference-source discriminator:

```
INT4_QUANTIZATION_NOT_SUFFICIENT = PROVEN
```

Method (verifier-owned, committed under `verify/`):
1. Independent dequantizer (`v_ling3_int4_dequant.py`) implementing the
   compressed-tensors contract directly from checkpoint tensors (LSB-first
   unsigned nibbles q+8 ∈ [0,15], q∈[−8,7], g32 symmetric, BF16 scales,
   `[O,I]` shape restore). NOT the native C dequant path.
2. Row-level validation vs official BF16 rows: cos 0.99479–0.99544,
   maxrel ≈ 0.068–0.072 across 6 sampled matrices — matches the banked
   "cosine 0.9953 = g32 noise" figure. The dequantizer is faithful.
3. Reference B generator (`v_ling3_refgen_int4deq.py`): official modeling code
   with routed experts REPLACED by independently dequantized official INT4
   values; everything else official BF16 (exactly what the engine consumes).
   Deterministic across runs (byte-identical hidden/logits/router).
4. Three-way decomposition (teacher-forced identical inputs, first 4 layers):
   stage deltas eng-vs-BF16ref 0.128/0.208 (L1/L2), eng-vs-INT4dq-ref
   0.138/0.191, **BF16ref-vs-INT4dqref only 0.014/0.028**.
5. Sublayer localization: with IDENTICAL injected inputs, attention-path
   deltas grow 0.034→0.138→0.217→1.03 (L0→L3) while MoE deltas stay
   ≤0.017 max at L0–L2 (1.73 max at L3) — divergence concentrates in the
   attention path, not the experts.

Conclusion: replacing experts with their exact INT4-dequantized values does
NOT collapse engine drift; the engine sits roughly equidistant from BOTH
references while the two references nearly agree at shallow depth. Expert
quantization explains only ~10% of the shallow-depth engine-vs-official
distance. The receipt §8 mechanism ("official INT4-g32 expert quantization
amplified downstream") is FALSIFIED as the dominant source; it remains a real
but secondary channel. This does NOT violate the frozen claim boundary (the
candidate never claimed INT4 to be the sole drift source — explicitly
disclaimed).

## 8. GATE 8 — full-depth streamed official reference harness: BUILT + RUN

Verifier-owned harness committed: `verify/v_ling3_streamed_ref.py`.
Design: official modeling code, fla CPU shim, explicit 4D causal mask, fp32;
global tensors (embeddings/final-norm/lm_head) resident once; ONE decoder
layer constructed, loaded from safetensors lazily, executed on the chained
residual stream, and released before the next layer (`gc.collect()` between
layers). Peak RSS ≈ 2.8 GB — the 15 GB snapshot is never materialized whole.
Validation: streamed outL0/outL1/outL2 magnitudes reproduce the monolithic
fixture exactly (0.2967/0.5437/1.4375).

24-LAYER PARITY MAP (engine chained trace vs streamed official BF16 ref;
max abs diff over [21×1536]; argmax agreement over 21 positions):

| L | max | mean | | L | max | mean |
|---|---|---|---|---|---|---|
| 0 | 0.0272 | 0.0013 | | 12 | 1.29 | 0.099 |
| 1 | 0.111 | 0.0040 | | 13 | 1.44 | 0.111 |
| 2 | 0.326 | 0.0067 | | 14 | 2.06 | 0.126 |
| 3 | 0.198 | 0.0114 | | 15 | 2.30 | 0.148 |
| 4 | 0.219 | 0.0185 | | 16 | 3.08 | 0.174 |
| 5 | 0.368 | 0.0267 | | 17 | 3.58 | 0.216 |
| 6 | 0.436 | 0.0368 | | 18 | 4.94 | 0.283 |
| 7 | 0.519 | 0.0459 | | 19 | 6.58 | 0.318 |
| 8 | 0.490 | 0.0533 | | 20 | 7.34 | 0.366 |
| 9 | 0.522 | 0.0620 | | 21 | 24.97 | 0.463 |
| 10 | 0.847 | 0.0740 | | 22 | 10.59 | 0.509 |
| 11 | 1.08 | 0.0865 | | 23 | 107.7 | 0.698 |

post-final-norm max 1.83 (mean 0.224); logits max 3.95 (mean 0.41), greedy
argmax agreement 16/21 vs official.

Reading: growth is CONTINUOUS — no structural cliff anywhere. Mean-relative
error stays sub-1% of residual magnitude; worst single elements at the deep
chaotic tail reach ~19% of `|out|max≈558`. First structural divergence: none —
error enters as ordinary per-layer noise at the sealed floor (L0 0.027) and
amplifies through depth (chaotic tail beyond ~L16 where even the two official
references diverge mutually up to 83 at L23, see `gate8_three_way.json`).
Full-depth three-way map (eng-vs-BF16 / eng-vs-INT4dq / BF16-vs-INT4dq) is
banked in `verify/evidence/gate8_three_way.json`; it confirms the engine is
equidistant from both references at essentially every depth.

This closes `FULL_24_LAYER_OFFICIAL_PARITY`: measured, quantified, banked —
it is NOT claimed as tight-parity beyond the frozen layer-0 envelope.

## 9. GATE 9 — thread policy: PASS

All nine precedence cases re-run independently; resolution line read directly
from engine stderr:

| case | env | resolved prefill/decode |
|---|---|---|
| 1 | none | 16 / 8 (= Dell default, REQUIRED) ✓ |
| 2 | `L3_THREADS=4` | 4 / 4 ✓ generic binds both |
| 3 | `L3_THREADS_PREF=12` | 12 / 8 ✓ specific overrides generic |
| 4 | `L3_THREADS_DEC=6` | 16 / 6 ✓ |
| 5 | `T=4 PREF=12` | 12 / 4 ✓ |
| 6 | `T=4 DEC=6` | 4 / 6 ✓ |
| 7 | `L3_THREADS=0` | rejected, message printed, exit 1 ✓ |
| 8 | `L3_THREADS=-3` | rejected, exit 1 ✓ |
| 9 | `L3_THREADS=abc` | rejected, exit 1 ✓ |

Decode sweep (bench prompt, ngen=64, prefill pinned 16):
DEC=1 → 1.6 tok/s · DEC=4 → 5.9 · **DEC=8 → 10.1** · DEC=16 → 9.6.
8 is the measured local knee on this Dell i7-11700 — reproduced; not
universalized. PROVEN.

## 10. GATE 10 — expert kernel discriminators: PASS

- Scalar fallback build (`gcc … -DL3_NO_AVX2`, SHA256 `9e9bdb78…`) compiles
  clean and runs.
- Determinism: scalar and AVX2 arms produce IDENTICAL generated token streams
  on the bench prompt.
- `l3_tracecmp` AVX2-vs-scalar over full bench traces:
  **GLOBAL max|diff| = 0.000427246 @ float 1131842 (of 9,756,672)** — exactly
  the banked discriminator result (fp32 reduction-order noise; no structural
  divergence). Traces hashed: `g10/avx2.trace e6ee8cde…`,
  `g10/scalar.trace ac270801…`.
- Throughput gain reproduced: scalar decode 8.3 tok/s / experts phase 8.00 s
  vs AVX2 10.1 tok/s / experts 2.08 s (experts 3.85× faster; decode +22%,
  within reasonable variance of the frozen +26%).

## 11. GATE 11 — frozen Dell performance: PASS

Protocol: frozen artifacts, deterministic 256-id prompt, greedy, ngen=64,
default policy, warmup + 3 recorded reps (`L3_PHASES=1`):

| rep | prefill tok/s | TTFT s | decode tok/s | RSS GB | experts s |
|---|---|---|---|---|---|
| 1 | 28.4 | 1.124 | 10.1 | 8.07 | 2.091 |
| 2 | 28.0 | 1.136 | 10.0 | 8.07 | 2.141 |
| 3 | 19.8 ⚠ | 1.131 | 9.7 | 8.07 | 3.771 |

⚠ rep3 hit a transient prefill slowdown (attn 11.2 s vs 9.3 s; single event,
not reproducible on demand) — classified as node contention/noise, logged
honestly rather than discarded silently.

Neighborhood check vs frozen approximations: prefill ~28.3 ✓ (28.0–28.4),
TTFT ~1.12 ✓ (1.124–1.136), decode ~10.1 ✓ (median 10.0–10.1), RSS ~8.07 ✓
(exact). Phase decomposition medians: attn 9.39 s · norms 0.027 · router
0.137 · experts 2.09–2.14 · shared 1.22–1.23 · head 2.08 · accounted
~14.9 s — matches receipt §4 within noise. No regression.

## 12. GATE 12 — zero post-init model-weight reads: PASS (strongest form)

Independent reproduction: engine launched under verifier control; strace
attached AFTER `init done` (`strace -p <pid> -f -e trace=pread64,read`);
run covered prefill + 96-token decode to process exit.

Result: `syscalls.txt` contains ZERO read/pread64 lines — only thread-exit
markers (16 threads, exit 0). Zero file reads of ANY kind post-init, which is
the strongest userspace-provable form.

```
POST_INIT_MODEL_WEIGHT_READS = 0   PROVEN
```
Evidence hash: `gate12/syscalls.txt` =
`7e2133cbdc86c63cd2805bcbf878fff57f1914f4491d0448a9d880c2a15c9f56`.

## 13. GATE 13 — residency / memory accounting: PASS

- Steady VmRSS flat at 8,004,652 kB (8.07 GB decimal) across all samples
  during prefill + 96-token decode; peakRSS == steady RSS (no growth).
- Resident weight bytes accounted by loader: 8.13 GB on a host with 14 GiB
  usable RAM; no swap/OOM instability (RSS flat, run exits 0).
- Health inferred from actual memory telemetry, not merely process exit.

## 14. GATE 14 — generation stability: PASS with classification

| prompt class | behavior |
|---|---|
| Frozen synthetic bench IDs (`1000+(i·7919)%150000`) | degenerate repetition (token 27059 loops) — CONFIRMS builder observation |
| Real chat prompt ("Explain in two sentences why the sky is blue.") | coherent, on-topic English (Rayleigh scattering), reasoning-style lead-in; 60 distinct 4-grams of 61, max repeat 2× |
| Random-ID prompt (20 random ids) | coherent meta-commentary about garbled input; 61/61 distinct 4-grams |

Classification: quality is NOT broken; pathological repetition is specific to
synthetic random-ID prompts, exactly as the builder reported. No separate
quality finding required for real chat prompts.

## 15. GATE 15 — cross-host regression: PASS

- `make -C c test-c` exit 0 on Dell gcc 15.2.0 AND Mac clang 21.0.0
  (full suite incl. st dtype fork negatives, uring Linux gates, e8/rans
  oracles).
- Sibling engines rebuilt on Dell: `kimi_k3`, `olmoe`, `inkling`, `colibri`
  all OK; `ling3`, `ling3_check`, `l3_tracecmp` built on BOTH hosts.
- Any shared `st.h` regression would have surfaced here: none.

## 16. Corrections issued by this verification (new source truth)

| # | correction | class |
|---|---|---|
| C1 | Closeout §8 OBSERVED tail numbers ("post-final-norm 0.567; logits max 1.05 mean 0.131, argmax 18/21") could not be reproduced from ANY frozen artifact — including the builder's OWN banked dumps (`run_tf.logits` vs banked fixture gives max 18.50, argmax 0/21). Provenance unrecoverable; superseded by §8 numbers above. | FALSIFIED (as stated) |
| C2 | Drift-mechanism attribution "official INT4 expert quantization amplified downstream" as the dominant source: falsified by the Gate-7 discriminator (`INT4_QUANTIZATION_NOT_SUFFICIENT = PROVEN`). Quantization is a real secondary channel (~10% of shallow-depth distance; dominant only inside the chaotic deep tail where both references diverge mutually). | FALSIFIED (as inference) |
| C3 | True outL3 teacher-forced delta never previously quoted: now banked at max 2.2084 / mean 0.0344 (first gated-MLA layer). Continuous growth, no structural break. | NEW MEASUREMENT |
| C4 | Full-depth 24-layer parity map (§8): previously UNMEASURED, now measured/banked. Greedy argmax agreement vs official BF16 at logits level: 16/21. | NEW MEASUREMENT |

None of these corrections touches the frozen claim boundary: the candidate
explicitly disclaimed full-depth parity and INT4-as-sole-source, and the
corrected numbers were labeled OBSERVED (not PROVEN) in the closeout.

## 17. Evidence artifact hashes (node-side, `/data/ANVIL/ling3/verify/`)

```
gate3_run1/hidden.f32        ed356aafac74b39d738ce15f708d1538f3d0760fbbec6823bd59943c26bfea7e
gate3_run1/logits.f32        24514ef0fa8241314115fe76a9e85bfb484d8df3b23ef72e952d5bfd6f56607c
gate3_run1/router.json       511d2b55417dad0e2850eb15c4a28354d5c8d01de82d2070a46631ca08396c69
g4.route.bin                 a31238c3c89fd90075f38a47630410ae180bcf0479484fe5d80a68701ecc3de9
g4b.route.bin                d1b9c882059cd55979cc72f7216cb8b5405d08be6adc14ed94fc3fecd413d913
g4_chain/trace.f32           3bf94523e7dcad649f40c6fd1cd47ad8554b36341a39adf1b01c1cb0fccb1155
g4_chain/logits.f32          00fa52028ab5616a22f30bea9529dfc5a28bd6d49a3ea9c4543416b8f3dbe8ec
g10/avx2.trace               e6ee8cde7677f9e65f27de26d21417795a08b3455c7d9b32d08ab28c0e9d6308
g10/scalar.trace             ac2708014bef28bb58381342267eaa7a5425c330e7aa484fc54e6f9b50ac4416
gate12/syscalls.txt          7e2133cbdc86c63cd2805bcbf878fff57f1914f4491d0448a9d880c2a15c9f56
gate8_streamed/ref.npz       bc91044c0aa2f5d580f5b7eadb158f93c6e94df052e3c60bbba562eb1839dcac
gate8_int4_streamed/ref.npz  1e65d423e39d094f20c854e26e026a65798b028215e7ded5f63c6b742a1c9d67
gate8_parity_map.json        44bf9d3ca50f4400b9fcba622c20ab49adfd853a023743fa0417795648bf16c2
c/ling3                      15bcd6fa2e4e6b6bc658ccf63e047830e7c56f4dfefcba79e32584b50c767f10
c/ling3_scalar               9e9bdb7805aec65e6b41288a00f872c200a1404c8ad3ac283957b056a8b991b5
```

Bulk evidence (traces, fixtures, npz maps, logs) remains node-side on the
Dell 4 TB per the no-local-pull law; this receipt banks hashes + numbers.

## 18. Verifier-only files added (this branch)

```
verify/v_ling3_hookdump.py         f-c reference sublayer hooks (attn/mlp/post-final-norm)
verify/v_ling3_stagecmp.py         per-stage/sublayer comparator (JSON summary)
verify/v_ling3_int4_dequant.py     independent compressed-tensors int4-g32 dequantizer
verify/v_ling3_refgen_int4deq.py   Reference-B fixture generator (INT4-dequant experts)
verify/v_ling3_streamed_ref.py     streamed full-depth 24-layer reference harness
verify/evidence/gate8_parity_map.json
verify/evidence/gate8_three_way.json
docs/LING3_TINY_WAVE1_INDEPENDENT_VERIFICATION_RECEIPT.md  (this file)
```

No candidate runtime source (`c/ling3.c`, shared production headers) modified.

## 19. Claim-boundary scorecard

| frozen claim | verdict |
|---|---|
| native Ling 3.0 Tiny engine exists | VERIFIED |
| official INT4 routed experts consumed directly | VERIFIED (loader enforces exact ST_I32/ST_BF16; census+geometry proven) |
| useful model state fully resident on Dell | VERIFIED (8.13 GB resident, RSS flat 8.07 GB) |
| zero post-init model-weight reads during decode | VERIFIED (strongest form: zero reads of any kind) |
| router parity 63/63 on frozen fixture | VERIFIED (max w-diff 6.75e-07) |
| layer-0 TF drift within ~3e-2 envelope | VERIFIED (0.0272282) |
| deeper-stage drift larger and quantified | VERIFIED + EXTENDED (L1 0.128 / L2 0.208 / L3 2.21; full-depth map banked) |
| ultra-tight historical numbers FALSIFIED | CONFIRMED FALSIFIED (could not reproduce either) |
| thread policy prefill=16/decode=8 on Dell defaults | VERIFIED (+ full matrix, knee at 8) |
| AVX2 kernel improves decode vs scalar | VERIFIED (10.1 vs 8.3 tok/s; experts 3.85×) |
| perf ~28.3 / ~1.12 / ~10.1 / ~8.07 | VERIFIED (within noise; one contention rep logged) |
| st.h fail-closed | VERIFIED (both hosts, fork-based negatives) |
| tracked generated Ling binaries removed @6b0dd8f | VERIFIED (both hosts + bit-identical rebuild) |

Not claimed by candidate (confirmed still NOT claimed): full-depth tight
parity; INT4 as sole drift source; old ultra-tight numbers; zero-allocation
decode; LM-head INT8; Exact Wave 2.

## 20. Canonical final status line

```
LING3_TINY_COLIBRI_WAVE1_INDEPENDENTLY_VERIFIED (router 63/63@6.75e-07 · L0 0.0272 · full-depth map banked · zero-read proof reproduced · perf/thread/kernel/residency/hygiene all pass · corrections C1–C4 banked)
```
