# OLMoE FORGE REAL E2E GATE — 2026-08-24 (Forge F1 Lane B)

**Lane**: Forge F1 Lane B (independent evidence-builder; Lane A untouched)
**Verdict**: `OLMOE_FORGE_REAL_E2E_GATE_PROVEN`
**Production mutation**: NONE — no protected surface touched; all work in isolated node clones + this evidence branch.

## 1. Startup repo/refs (reverified live at session start)

| Ref | Expected | Observed | Verdict |
|---|---|---|---|
| remote | naytewilson/colibri | https://github.com/naytewilson/colibri.git | PASS |
| `forge/f1-ooc-moe-runtime` | `486f557dc8e1c4bfb10ff7382dcb7c10a7602cd7` | exact (local + origin) | PASS |
| `verify/forge-f1-20260824` | `5689298b89eb6bfe5d517262e6fd0088b59c6882` | exact (`origin/…`, commit present) | PASS |
| pre-Forge baseline | `f04359aab31a388cc47d36e96c3ea36400061cb6` | ancestor of candidate | PASS |

## 2. Canonical model identity (Mission A)

| Item | Value | Class |
|---|---|---|
| Family | AI2 OLMoE-1B-7B MoE (Llama-style + QK-norm, 16L × 64 experts top-8) | PROVEN |
| Oracle-binding upstream ID | **`allenai/OLMoE-1B-7B-0924`** (base) | PROVEN — deleted `validate_ref.py` (visible at `2000874^`) generated `ref.json` from it via greedy transformers BF16; deleted `run.py` loaded `models--allenai--OLMoE-1B-7B-0924`; `olmoe.c:1-6` declares ref.json as its Stage-A validation target |
| Serving-converter example ID | `allenai/OLMoE-1B-7B-0125-Instruct` | OBSERVED — docstring example only (`convert_olmoe_merged.py:17`), not oracle-bound |
| Upstream revision pin in source truth | none recorded anywhere | UNKNOWN → frozen at acquisition (below) |
| Existing artifact in any storage | none (verifier sweep + node `/data` + Mac volumes rechecked) | PROVEN (absence) |

Kill condition A NOT triggered.

## 3. Acquisition + conversion (Mission B)

- Upstream snapshot frozen at HF revision **`6d84c48581ece794365f2b8e9cfb043c68ade9c5`** (repo main SHA at 2026-08-24, verified via HF API before download).
- Download method: `convert_olmoe_merged.py --repo allenai/OLMoE-1B-7B-0924` streaming path, with `hf_hub_download` monkeypatched to inject the frozen revision (wrapper `run_convert_pinned.py`; repo converter code byte-identical).
- Converter: blob **`36ac01d1f68a2f1914fe3983525a53035a3a659d`** at candidate ref; file sha256 **`3e4ab4b4dd9b2a0a21925d2ce30945e059263a427773b7be17ff962de2a9c21a`**. Candidate-vs-main converter diff = crash-safety only (atomic tmp+rename, paired expert publication); quantization math identical.
- Converter command: `python3 run_convert_pinned.py` (venv `/data/ANVIL/venvs/hf312`, torch 2.13.0+cpu, hf_hub 1.28.0) on anvil-node-02.
- Conversion log tail: `[3/3] … 0 experts still incomplete … DONE.`

### Container hashes

Path: `anvil-node-02:/data/ANVIL/models/olmoe_gate_20260824/container/` (7.0 GB)

```
config.json              3643aa880d2f1c9b418156269ae791c73e5612d6b6b6fde0724d927cf89b6335
tokenizer.json           a094266ac6c4982efba277bc251349a5a6d6ad37efb39a2a90f53d8be2a40a40
tokenizer_config.json    78a839c7851f14f9fb30e664c2b46166dc0628f2900679e5ec160656f702edff
special_tokens_map.json  b77491e270c6fcc5b2ecf22370f7318a6a18d3cabea09ba7bab92e9bf12656c2
generation_config.json   d77272ffaa7e62a904e8e130bb25ab11585bd4a5026e388d6d682e4b82892ce2
model-00000.safetensors  063531ee6b951d4cb68db637728c03916dc8565404245cc03f9504b1cb5ab2a6
model-00001.safetensors  037da9604fdda14bacd8291483cb9c85b03c6afd7f42a83a9dc86cda29109042
model-00002.safetensors  67e170c07601bb88075b82825253d1785395e61c8683b96ee930a6bdcac971c1
model-00003.safetensors  52fbd28eaa1ec8bbc9f27876d5c0eb1e1ea77a5c879a0caccca8d674f92e9fcf
model-00004.safetensors  0e931099ea8c2bb0a83a4b69a4e8213398d20c8a363afa020f65179d7e9ae12e
manifest sha256          250c563439660ac48904507c82e7844efc1d278dc74428a0a85e0c5c76070736
ref.json (oracle copy)   745d8e6d0ae2bd902f3c6e1514e53a1e7c6c6405ab37c293706b44ab0f11000f
```

## 4. Container contract census (Mission C) — fail-closed, ALL PASS

Probe: `olmoe_census.py` (exit≠0 on any failure). Machine-readable: `census.json` sha256 `f2a7171540e0553e088f9d952948b4230edad6368f02fd0250c9650d1f41d0c8`.

- config geometry: hidden=2048, L=16, heads=16, kv_heads=16, experts=64, topk=8, inter=1024, vocab=50304, model_type=olmoe — PASS ×9
- tokenizer.json present — PASS
- tensors exactly 2195 (= 3 global dense + 16×9 per-layer dense + 16×64×2 expert), zero missing, zero extra — PASS
- every expert: `merged_weight` I8 nbytes=6291456 (=3·1024·2048), `qs` F32 shape (4096,) — PASS ×2048
- dense shapes/dtypes spot set (embed/lm_head/norm/q_proj/gate BF16) — PASS
- no qwen/deepseek/glm/kimi/mtp name leakage — PASS

## 5. Baseline identity + determinism contract (Mission D)

- Baseline source: **pre-Forge Colibri ref `f04359a`** (verifier's own frozen rollback anchor; preference 1 satisfied).
- Baseline binary: gcc (Ubuntu 15.2.0) `-O3 -march=native -fopenmp -pthread`, sha256 **`a8a051e027fae2998768b2c6d221b5c3fc62cf190f07304ef9485c3a56523687`**
- Forge binary: same compiler+flags, candidate `486f557`, sha256 **`96ffe8cc9ead71e3c86a3c4e6bdc2b0641938143d664c85f1fe704956189ede6`** (built fresh from isolated clone — no builder-binary reuse)
- Host: anvil-node-02, i7-11700 (8P/16E), omp_tune physical-core sizing (omp_tune.h identical both refs → threads=8 both sides)
- Invocation (both engines): `SNAP=<container> ./olmoe 16 8 ref.json` (+ arms below)
- Env policy: defaults-off (no knobs set)

### Predeclared adjudication contract (declared BEFORE forge execution)

1. PRIMARY — baseline-vs-forge differential: stdout byte-identical after stripping timing/RSS telemetry lines, all arms.
2. SELF-DETERMINISM — 3 repeats per engine token-identical.
3. ORACLE SANITY (directional, NOT equality): ref.json was produced by transformers-BF16 without expert quantization; engine computes int8-expert math (repo's own #108 note: int8 ppl 12.11 vs bf16 12.25). Exact 12/12 is not a property of this container class. Report match count + first-divergence index; require forge ≥ baseline and identical divergence position.
4. STRESS — cap=2 forced-eviction arm must preserve tokens.
5. NUMERIC ANCHOR — PPL=1 TF-NLL equal between engines.

## 6. Results (Missions E/F/G)

Generated-token line (identical across ALL 8 generation arms, both engines):

```
Reference: 7785 15 187 187 510 3565 3448 273 6181 310 5112 15
C engine : 7785 15 187 187 510 5347 273 253 1986 2077 310 5041
Matching tokens: 5/12   (first divergence at generated index 5)
```

| Check | Expected | Observed | Verdict |
|---|---|---|---|
| Baseline runs rc | 0 | 0,0,0 | PASS |
| Baseline self-determinism | 3× identical tokens | 3× `7785 15 187 187 510 5347 273 253 1986 2077 310 5041` | PASS |
| Forge runs rc | 0 | 0,0,0 | PASS |
| Forge self-determinism | 3× identical tokens | identical to baseline | PASS |
| Differential run1/2/3 | byte-identical (stripped) | BYTE-IDENTICAL ×3 | PASS |
| cap=2 eviction stress | tokens preserved | identical tokens, 5/12 | PASS |
| PPL arm TF-NLL | engines equal | `0.6573 nats/token · ppl 1.93` BOTH | PASS |
| Oracle sanity | forge ≥ baseline, same divergence idx | equal (5/12, idx 5) | PASS |
| Crash/hang/corruption | none | none; clean exits, no lease reports | PASS |
| Peak RSS | bounded | 3.29 GB both engines, stable | PASS |

Correctness comparison class: **token-identical differential + exact numeric-anchor equality** (strongest honest gate given quantized-container vs BF16-oracle regime).

Earliest divergence vs ref.json: generated index 5 (`3565`→`5347`) — SAME position and SAME tokens on both engines ⇒ attributable to the shared int8-vs-BF16 numerical regime, NOT to the Forge change.

## 7. Shared-substrate proof (Mission H)

Real Forge run under `strace -f -c` (`forge.run1.strace`):

- Expert I/O dominated by **2493 `pread64` calls** (95.9% of syscall time); zero expert-tensor mmap reads — experts flow exclusively through the generic pread backend opened by `coli_expert_backend_pread_open` (`olmoe.c@486f557` adapter: n_layers/n_experts/slots_per_layer/expert_numel declared, tensor-name templates `model.layers.%d.mlp.experts.%d.merged_weight|.qs`).
- Engine cache counters identical to baseline: hit=887 miss=1161 (43.3%) — same logical request stream through the new store path.
- Lease/accounting balance: **zero** `pbs_destroy` lease/reservation leak reports at close on all runs (reports are active in this build — `#ifndef NDEBUG` ungated by Makefile).
- Old duplicate OLMoE cache machinery: absent at candidate ref (verified independently by verifier; reconfirmed — `loading[]`/private `LCache` slots gone from olmoe.c).
- `would_evict` exercised read-only (source-verified; not triggered by default arms).
- Coalescing/eviction internals: covered by candidate's own probe suites (verifier-reproduced: winners=1/coalesced=7/physical_loads Δ=1); my cap=2 arm exercised real eviction pressure E2E with preserved output.
- Defaults-off: both gates ran with zero env knobs — parity holds.

## 8. Performance sanity (Mission I) — same-window, 12-token micro-runs

| Arm | tok/s | Note |
|---|---|---|
| baseline run1 / 2 / 3 | 0.10 / 0.22 / 4.94 | cold→warm page-cache progression |
| forge run1 / 2 / 3 | strace-instrumented / 4.20 / 1.10 | run3 contention-noisy |

OBSERVED only: warm-window medians overlap (~4–5 tok/s both sides); N=3 micro-runs with page-cache state confounds — **no performance claim made or implied**. Physical reads ≈ 2493 preads/run (forge); RSS 3.29 GB both.

## 9. Artifacts & rerun

Raw evidence pulled to Mac: `tmp/olmoe_gate_20260824_raw/` (runs_baseline/, runs_forge/, census.json, container_hashes.txt, container_manifest_hash.txt, convert.log, gate_artifact_hashes.txt).

Rerun command (anvil-node-02):

```bash
cd /data/ANVIL/models/olmoe_gate_20260824
bash olmoe_gate.sh <path-to-olmoe-binary> <tag> runs_<tag> 16 [strace:0|1]
# oracle copy ref.json sha256 745d8e6d0ae2bd902f3c6e1514e53a1e7c6c6405ab37c293706b44ab0f11000f
```

## 10. Handoff statement (Lane A dependency)

Candidate diff f04359a→486f557 touches generic store/admission surfaces (`expert_backend_pread.c`, `admission.c/.h`, `expert_store*`, `expert_telemetry.h`) that OLMoE consumes. If Lane A lands further changes to those surfaces:

- changes confined to admission/acquire_batch paths that shipped engines do not call (per verifier's accounting note) → targeted smoke sufficient (this gate's run1 only);
- changes to `expert_backend_pread.c` reserve/publish/pbs victim logic → FULL gate rerun required (all arms);
- receipt-level claims here are independent of Lane A's edits either way — they were produced against `486f557` as-is.

## Claim ledger

- PROVEN: refs; identity chain; acquisition/conversion; census; binary builds; determinism; differential byte-parity (5 arms); numeric-anchor equality; substrate pread usage; lease balance; absence of prior artifact; production non-mutation.
- OBSERVED: perf micro-runs; hit-rate/RSS telemetry; strace census counts.
- INFERRED: 5/12-vs-ref divergence cause = int8-vs-BF16 regime (supported by #108 historical cross-check; not separately re-proven).
- FALSIFIED: "no canonical OLMoE model identity recoverable" (identity recovered); "OLMoE real E2E impossible" (executed).
- UNKNOWN/BLOCKED: historical upstream revision actually used for original ref.json (pre-acquisition pin never recorded).

---

# RERUN ON LANE-A LIVENESS REPAIR — 2026-08-24 (Lane B2)

**Verdict**: `OLMOE_FORGE_REAL_E2E_GATE_REPROVEN_ON_LIVENESS_REPAIR`

## Why full rerun was mandatory

Lane A moved the candidate `486f557…` → **`0b789e06eb6448cd16bf904b23cc66e565f44295`** (5 commits, 13 files), touching exactly the surfaces this receipt's handoff statement flagged for full rerun: `c/expert_backend_pread.c` (F1-LIVE-1 closure: stage-3 victim spin removed → non-blocking claim, new transient `COLI_EXPERT_ERR_SATURATED`, engines poll unlocked with residency recheck), `admission.{c,h}`, `expert_store.h`, and `c/olmoe.c` lock-order rework (`g_xorder_mx` never held across `reserve()`; BUSY/SATURATED retry outside the mutex). Diff scope audited: liveness/lease closure only — no unrelated drift (closure doc + 421-line `tests/test_pread_liveness.c` added in-tree).

## Model packet (Mission B) — intact, not reconverted

All 10 container files rehashed OK on node; manifest sha256 recomputed = `250c563439660ac48904507c82e7844efc1d278dc74428a0a85e0c5c76070736` — exact. Census probe re-run: 15/15 PASS (2195 tensors exact).

## Binaries (Mission C) — rebuilt from source, no reuse

| Engine | Source | Binary sha256 |
|---|---|---|
| baseline | f04359a | `a8a051e027fae2998768b2c6d221b5c3fc62cf190f07304ef9485c3a56523687` (**byte-identical to Lane B build → zero toolchain drift**) |
| repaired candidate | 0b789e0 (HEAD verified exact on node clone) | `2b11a2489b3ee1a4e225cc488fa82ae5c9fc887fa8bdc1af48358d8ec070fb98` |

Compiler: gcc Ubuntu 15.2.0 `-O3 -march=native -fopenmp -pthread` (both).

## Full gate rerun (Mission D) — same frozen contract, no weakening

| Check | Expected | Observed | Verdict |
|---|---|---|---|
| baseline repeats ×3 | identical tokens | 3× accepted sequence | PASS |
| repaired-candidate repeats ×3 | identical tokens | 3× `7785 15 187 187 510 5347 273 253 1986 2077 310 5041` | PASS |
| differential run1/2/3/cap2/ppl | byte-identical (stripped) | BYTE-IDENTICAL ×5 arms vs rebuilt baseline | PASS |
| generated sequence | prior accepted | exactly prior accepted | PASS |
| BF16 ref divergence | regime characteristic | unchanged: 5/12, first divergence gen idx 5 | PASS |
| TF-NLL anchor | 0.6573 nats/tok · ppl 1.93 | identical both engines | PASS |
| cap=2 eviction stress | tokens preserved | identical tokens; rep2 hit=10 miss=2038 (real churn through new claim path); RSS 1.98 GB | PASS |
| clean termination / rc | 0 everywhere | 8/8 runs rc=0, no hang | PASS |
| RSS cap16 | bounded | load 1.79 GB, peak 3.29 GB | PASS |
| pbs_destroy lease/reservation reports | zero | zero across all runs | PASS |

## Liveness-specific results (Missions E/F)

- Source audit: reserve() non-blocking post-fix; SATURATED/BUSY retry polls with `g_xorder_mx` RELEASED; residency recheck under lock before every retry; prefetch admission skips on saturation (advisory never contends with demand).
- Runtime discriminator: cap=2 arm forced 2038 misses through constant reserve→publish→evict cycling on the real model via the repaired path — zero deadlock, zero hang, zero SATURATED fatality (all rc=0).
- No duplicate physical loads: strace pread64 count 2493 — identical to pre-repair Lane B run at same cap.
- Accounting: hits/misses identical to baseline (887/1161 @cap16); coalescing covered by in-tree `test_pread_liveness.c`; logical request stream unchanged.

## Same-window performance (Mission G) — OBSERVED only

baseline warm run3 6.31 tok/s (cold 0.31 → 1.98 → 6.31) vs repaired candidate 5.71 / 5.71 / 5.68 tok/s (stable, warm from start of window). Within micro-run noise; correctness and liveness outrank. No performance claim.

## Production mutation check

NONE — candidate branch untouched; docs-only append to this evidence branch, pushed non-force.

Raw evidence: Mac `tmp/olmoe_gate_20260824_raw/b2_rerun/` (runs_b2_rerun/, runs_rep2/, rerun_b2.log).

---

# INDEPENDENT RE-VERIFICATION — 2026-08-24 (Lane B3, fresh operator)

**Verdict**: `OLMOE_FORGE_REAL_E2E_GATE_REPROVEN_INDEPENDENTLY` — every load-bearing claim of Lane B/B2 re-established from source truth by a fresh session with no reliance on prior binaries or prior process state.

## Refs at B3 start (live-fetched)

| Ref | Observed | vs receipts |
|---|---|---|
| `origin/forge/f1-ooc-moe-runtime` | `0b789e06eb6448cd16bf904b23cc66e565f44295` | = Lane B2 tested ref; **no newer Lane A ref exists** |
| `486f557d…` (mission handoff candidate) | present in object store | = Lane B tested ref |
| `verify/forge-f1-20260824` | `5689298b…` | exact match to mission file |

## Artifact integrity — rehashed, not trusted

- Container: all 10 files re-hashed on node against `container_hashes.txt` → 10/10 OK. Manifest hash file sha256 `250c5634…0736` exact.
- Oracle copy: `ref.json` sha256 `745d8e6d0ae2bd902f3c6e1514e53a1e7c6c6405ab37c293706b44ab0f11000f` exact.
- Node clones: baseline_colibri @ `f04359a`, forge_colibri @ `0b789e0`, both clean worktrees.

## Binaries — rebuilt from clean source, zero reuse

Prior binaries deleted (`rm -f olmoe`) and rebuilt in place from verified refs, gcc (Ubuntu 15.2.0) `-O3 -march=native -fopenmp -pthread`:

- baseline `f04359a`: `a8a051e027fae2998768b2c6d221b5c3fc62cf190f07304ef9485c3a56523687` — byte-reproduced
- forge `0b789e0`: `2b11a2489b3ee1a4e225cc488fa82ae5c9fc887fa8bdc1af48358d8ec070fb98` — byte-reproduced

Reproducible-build property itself independently confirmed.

## Frozen-contract gate rerun (same arms, same invocation, no weakening)

`bash olmoe_gate.sh <bin> b3_<engine> b3_runs_<engine> 16 0 1` per engine:

| Check | Expected | Observed | Verdict |
|---|---|---|---|
| baseline ×3 rc | 0 | 0/0/0 | PASS |
| forge ×3 rc | 0 | 0/0/0 | PASS |
| generated sequence | accepted `7785 15 187 187 510 5347 273 253 1986 2077 310 5041` | exact, all 8 gen runs | PASS |
| self-determinism ×3/engine | 1 unique sequence | 1 / 1 | PASS |
| differential run1/2/3/cap2/ppl | byte-identical post-telemetry-strip | BYTE-IDENTICAL ×5 (guarded adjudicator; empty-payload comparisons rejected after first attempt caught a transport-mangled strip regex) | PASS |
| TF-NLL anchor | engines equal | `0.6573 nats/token · ppl = 1.93` both | PASS |
| cap=2 eviction stress | tokens preserved | identical sequence, 5/12 | PASS |

Adjudicator note: first strip attempt silently compared empty payloads (ERE mangled in SSH transport) and would have reported false BYTE-IDENTICAL; v2 rejects zero-line payloads. The guarded result is the one of record.

## Shared-substrate spot-checks (source truth, forge clone @ 0b789e0)

- `olmoe.c:403` opens generic backend (`coli_expert_backend_pread_open`); loads via adapter at :140; evict-notify wired at :434.
- Old local cache machinery: zero occurrences of `loading[` / private `LCache` typedef in olmoe.c.
- `coli_expert_backend_pread_would_evict` (`expert_backend_pread.c:713`): read-only policy preview — locks, inspects, unlocks, mutates nothing, charges no counters; docstring states the contract.
- Sole olmoe.c call site (:904) sits behind opt-in `g_pilot_evict_guard`; defaults-off parity held all B3 runs with zero env knobs.
- Runtime counters identical across engines: hit=887 miss=1161 @cap16 (identical logical request stream through shared store).

## Production mutation check

NONE — candidate branch untouched (still `0b789e0`); docs-only append here; raw text receipts parked at Mac `tmp/olmoe_gate_20260824_raw/b3_indep/`.

## Handoff statement (unchanged in substance)

No newer Lane A ref exists as of this verification; evidence is current for the live Forge head. If Lane A lands further commits: admission/acquire_batch-only changes → targeted smoke sufficient; changes to `expert_backend_pread.c` reserve/publish/victim logic or olmoe lock-order → full gate rerun required (rerun command above).
