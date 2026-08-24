# Qwen3.6 / Colibrì Campaign Ledger

> Durable source-of-truth record for the `Qwen3.6-35B-A3B` Colibrì optimization campaign.
>
> This document exists so future agents do **not** reconstruct campaign state from chat memory, stale summaries, or isolated benchmark snippets.
>
> **Preservation rule:** historical measurements are append-only. A later result may mark an earlier result `SUPERSEDED`, `FALSIFIED`, or `NONCOMPARABLE`, but must not delete it.

## Snapshot metadata

- Repository: `naytewilson/colibri`
- Live campaign branch at capture: `feat/qwen36-packed-int4-cpu-residency`
- Live branch HEAD at capture: `3d55e925e89c42c60430ea41360a152011de6f1e`
- Capture date: `2026-08-22` (America/New_York)
- Snapshot type: `COMPOSITE`
- Current mutable state must be refreshed before execution; immutable commits/hashes below remain durable historical evidence.

## Evidence classes used here

- `GITHUB_PROVEN` — verified directly against current GitHub source/commit state.
- `RECEIPT_SUPPORTED` — recorded in durable experiment/agent receipts supplied during the campaign; not independently rerun by this document author.
- `HISTORICAL` — accepted historical campaign context; use only with its stated protocol.
- `SUPERSEDED` — historically useful but no longer describes the current implementation.
- `CONTRADICTORY` — evidence conflict is preserved rather than silently reconciled.
- `UNKNOWN` — insufficient current proof.

---

# 1. Locked model/runtime identity

## Model

`Qwen3.6-35B-A3B`

Accepted campaign geometry:

- 40 layers
- hidden size 2048
- vocabulary 248,320
- 256 routed experts + 1 shared expert per layer
- top-k = 8 routed experts/token
- 10 attention layers + 30 Gated DeltaNet layers
- ~34.66B total parameters
- ~3.455B active parameters
- routed intermediate size 512
- DeltaNet vheads 32
- DeltaNet kheads 16
- DeltaNet k/v dimensions 128
- no MTP tensors in the canonical artifact (`UNKNOWN_ABSENT_IN_TENSORS` / no positive tensor evidence)

These geometry values are historical campaign facts and should be re-derived from the artifact when a new model/container is introduced.

## Runtime

**Colibrì is the locked runtime for this campaign.**

Do not silently substitute llama.cpp or another runtime for performance claims.

---

# 2. Quantization / container facts

## INT3 format

Historical campaign format:

- INT3, group size 64
- 24 packed bytes per 64 weights
- plus f32 scale
- effective storage ~3.5 bits/weight

## Mixed container

Historical mixed container:

`/home/nayte/models/qwen36_mixed_low`

Recorded quant census:

- 1,536 INT4 expert tensors
- 8,704 INT3 expert tensors

BF16 source historically under:

`/data/ANVIL/models/hf/source_shards/`

Paths are historical orientation only; verify local existence before use.

---

# 3. Performance baselines — preserve forever

These numbers are **not interchangeable** unless the protocol/corpus/topology matches. Do not average them across incompatible runs.

## INT4 cap128 baseline

`HISTORICAL`

- throughput: ~2.23–2.28 tok/s
- cache hit rate: ~83.1%
- admission: ~4.6 ms/load
- MoE: ~164–166 ms/token
- step: ~221–224 ms/token
- RSS: ~11.17 GiB

## Clean INT3 cap128

`HISTORICAL`

- throughput: ~3.26–3.33 tok/s
- cache hit rate: ~83.7%
- admission: ~2.68–2.79 ms/load
- MoE: ~97 ms/token
- step: ~156–157 ms/token
- RSS: ~8.21 / 9.26 GiB (protocol-dependent observations)

## Persistent uniform INT3

`HISTORICAL`

- cold: 3.11 tok/s
- warm turns: 4.46 / 4.98 / 5.58 tok/s
- warm median: 4.98 tok/s
- cache hit rate: ~89.5%
- MoE: 89.83 ms/token
- DeltaNet: 36.07 ms/token
- attention: 8.23 ms/token
- LM head: 12.77 ms/token
- step: 148.3 ms/token
- RSS: ~9.52 GiB

## Uniform INT3 cap208 negative result

`RECEIPT_SUPPORTED`

- throughput: 3.03 tok/s
- cache hit rate: 84.4%
- MoE: 101.59 ms/token
- step: 161.1 ms/token
- RSS: ~10.0 GiB

**Negative knowledge:** simply increasing a uniform per-layer cap did not improve this workload. Do not repeat “bigger uniform cap” as a new idea without changed conditions.

## Mixed low — historical promoted warm anchor

`RECEIPT_SUPPORTED`

Four-turn historical run:

- cold: 3.13 tok/s
- warm: 5.38 / 5.33 / 5.15 tok/s
- warm median: **5.33 tok/s**
- hit rate: ~91.2%
- RSS: ~10.79 GB

**Campaign anchor:** until a strictly comparable protocol supersedes it, `5.33 tok/s warm median` is the accepted headline serving anchor.

Targets:

- 5.6 tok/s = meaningful next win
- 6.0 tok/s = major milestone
- 6.5 tok/s = stretch

---

# 4. Quality evidence — preserve with provenance warning

Historical 443-token / 8-domain quality record:

- INT3: NLL 1.6873 / PPL 5.41
- mixed: NLL 1.6848 / PPL 5.39
- INT4: NLL 1.6616 / PPL 5.27

Interpretation at the time: no material mixed-vs-INT3 quality regression.

## Current provenance warning

`GITHUB_PROVEN` from current `c/tools/run_quality_gate.sh` inspection during this campaign:

The script was still using:

`/home/nayte/prompts/heldout_eval.json`

for all PPL arms.

Earlier C0R work established:

- `heldout_eval.json` = single-sample corpus
- `heldout_multidomain.json` = intended 8-domain corpus

Therefore the exact historical PPL provenance above remains unresolved until the live campaign verifies which corpus actually produced the promoted measurements.

**Rule:** never relabel historical PPLs as multidomain merely because today's script is later fixed. Record new measurements separately with hashes.

---

# 5. Instrumentation / profiling chronology

## Historical acquisition-count contradiction

`SUPERSEDED`

An older trace produced ~109,120 acquisitions over 252 tokens (~433/token) against an expected `40 layers × topk 8 = 320/token`.

This was eventually explained by mixed accounting / prefill contamination and later by duplicate residency work. Current code uses decode-window baselines and explicitly checks expected acquisitions/token.

Do **not** use the old 433/token value as a current optimization opportunity.

## Admission decomposition

Current campaign instrumentation decomposes admission/cache work into at least:

- total demand admission
- weight segment
- scale segment
- slot acquisition
- lock wait
- hit/miss lookup
- LRU/victim selection

Important semantic note:

`expert_admission` is **not** pure NVMe/storage latency. It includes packed weight read plus optional unpack/copy work plus scale read. Do not subtract theoretical storage bandwidth from this timer and call the residual synchronization.

---

# 6. Major causal repair: duplicate residency / in-flight coalescing

## Commit

`GITHUB_PROVEN`

`2abb9cf487fe4eb927f72022d677746672be06a6`

Commit title:

`PILOT_DUPLICATE_RESIDENCY_CAUSAL_REPAIR`

## Mechanism

Per-expert `loading[eid]` registry in each layer cache:

- set under `g_pilot_mx` at reservation
- cleared at publish
- demand sees an already-loading expert and waits/coalesces instead of duplicating the load
- pilot skips an expert already being admitted
- miss accounting moved after the coalescing decision

## Causal A/B

Recorded candidate improvement over control:

- duplicate-resident inserts: **1523 → 0**
- demand loads: **−15.9% / ~−16%**
- demand bytes: ~17.03 GB → 14.30 GB (~−16%)
- pilot bytes: ~6.60 GB → 5.76 GB (~−12.7%)
- demand admission ms/token: ~89.72 → 66.03 (~−26.4% / ~−27%)
- MoE: ~117.41 → 104.68 ms/token (~−10.8% / ~−11%)
- decode hit rate: ~92.47% → 94.48% (~+2.0 percentage points)
- peak RSS: ~11.31 → 10.84 GB (~−0.47 GB)
- demand coalesce waits observed: 1,164
- pilot coalesce skips observed: 203
- census fingerprint remained byte-identical

Four-turn control/candidate speeds recorded in the repair campaign:

- control: 2.58 / 3.60 / 4.17 / 4.81 tok/s
- candidate: 2.62 / 4.08 / 4.59 / 5.15 tok/s

Correct warm-only median (turns 2–4):

- control: 4.17 tok/s
- candidate: 4.59 tok/s
- improvement: ~10.1%

Do not compare those raw values directly to the historical 5.33 warm anchor unless the exact protocol is shown equivalent.

## Promoted artifact identities recorded during campaign

`RECEIPT_SUPPORTED`

- promoted runtime binary SHA-256: `1cd25904379899bec46712ed5c50fe9d410f51f9ec7636917bee95776fe0aac3`
- earlier promoted trace SHA prefix: `129f001ceb13…`
- census fingerprint prefix: `1e91ac53…`
- decision record: `#54506`
- handoff record: `#54507`

---

# 7. Trace / replay evolution

## v3 trace

Commit lineage included `bdfe26b18cc0f84673eb3f42ebdf0e0bbe7f30d8`.

v3 is a **runtime mutation/outcome stream** containing baseline decisions such as HIT / EVICT / INSERT and exact slots.

Correct label:

`TRACE_EVENT_STREAM_SELF_CONSISTENT`

It is useful as an audit oracle. It is **not** by itself a counterfactual policy replay stream.

## First v4 foundation

Commits after `2abb9cf` introduced request-stream and simulator work, including:

- `947fa30`
- `03fd760`

That first foundation was rejected for counterfactual policy work because baseline policy outcomes filtered or controlled parts of the supposed independent stream.

Preserve this as negative knowledge: a replay tool can match the baseline while still being invalid for counterfactual policy evaluation.

## V4R repair

`GITHUB_PROVEN`

Commit:

`00d845d505a02c2c0332205064d2f31fcd266b2b`

Major repairs:

- `C PC` pilot candidate intent emitted before baseline residency/queued/ring gating
- baseline `Q/P/W` outcome events removed from v4 input
- simulator owns queue/dequeue/admission/publish/wait logic under a frozen service schedule
- candidate confidence captured before freeing source buffer (UAF repair)
- `score_rank` separated from deterministic `enqueue_order`
- one-digit sequence parser bug repaired
- `B GEN` boundary mutex ordering repaired

Adversarial fixture commit:

`3d55e925e89c42c60430ea41360a152011de6f1e`

Fixture suite demonstrates the simulator can:

- retain a pilot opportunity that baseline residency would have suppressed
- create a pilot admission for which no baseline publish marker exists
- derive demand MISS without baseline outcome input
- derive later HIT from simulator residency
- consume sequence-numbered `E` records 2..9

## V4R receipt-reported reconstruction quality

`RECEIPT_SUPPORTED`

At `svc_k=3`:

- demand requests/accounted: 109,120 / 109,120
- sim hits: 99,307
- oracle hits: 99,284
- sim misses: 9,813
- oracle misses: 9,836
- demand admissions: 9,813 vs 9,836
- pilot admissions: 4,098 vs 4,076
- demand bytes: ~14.24 GB vs 14.27 GB
- pilot bytes: ~5.82 GB vs 5.80 GB
- residual: 23 / 109,120 demand outcomes (~0.021%)
- unresolved waiters: 0 / 0

Receipt verdict was `COUNTERFACTUAL_REPLAY_FROZEN_SCHEDULE_READY`.

## Independent review status at this capture

**NOT YET ACCEPTED FOR POLICY TOURNAMENT.**

The current simulator still has a source-level issue: candidate paths that `continue` on resident/already-queued outcomes can bypass the service-event counter increment, allowing cache-policy state to alter worker service-opportunity timing. That violates a truly exogenous frozen schedule.

A narrow V4R.1 repair was requested:

- schedule ordinal must advance from input intent order only, never simulator outcome
- oracle comparison must become a real executable pass/fail gate
- `C PC` ordering claim must match actual mutex semantics
- final resident mismatch must be described as resident-set mismatch, not slot-order mismatch
- counterfactual policy results must be checked across a bounded cadence envelope, not assigned a universal ±0.03% error from one baseline run

**Current next live-Qwen package:** `COUNTERFACTUAL_REPLAY_V4R.1`.

---

# 8. Candidate optimization archive

These are deliberately preserved even when not yet executable.

## A. Layer-adaptive byte budgets

Status: `CANDIDATE / BANKED UNTIL REPLAY_READY`

Rationale:

- uniform cap increase to 208 failed
- per-layer working sets may differ substantially
- evaluate under the **same total byte budget** as baseline

Tournament order once replay is valid:

1. current uniform LRU
2. pinned-hot
3. layer-adaptive budget
4. combined policies

Earlier planning gate: promote to live A/B only if replay predicts approximately ≥20% fewer demand misses **or** ≥10 ms/token admission reduction, subject to replay-model validity.

## B. Top-K batch acquisition

Status: `CANDIDATE / DISCRIMINATE FIRST`

Current `moe()` knows all top-k expert IDs and calls `expert_get()` sequentially.

Potential mechanism:

- batch residency lookup/reservation/coalescing
- reduce repeated lock/lookup/victim bookkeeping
- potentially improve admission scheduling

Do **not** assume “8 mutex calls → 1” translates directly into speed. First read current lock/lookup/victim vs weight/scale timing. If bookkeeping is tiny, the idea is low value.

## C. Adjacent expert overlap analysis

Status: `BANKED / OFFLINE-ELIGIBLE`

Use token/layer/eid traces to quantify adjacent-token expert-set overlap.

Useful for:

- cache retention
- pilot depth
- hotness
- prefetch design

Do not relabel this as MTP/speculative decoding; the artifact has no MTP tensors.

## D. Wall-clock policy model

Status: `HYPOTHESIS / LATER`

A future simulator may map admissions to measured latency classes, but current replay is screening-grade and must not make wall-clock tok/s predictions until the timing model is independently validated.

---

# 9. Negative knowledge — do not rediscover expensively

1. **Uniform cap208 alone did not help.**
2. **The historical ~433 acquisitions/token discrepancy is superseded.** Do not claim a 0–19 ms/token opportunity from it.
3. **`expert_admission` is not raw disk latency.** The old theoretical-bandwidth subtraction that implied a path to ~10 tok/s is invalid arithmetic.
4. **v3 mutation replay is not counterfactual replay.**
5. **A baseline-matching replay can still be policy-dependent.** Policy independence must be structurally proven, not inferred from a good oracle match.
6. **No MTP tensors are present in the canonical Qwen artifact.** Do not build an MTP campaign from adjacent-expert overlap alone.
7. **Historical quality provenance is not sealed.** Do not claim the old PPL figures are definitely from the multidomain corpus until hashes/provenance are resolved.

---

# 10. Durable preservation protocol for every future Qwen3.6 experiment

Every promoted, rejected, or decision-relevant experiment should append a record containing:

## Identity

- repository
- branch
- commit SHA
- runtime binary SHA-256
- model/container identity SHA-256 or deterministic tree digest
- quant/container variant
- exact host/topology

## Workload

- corpus/prompt-set name
- corpus SHA-256
- prompt count
- token count
- prefill/decode distinction
- cold/warm protocol
- cache cap/byte budget
- pilot/pinning/policy settings
- OMP/thread settings

## Performance

- tok/s per turn, not only mean
- cold/warm median as explicitly defined
- MoE ms/token
- admission ms/token
- weight/scale/lock/lookup/victim timing where available
- hit/miss counts and rates
- admissions
- bytes admitted
- evictions
- RSS / peak RSS

## Correctness / quality

- token/parity checks if applicable
- quality metric
- corpus hash
- quality protocol
- any semantic-risk transformation

## Provenance

- artifact hashes
- receipt path
- decision/ledger IDs
- whether result is `PROMOTED`, `REJECTED`, `SUPERSEDED`, `CONTRADICTORY`, or `SCREENING_ONLY`

## Rule

Never overwrite the historical number. Append the new result and explicitly state comparability.

---

# 11. Current campaign scoreboard at capture

| Lane | State |
|---|---|
| Qwen3.6 runtime correctness | STRONG / active |
| Duplicate-admission repair | PROMOTED |
| Current headline warm anchor | 5.33 tok/s historical comparable anchor |
| Cache/admission instrumentation | STRONG |
| Replay v3 audit fixture | VALID FOR SELF-CONSISTENCY / ORACLE |
| Counterfactual V4R | REPAIRING |
| V4R.1 exogenous-clock gate | NEXT |
| Policy tournament | LOCKED UNTIL V4R.1 ACCEPTED |
| Layer-adaptive/pinned-hot screening | BANKED |
| Top-K batch acquisition | BANKED, telemetry-gated |
| Quality corpus provenance | UNRESOLVED / blocks sealing future promotion quality claims |
| Runtime switch away from Colibrì | FORBIDDEN WITHOUT EXPLICIT CAMPAIGN CHANGE |

---

# 12. Refresh triggers

Before acting on this ledger, refresh only the mutable facts that can change the route:

- current branch HEAD
- dirty/staged worktree state
- active owner/agent of `qwen36.c`
- whether V4R.1 has landed and passed independent review
- current node availability/load if a live run is contemplated
- current quality-corpus provenance/seal state

Immutable commits, historical measurements, and artifact hashes above do not become false merely because they are old; they become `SUPERSEDED` only when a newer comparable result is explicitly recorded.

---

# 13. Canonical use by future agents

A future agent should read this file **before** proposing Qwen3.6 optimization work, then inspect current GitHub/source truth.

Use this file to answer:

- what was already tried?
- what actually improved runtime?
- which measurements are comparable?
- which ideas are banked rather than executable?
- which contradictions were already resolved?
- what must not be resurrected?

Use current source to answer:

- what is true now?
- what is dirty/owned?
- which package is currently executable?

**Conversation memory is never a substitute for either.**
