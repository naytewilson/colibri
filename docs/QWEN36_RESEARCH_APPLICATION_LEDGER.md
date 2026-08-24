# Qwen3.6 Research Application + Model-Finish Ledger

> Durable controller ledger for deciding **what gets applied, when, and why** in the Qwen3.6-35B-A3B campaign and its successor Qwen lanes.
>
> This file complements `docs/QWEN36_CAMPAIGN_LEDGER.md`. The campaign ledger preserves historical measurements; this ledger controls the forward research/application queue.
>
> **Rule:** research is not "applied" until it has an implementation or artifact, correctness/quality evidence, a comparable node benchmark, and an explicit production/promotion decision.

## 0. User directive / campaign north star

The 35B MoE source model is **not** intended to remain physically oversized forever on a 16 GB node.

The user's preferred long-term Qwen destination is **dense**, not MoE.

That changes the product destination but does **not** invalidate the current MoE campaign. The current Qwen3.6 MoE work is the control specimen and runtime proving ground from which Colibri has learned low-memory serving, quantized execution, residency, I/O, telemetry, promotion, and rollback discipline.

The order is now deliberate:

1. finish the exact current MoE serving/runtime baseline correctly;
2. freeze a trustworthy full-size MoE control;
3. prepare and select the dense-Qwen target from live source/model truth rather than guessing a checkpoint;
4. prefer a **resident-fit dense build** through aggressive but quality-validated low-bit representation;
5. use out-of-core dense weight scheduling only for the remainder that cannot safely stay resident;
6. retune Colibri for dense layer/block residency and sequential prefetch;
7. promote only after quality + throughput + memory + rollback evidence is complete.

MoE structural compression remains valuable as a bounded research/serving lane, but it is no longer allowed to become an indefinite product detour after the full-size control is sealed.

Do not spend indefinite effort polishing the full-size MoE model after the exact-runtime frontier is measured. The full-size artifact remains the control/oracle, not the permanent destination.

---

## 1. Live GitHub source truth — verified 2026-08-23

### Frozen production-source line

Branch:

`feat/qwen36-async-expert-batch`

Verified branch relation:

`f04359aab31a388cc47d36e96c3ea36400061cb6` == current branch tip.

Important evidence at this commit:

- BATCH prefill acquisition is present;
- FUSED/BATCH exact-runtime work is banked;
- prefill batch effective I/O concurrency measured around 7–8 streams;
- QD16 was worse / queue-thrashing and is falsified for that regime;
- remaining prefill residual was localized to compute + routing recomputation;
- decode path remained unchanged by that instrumentation commit.

This branch is the current **full-size MoE control source** unless later source truth explicitly supersedes it.

### Exact-next diagnostic line

Branch:

`feat/qwen36-exact-next`

Verified tip:

`09e5bf13b758fb2e18f1e68679f2ff6a761a2f75`

It is two commits ahead of `f04359a...` and contains the W3a routing-reuse experiment. W3a removes redundant prefill selection work by reusing BATCH pre-pass selections. It is diagnostic/experimental evidence, not automatically production-promoted.

### Historical campaign ledger

Branch:

`docs/qwen36-campaign-ledger-20260822`

Existing durable files include:

- `docs/QWEN36_CAMPAIGN_LEDGER.md`
- `docs/QWEN36_CAMPAIGN_CAPSULE.json`
- `docs/COLIBRI_RUNTIME_LEARNINGS_LEDGER.md`

The campaign ledger preserves the historical 5.33 tok/s warm-median serving anchor and the model/runtime chronology. Current mutable node state must still be re-verified locally.

---

## 2. Critical path — what we do next

### GATE A — Finish the full-size exact MoE control correctly

**Priority: NOW.**

This is a closure wave, not another open-ended optimization campaign.

Required before changing model family/structure:

- verify the exact promoted source/binary/snapshot identities on the node;
- run the current promoted FUSED+BATCH binary under the same frozen warm-serving protocol used for the accepted headline anchor, so current tok/s is apples-to-apples;
- re-establish the quality baseline using a hashed, explicitly named evaluation corpus;
- verify deterministic route/output parity and accounting conservation for the promoted configuration;
- preserve deployment `current` + rollback proof;
- record RAM, expert-pool bytes, cache residency, physical expert misses/token, bytes/token, and storage latency under the same run;
- freeze these results as the full-size control for later MoE and dense comparisons.

**Exit:** `QWEN36_FULLSIZE_CONTROL_FROZEN`

Once this exit is earned, stop spending critical-path effort on tiny exact-runtime tweaks unless a measurement shows a new high-value software wall.

### GATE B — Dense-Qwen target census and resident-fit plan

**Priority: NEXT, immediately after Gate A. This is now the primary destination lane.**

Do a fresh, current census before choosing a checkpoint. Do not assume that the current 35B MoE should be mechanically converted into dense form.

Allowed candidate classes:

1. a native dense Qwen-family checkpoint whose quality/capability matches the user's intended use;
2. a dense student distilled/derived from a stronger Qwen teacher, if source/research evidence supports that route;
3. a smaller native dense Qwen whose low-bit artifact can fit safely in the 16 GB node with useful context and runtime headroom.

For each candidate record:

- exact model/revision/license/source;
- architecture and parameter count;
- context/KV geometry;
- tokenizer/chat-template identity;
- dense weight bytes at source precision;
- predicted bytes at candidate quant formats;
- actual converted artifact bytes when built;
- RAM required for fixed weights, runtime workspaces, KV, OS, and safety headroom;
- whether the preferred operating mode is fully resident or partially out-of-core;
- quality baseline/provenance.

**Preferred strategy:** make the dense model fit as fully resident as practical. A dense model touches essentially every layer every token, so naive per-token SSD weight streaming is a fallback, not the design goal.

**Exit:** `QWEN_DENSE_TARGET_AND_MEMORY_PLAN_FROZEN`

### GATE C — Dense artifact build + correctness/quality baseline

Build the selected dense candidate under a new immutable artifact ID. Do not mutate the MoE control.

Prioritize:

- low-bit dense weight formats with real native kernels;
- activation/outlier-aware precision allocation where evidence warrants it;
- sensitive tensors protected at higher precision when measured quality requires it;
- exact container geometry and bounds checks;
- tokenizer/chat-template parity;
- deterministic reference checks before performance work.

A smaller file is not enough. The runtime must actually consume the low-bit representation without expanding it back into a RAM footprint that defeats the point.

**Exit:** `QWEN_DENSE_ARTIFACT_QUALITY_PROVEN`

### GATE D — Dense runtime execution strategy

First attempt the **resident-fit** path.

If the full useful dense artifact fits within measured safe RAM headroom, keep it resident and optimize compute. Do not invent disk streaming merely because Colibri knows how to stream MoE experts.

If some dense weights must remain out-of-core, use a dense-specific strategy:

- address weights by layer/tensor/block rather than `(layer, expert)`;
- exploit deterministic layer order to prefetch the next block/layer;
- use bounded double-buffer/window residency;
- batch/merge reads into contiguous ranges where the container permits;
- treat OS page cache as a measurable tier;
- distinguish logical block requests from physical disk reads;
- never evict a block while a consumer still holds a lease;
- measure physical bytes/token and prove the requested throughput is compatible with actual storage bandwidth/latency.

Do not use naive whole-layer rereads per generated token if the measured I/O budget makes the target impossible.

**Exit:** `QWEN_DENSE_RUNTIME_BASELINE_PROVEN`

### GATE E — Optional bounded MoE structural-compression lane

MoE structural compression remains useful only when one of these is true:

- it materially improves the currently useful deployed service before dense Qwen is ready;
- it produces reusable quantization/pruning/distillation evidence for the dense lane;
- it exposes a runtime mechanism that should be carried into Colibri.

Candidate research includes:

- progressive expert pruning with re-scoring;
- diversity-aware expert scoring;
- expert merging / partial-preservation merging;
- joint structural pruning + mixed precision;
- layer/expert sensitivity;
- KD recovery for promising lossy candidates.

Keep the full-size MoE control immutable.

This lane must not delay Gate B/C/D without evidence that it unlocks the dense destination.

**Exit if pursued:** `QWEN36_STRUCTURAL_SUCCESSOR_PROVEN`

### GATE F — Final dense production promotion

Required:

- source SHA/revision;
- artifact hashes;
- deterministic correctness evidence;
- quality comparison against an explicit reference/control;
- comparable cold + warm throughput;
- TTFT/prefill/decode decomposition;
- RSS / resident weight bytes / page-cache effect / physical bytes read;
- context/KV memory behavior;
- clean promotion directory;
- atomic current pointer;
- rollback proof;
- smoke run.

**Exit:** `QWEN_DENSE_NODE_FINAL_PROMOTION_PROVEN`

---

## 3. Research application matrix

| Mechanism / research line | State | Production? | Next action |
|---|---|---:|---|
| Mixed INT3/INT4 MoE expert storage | APPLIED + MEASURED | yes/current MoE lineage | preserve as control and format evidence |
| Per-layer MoE cache / LRU residency | APPLIED + MEASURED | yes | harvest lifetime/tier lessons into generic runtime |
| Duplicate in-flight expert-load coalescing | APPLIED + CAUSAL | yes | generalize to weight/block admission |
| Expert parallelism | APPLIED + MEASURED | yes | MoE-specific; do not force onto dense path |
| FUSED loading | APPLIED + MEASURED | yes | generalize contiguous/read-bundling lessons where valid |
| BATCH acquisition / dedupe | APPLIED + MEASURED | yes | generalize scheduler concepts, not expert assumptions |
| QD16 / deeper device queue | FALSIFIED in current regime | no | do not repeat unless model/hardware regime changes |
| Uniform cap increase alone | FALSIFIED in historical MoE regime | no | do not repeat unchanged |
| W3a routing recomputation reuse | CORRECT EXPERIMENT; wall value not established as production win | no | keep diagnostic; does not block dense lane |
| Argmax-prune exact-runtime discriminator | FALSIFIED / banked negative knowledge | no | do not confuse with structural model pruning |
| Adjacent/KRS prefetch probes | BANKED / not production-winning | no | dense lane uses deterministic layer order instead |
| Progressive structural MoE expert pruning | RESEARCHED, NOT YET APPLIED TO FINAL 35B LINE | no | optional Gate E, no longer primary destination |
| Diversity-aware expert scoring | RESEARCHED, NOT YET APPLIED | no | optional Gate E |
| Expert merging / partial-preservation merge | RESEARCHED, NOT YET APPLIED | no | optional Gate E / possible teacher-student evidence |
| Joint pruning + mixed precision | RESEARCHED, NOT YET APPLIED | no | useful to both model-reduction and dense candidate design |
| Knowledge distillation recovery | RESEARCHED, NOT YET APPLIED | no | candidate for dense-student route if earned |
| Native dense-Qwen candidate census | READY | no | Gate B: resolve live model/revision instead of guessing |
| Dense low-bit resident-fit plan | READY | no | Gate B/C primary path |
| Dense sensitive-tensor precision allocation | READY / RESEARCH REQUIRED | no | Gate C with quality gates |
| Dense layer/tensor/block WeightStore | DESIGNED CONCEPTUALLY | no | Gate D only if out-of-core is required |
| Dense sequential next-layer prefetch / double buffer | DESIGNED CONCEPTUALLY | no | Gate D discriminator |
| Dense contiguous read bundling/windowing | RESEARCHED CONCEPTUALLY | no | Gate D if storage path is used |
| Dense activation-sparsity / conditional compute | RESEARCHED CONCEPTUALLY, QUALITY-CHANGING | no | separate approximate lane after exact dense baseline |
| AATC-style KV compression | RESEARCHED, NOT YET APPLIED | no | revisit when dense KV memory is measured; likely more relevant than in 10-attention-layer MoE |
| CompressKV-style KV eviction/compression | RESEARCHED, NOT YET APPLIED | no | same: rank from measured dense KV pressure |
| Generic Colibri low-memory model runtime Forge | DESIGNED, NOT YET FORGED | no | must support both MoE experts and dense weight blocks |

---

## 4. Why the MoE runtime work was still the correct first investment

The work on the full-size MoE model was not wasted even though dense Qwen is the desired destination.

It gave us:

- a correct and measurable control;
- route/output parity machinery;
- reliable storage/cache telemetry;
- exact knowledge of where wall time goes;
- low-bit format and kernel experience;
- lease/lifetime and duplicate-admission lessons;
- page-cache/tier awareness;
- a faster control that makes later experiments honest;
- negative knowledge that prevents repeating bad I/O/cache experiments;
- promotion/rollback discipline.

The MoE runtime work was the **measurement instrument and low-memory serving laboratory**. The dense lane now gets to reuse what is actually general while discarding expert-specific assumptions.

---

## 5. Dense-Qwen acceptance rules

Every dense candidate must report at minimum:

1. model/revision/source and artifact hashes;
2. architecture/parameter count and tokenizer/chat-template identity;
3. source weight bytes and actual converted physical bytes;
4. precision map by tensor class;
5. fixed resident weight bytes;
6. peak/current RSS;
7. KV/context memory growth;
8. physical bytes read per generated token after warmup;
9. page-cache vs physical-disk evidence where measurable;
10. prefill wall / TTFT;
11. decode tok/s under a frozen protocol;
12. total request wall;
13. quality corpus hash and metrics;
14. deterministic repeat / correctness checks;
15. whether any approximation, sparsity, pruning, or distillation was used;
16. rollback/control artifact identity.

No candidate promotes on throughput alone.

No candidate promotes on PPL/NLL alone.

No candidate promotes because a paper predicts a win.

No out-of-core dense design promotes if its measured physical bytes/token make the target throughput impossible on the actual storage device.

---

## 6. Prioritization rule for the 16 GB node

The strategic order is now:

1. **finish/freeze the current MoE control**;
2. **select a dense Qwen target from live source truth**;
3. **make the dense weights fit resident if at all practical** using quality-validated low-bit representation;
4. **protect context/KV headroom** rather than filling all RAM with weights;
5. use **out-of-core dense block scheduling only for the remainder** that cannot fit;
6. optimize compute and sequential prefetch for the actual dense artifact;
7. apply KV compression when measured context pressure makes it worthwhile;
8. explore activation sparsity/conditional compute only in an explicitly quality-changing lane.

The objective is not merely "fewer parameters." The objective is a dense Qwen that is genuinely useful on the real 16 GB node and converts every saved byte into stable capability, context, or throughput.

---

## 7. Platform Forge becomes model-general

The eventual platform target is broader than `ColiExpertStore`.

The runtime must preserve the proven ExpertStore contract for MoE while gaining a model-neutral weight-residency layer capable of representing dense blocks/tensors without forcing fake expert identities.

Candidate decomposition:

- **Model adapter**: architecture, tensor naming, execution order, tokenizer/chat semantics;
- **WeightStore**: model-neutral lease/release/prefetch for addressable weight units;
- **ExpertStore adapter**: maps `(layer, expert)` onto WeightStore-like lifetime/tier semantics without losing MoE-specific behavior;
- **Admission scheduler**: coalescing, bounded parallel I/O, batch/window prefetch;
- **Tier manager**: resident RAM, OS page cache observability, NVMe, optional GPU/other accelerators;
- **Format dispatch**: real low-bit bytes + scales + native kernels;
- **Resource planner**: fixed weights + KV growth + workspaces + OS/safety headroom;
- **Telemetry**: logical requests, physical reads, bytes, latency, residency, page-cache effects;
- **Promotion harness**: hashes, quality/parity, benchmark, deployment pointer, rollback.

The Forge is successful only if it supports both an MoE model and a dense Qwen without copying an entire storage/runtime subsystem into each engine.

---

## 8. Current campaign position

**Main Quest:** finish Gate A, then immediately execute dense-Qwen Gate B.

**Current Boss:** lack of one final apples-to-apples promoted FUSED+BATCH control measurement + fully pinned quality/provenance packet.

**Primary future destination:** `QWEN_DENSE_NODE_FINAL_PROMOTION_PROVEN`.

**Do not open before Gate A closes:** destructive mutation of the production MoE artifact or an ungrounded dense-model implementation based on a guessed checkpoint.

**Do prepare now:** dense runtime interfaces, memory-budget calculator inputs, model-candidate census contract, low-bit container requirements, deterministic reference harness, and candidate artifact namespace.

---

## 9. Update discipline

This file is append/transition oriented.

When a research item changes state, record:

- date;
- exact branch/SHA/model revision;
- experiment/artifact ID;
- previous state;
- new state;
- evidence/receipt path;
- measured effect;
- production decision.

Allowed state vocabulary:

`RESEARCHED` / `READY` / `RUNNING` / `APPLIED` / `MEASURED` / `PROMOTED` / `FALSIFIED` / `BLOCKED` / `SUPERSEDED` / `NOT_APPLICABLE`.

Do not erase negative results. They are part of the asset.
