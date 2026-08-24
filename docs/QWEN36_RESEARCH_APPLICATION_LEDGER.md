# Qwen3.6 Research Application + Model-Finish Ledger

> Durable controller ledger for deciding **what gets applied, when, and why** in the Qwen3.6-35B-A3B campaign.
>
> This file complements `docs/QWEN36_CAMPAIGN_LEDGER.md`. The campaign ledger preserves historical measurements; this ledger controls the forward research/application queue.
>
> **Rule:** research is not "applied" until it has an implementation or artifact, correctness/quality evidence, a comparable node benchmark, and an explicit production/promotion decision.

## 0. User directive / campaign north star

The 35B source model is **not** intended to remain physically oversized forever on a 16 GB node.

The order is deliberate:

1. finish the exact serving/runtime baseline correctly;
2. freeze a trustworthy full-size control;
3. structurally reduce the model/expert footprint using the best supported research;
4. re-optimize the runtime for the new smaller residency regime;
5. promote only after quality + throughput + rollback evidence is complete.

Do not spend indefinite effort polishing the full-size model after the exact-runtime frontier is already measured. The full-size artifact remains the control/oracle, not the permanent destination.

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

This branch is the current **full-size control source** unless later source truth explicitly supersedes it.

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

The campaign ledger preserves the historical 5.33 tok/s warm-median serving anchor and the model/runtime chronology. Current mutable node state must still be re-verified locally.

---

## 2. Critical path — what we do next

### GATE A — Finish the full-size exact control correctly

**Priority: NOW.**

This is a closure wave, not another open-ended optimization campaign.

Required before model surgery:

- verify the exact promoted source/binary/snapshot identities on the node;
- run the current promoted FUSED+BATCH binary under the same frozen warm-serving protocol used for the accepted headline anchor, so current tok/s is apples-to-apples;
- re-establish the quality baseline using a hashed, explicitly named evaluation corpus;
- verify deterministic route/output parity and accounting conservation for the promoted configuration;
- preserve deployment `current` + rollback proof;
- record RAM, expert-pool bytes, cache residency, physical expert misses/token, bytes/token, and storage latency under the same run;
- freeze these results as the full-size control for every later compression candidate.

**Exit:** `QWEN36_FULLSIZE_CONTROL_FROZEN`

Once this exit is earned, stop spending critical-path effort on tiny exact-runtime tweaks unless a measurement shows a new high-value software wall.

### GATE B — Structural expert-footprint reduction

**Priority: NEXT, immediately after Gate A.**

Primary goal:

> reduce real executed/stored expert weight enough to materially change the 16 GB node's residency/I/O regime while preserving useful model quality.

Research families to evaluate together rather than as isolated folklore:

- progressive expert pruning with re-scoring after each step;
- diversity-aware expert importance/scoring;
- expert merging / partial-preservation merging;
- joint structural pruning + mixed precision;
- layer/expert sensitivity rather than uniform pruning;
- knowledge-distillation recovery only when a promising compressed candidate earns it;
- quantization retuning for the surviving experts.

Important distinction:

- masks/zeros/sparse-looking tensors do **not** count as structural compression if the runtime still stores/reads/executes the same physical work;
- a smaller model file does **not** count as a win by itself;
- the node must show lower real bytes/read work and/or higher residency and higher comparable tok/s.

Keep the full-size model immutable as control. Build compressed successors under new artifact IDs.

**Exit:** `QWEN36_STRUCTURAL_SUCCESSOR_PROVEN`

### GATE C — Re-optimize runtime for the compressed successor

**Priority: AFTER Gate B produces a quality-valid candidate.**

Compression changes the causal regime, so re-measure instead of carrying old conclusions forward blindly.

Revisit:

- cache capacity and per-layer budgets;
- pinned/resident set;
- FUSED/BATCH worker count;
- prefetch/pilot usefulness;
- expert admission concurrency;
- storage-vs-compute split;
- quantized kernels for the surviving precision mix;
- whether the expert pool now fits mostly or wholly in safe RAM headroom.

Previously falsified ideas stay falsified **for the old regime**, but may be re-opened only if the structural/hardware regime materially changes and the ledger records why.

**Exit:** `QWEN36_COMPRESSED_RUNTIME_TUNED`

### GATE D — Final production promotion

Required:

- source SHA;
- artifact hashes;
- deterministic correctness/route evidence;
- quality comparison against Gate-A control;
- comparable cold + warm throughput;
- TTFT/prefill/decode decomposition;
- RSS / residency / miss rate / bytes read;
- clean promotion directory;
- atomic current pointer;
- rollback proof;
- smoke run.

**Exit:** `QWEN36_NODE_FINAL_PROMOTION_PROVEN`

---

## 3. Research application matrix

| Mechanism / research line | State | Production? | Next action |
|---|---|---:|---|
| Mixed INT3/INT4 expert storage | APPLIED + MEASURED | yes/current lineage | preserve as control; retune after structural compression |
| Per-layer cache / LRU residency | APPLIED + MEASURED | yes | re-size after Gate B |
| Duplicate in-flight load coalescing | APPLIED + CAUSAL | yes | harvest into generic runtime later |
| Expert parallelism | APPLIED + MEASURED | yes | re-benchmark after Gate B |
| FUSED loading | APPLIED + MEASURED | yes | preserve through Gate A |
| BATCH acquisition / dedupe | APPLIED + MEASURED | yes | preserve through Gate A |
| QD16 / deeper device queue | FALSIFIED in current regime | no | do not repeat unless regime changes |
| Uniform cap increase alone | FALSIFIED in historical regime | no | do not repeat unchanged |
| W3a routing recomputation reuse | CORRECT EXPERIMENT; wall value not established as production win | no | keep diagnostic; do not block Gate B |
| Argmax-prune exact-runtime discriminator | FALSIFIED / banked negative knowledge | no | do not confuse with structural model pruning |
| Adjacent/KRS prefetch probes | BANKED / not production-winning | no | reopen only after changed residency regime |
| Progressive structural expert pruning | RESEARCHED, NOT YET APPLIED TO FINAL 35B LINE | no | Gate B primary experiment family |
| Diversity-aware expert scoring | RESEARCHED, NOT YET APPLIED | no | Gate B scoring arm |
| Expert merging / partial-preservation merge | RESEARCHED, NOT YET APPLIED | no | Gate B candidate after scoring baseline |
| Joint pruning + mixed precision | RESEARCHED, NOT YET APPLIED | no | Gate B high-priority interaction study |
| Knowledge distillation recovery | RESEARCHED, NOT YET APPLIED | no | use only for a promising lossy candidate |
| MTP distillation | CURRENTLY NOT DIRECTLY APPLICABLE | no | canonical artifact has no positive MTP tensor evidence; do not invent an MTP lane |
| AATC-style KV compression | RESEARCHED, NOT YET APPLIED | no | lower priority while expert I/O dominates; revisit if Gate B moves bottleneck |
| CompressKV-style KV eviction/compression | RESEARCHED, NOT YET APPLIED | no | same: only 10 full-attention layers; re-rank after Gate B |
| Generic Colibri out-of-core MoE runtime Forge | DESIGNED, NOT YET FORGED | no | preserve as platform Forge, but do not delay Gate B critical path |

---

## 4. Why runtime-first was still the correct order

The work on the full-size model was not wasted.

It gave us:

- a correct and measurable control;
- route/output parity machinery;
- reliable storage/cache telemetry;
- exact knowledge of where wall time goes;
- a faster control that makes every compression comparison fairer;
- negative knowledge that prevents repeating bad I/O/cache experiments;
- serving mechanisms the compressed successor can inherit.

Without that control, a structurally smaller candidate could appear faster simply because it silently changed routing, quality, caching, or work accounting.

The runtime work was the **measurement instrument and serving foundation**. Gate B is where we stop treating the 35B physical footprint as sacred.

---

## 5. Structural-compression acceptance rules

Every candidate must report at minimum:

1. source/control SHA and artifact hashes;
2. structural change: which layers/experts/tensors were actually removed/merged/re-quantized;
3. physical parameter/weight bytes before vs after;
4. expert-pool bytes before vs after;
5. RAM-resident bytes / residency ratio;
6. physical expert misses/token;
7. bytes read/token;
8. prefill wall;
9. decode tok/s under the frozen protocol;
10. total request wall;
11. quality corpus hash and metrics;
12. deterministic repeat / correctness checks;
13. whether KD/retraining was used;
14. rollback/control artifact identity.

No candidate promotes on throughput alone.

No candidate promotes on PPL/NLL alone.

No candidate promotes because a paper predicts a win.

---

## 6. Prioritization rule for the 16 GB node

The current strategic order is:

1. **expert-pool structural reduction** — attacks the dominant RAM/NVMe pressure directly;
2. **joint precision + structure** — can compound the residency gain;
3. **re-tune residency/cache/I/O runtime** after the pool changes;
4. **KV compression** when measured KV growth becomes important enough to matter;
5. micro-optimizations only when their measured ceiling exceeds run noise.

The objective is not merely "fewer parameters." The objective is to cross useful residency thresholds on the real node and convert saved bytes into real end-to-end token throughput without unacceptable quality loss.

---

## 7. Platform Forge is not forgotten

The reusable Colibri Out-of-Core MoE Runtime remains a required Forge objective, but it is **not allowed to delay the Qwen structural-compression critical path**.

Best timing:

- preserve all full-size serving lessons now in this ledger and the existing campaign ledger;
- finish Gate A;
- execute Gate B/C on Qwen;
- then extract the proven final mechanisms into the generic ExpertStore/admission/telemetry surfaces using both full-size and compressed Qwen evidence plus a second-model proof.

That way we generalize the mechanism that actually survives the final model, rather than freezing an abstraction around an intermediate experiment.

---

## 8. Current campaign position

**Main Quest:** finish Gate A, then immediately enter structural model compression.

**Current Boss:** lack of one final apples-to-apples promoted FUSED+BATCH control measurement + fully pinned quality/provenance packet.

**Do not open before Gate A closes:** destructive pruning/merging of the production artifact.

**Do prepare now:** structural-compression experiment design, scoring instrumentation, immutable source snapshots, and candidate artifact namespace.

---

## 9. Update discipline

This file is append/transition oriented.

When a research item changes state, record:

- date;
- exact branch/SHA;
- experiment/artifact ID;
- previous state;
- new state;
- evidence/receipt path;
- measured effect;
- production decision.

Allowed state vocabulary:

`RESEARCHED` / `READY` / `RUNNING` / `APPLIED` / `MEASURED` / `PROMOTED` / `FALSIFIED` / `BLOCKED` / `SUPERSEDED` / `NOT_APPLICABLE`.

Do not erase negative results. They are part of the asset.
