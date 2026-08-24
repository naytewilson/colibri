# Qwen Dense Runtime Readiness Contract

> Preparation contract for running the user's eventual **dense Qwen-family model** on the 16 GB Colibri node.
>
> This document is intentionally model-revision-neutral until a fresh model census selects an exact dense Qwen checkpoint. It defines what Colibri must know and prove before implementation or promotion.

## 0. Product intent

The current Qwen3.6-35B-A3B MoE campaign is a control/runtime proving ground.

The desired long-term product lane is a **dense Qwen-family model**.

Do not assume that this means mechanically converting the current MoE checkpoint into a dense checkpoint. A native dense Qwen release, a dense distilled student, or another evidence-backed dense Qwen derivative may be the correct target. Resolve that from live model/research truth at campaign start.

The primary objective is:

> maximize useful dense-Qwen capability on a 16 GB node while keeping interactive latency, context headroom, quality, reproducibility, and rollback honest.

---

## 1. Preferred execution hierarchy

### Tier 1 — Fully resident dense

Preferred whenever achievable with quality-valid low-bit weights.

Required memory accounting:

`fixed low-bit weights + scales/metadata + runtime workspaces + KV/context + allocator/OS/safety headroom <= measured safe RAM budget`

Do not use total physical RAM as the model budget.

Do not count an artifact as resident-fit if the runtime expands the majority of low-bit weights back to f16/f32 on load.

### Tier 2 — Mostly resident dense

If only a minority of weights exceed safe RAM headroom:

- pin high-value/frequently reused/sensitive blocks;
- use a small bounded rolling window for the rest;
- exploit deterministic layer order;
- prefetch the next layer/block while the current block computes;
- batch contiguous file ranges when possible;
- measure OS page-cache reuse separately from application residency.

### Tier 3 — Windowed out-of-core dense

Fallback only.

Before building it, calculate the storage lower bound from the proposed physical bytes/token and measured sustainable device throughput. If the lower bound cannot meet the desired interactive token time, stop and change the model/precision/structure instead of trying to schedule around impossible arithmetic.

---

## 2. Candidate selection contract

At the beginning of the dense campaign, perform a fresh 2026 census of available dense Qwen-family checkpoints and current compression/runtime research.

For each candidate capture:

- exact public/model source and immutable revision;
- license/use constraints;
- architecture family;
- parameter count;
- hidden size/layer count;
- attention/KV geometry;
- tokenizer + chat template;
- source precision and physical checkpoint bytes;
- supported context;
- quality/capability evidence relevant to the intended use;
- known quantization compatibility;
- whether the checkpoint is native dense or derived/distilled.

No candidate is selected from name recognition alone.

Selection should optimize the joint objective:

`quality × local usability × context × resident-fit potential × measured runtime support`

not parameter count for its own sake.

---

## 3. Dense format plan

The first conversion campaign should produce a format census rather than one irreversible quantization.

Candidate precision map should distinguish at least:

- embedding;
- attention projections;
- FFN projections;
- normalization parameters;
- final LM head;
- any outlier/sensitive tensor classes identified by evidence.

Requirements:

- native low-bit compute path;
- explicit scale/zero-point/metadata accounting;
- bounds-checked container reads;
- no hidden f32 expansion that defeats the RAM objective;
- deterministic tensor identity/hash receipt;
- model-level quality gate for every material precision change.

Potential formats/methods are experiment candidates, not assumptions. Refresh the current literature before implementation.

---

## 4. Runtime abstraction target

Dense support should extend Colibri coherently.

### Model adapter

Owns:

- architecture geometry;
- tensor naming/layout;
- layer execution order;
- tokenizer/chat semantics;
- architecture-specific math.

### WeightStore

Model-neutral addressable weight-unit contract.

Candidate key:

`(layer, role, block/index)`

Required operations:

- lookup/acquire;
- release;
- prefetch;
- stats;
- immutable identity while leased;
- resident/capacity byte accounting.

Do not fake dense blocks into `(layer, expert)` identifiers just to reuse an API.

### ExpertStore adapter

Keep the existing MoE contract and map it onto shared lifetime/tier infrastructure where safe.

Dense support must not regress expert semantics.

### Admission/window scheduler

Must support:

- in-flight coalescing;
- bounded I/O concurrency;
- deterministic next-layer/window prefetch;
- double-buffer or small rolling window;
- contiguous read bundling;
- fail-closed publication after load/validation;
- cancellation/cleanup on failure.

### Tier manager

Must distinguish:

- application-resident RAM;
- OS page-cache effect;
- physical NVMe reads;
- optional GPU/other accelerator residency.

### Telemetry

Must report enough evidence to distinguish logical cache behavior from actual device I/O.

---

## 5. Dense execution discriminators

### D0 — Memory feasibility

Given an exact candidate and precision map, compute:

- physical weight bytes;
- scale/metadata bytes;
- runtime workspace bytes;
- KV bytes/token and selected-context total;
- safe resident budget;
- expected spill bytes.

Verdicts:

- `DENSE_RESIDENT_FIT_CANDIDATE`
- `DENSE_MOSTLY_RESIDENT_CANDIDATE`
- `DENSE_OUT_OF_CORE_REQUIRED`
- `DENSE_TARGET_INFEASIBLE_ON_NODE`

### D1 — Conversion correctness

Prove:

- all expected tensors present;
- exact shapes/counts;
- container bounds valid;
- deterministic hashes;
- tokenizer/template parity;
- reference logits/tokens within the chosen exact/quantized acceptance contract.

### D2 — Resident path

If fit is possible:

- load once;
- prove no decode-time model-weight reads;
- measure RSS;
- measure TTFT/prefill/decode;
- record low-bit kernel timing;
- establish quality baseline.

### D3 — Windowed path

Only if required:

- measure actual bytes/token;
- measure page-cache vs physical reads;
- measure current-block compute overlap with next-block prefetch;
- sweep bounded window size and I/O concurrency;
- preserve exact output for exact mode;
- reject any configuration whose storage arithmetic cannot meet the target.

### D4 — KV/context pressure

Measure at multiple contexts.

Only after this evidence should KV compression/eviction become a priority.

### D5 — Approximate dense lane

Activation sparsity, neuron/block skipping, pruning, learned hot/cold partitioning, or other conditional compute belong here.

They require explicit quality comparison and separate promotion labels.

---

## 6. Benchmark contract

Every serious dense result must record:

- model ID/revision;
- source and converted artifact hashes;
- Colibri source/binary SHA;
- node CPU/RAM/storage fingerprint;
- precision map;
- physical fixed-weight bytes;
- resident weight bytes;
- window/cache capacity;
- OS page-cache policy;
- context length;
- KV bytes;
- TTFT;
- prefill tok/s or wall;
- decode tok/s;
- total request wall;
- physical bytes read during decode;
- bytes/token;
- read latency/throughput;
- RSS peak/current;
- quality corpus hash + metrics;
- exact vs approximate classification.

Do not compare a batched throughput number to an interactive single-session number without saying so.

---

## 7. Quality/promotion gates

Dense promotion requires:

1. deterministic artifact identity;
2. model-load correctness;
3. tokenizer/chat-template correctness;
4. quality corpus provenance;
5. quality result within an explicitly approved tolerance or a documented capability trade;
6. stable repeated generation;
7. memory within safe node limits;
8. comparable throughput result;
9. rollback artifact/path;
10. production launcher/profile pinned to exact model/runtime hashes.

No promotion on a pretty demo alone.

---

## 8. Reusable lessons inherited from MoE Qwen

Carry forward:

- low-bit native compute rather than decode-to-float-everywhere;
- lease/lifetime safety;
- duplicate admission coalescing;
- bounded device concurrency;
- page-cache awareness;
- logical-vs-physical I/O telemetry;
- exact vs quality-changing lane separation;
- source/binary/artifact hashing;
- frozen benchmark protocols;
- promotion/rollback discipline;
- negative-result preservation.

Do **not** carry forward blindly:

- expert routing assumptions;
- per-layer expert LRU geometry;
- top-k/K<=8 assumptions;
- pilot expert prediction;
- MoE-specific BATCH request shape.

---

## 9. Dense-specific research queue

Refresh these against current literature immediately before implementation:

- state-of-the-art dense low-bit quantization for CPU/native inference;
- mixed-precision/outlier-aware dense quantization;
- flash/NVMe-backed dense inference with windowing and bundled reads;
- activation sparsity / hot-cold neuron partitioning;
- structured pruning and distillation for dense students;
- KV-cache compression/eviction for the selected dense attention architecture;
- CPU/GPU/ANE or other accelerator partitioning if the node/hardware lane changes.

Research status vocabulary:

`RESEARCHED / READY / EXPERIMENTED / APPLIED / MEASURED / PROMOTED / FALSIFIED / BLOCKED`.

---

## 10. Campaign order

1. `QWEN36_FULLSIZE_CONTROL_FROZEN`
2. `QWEN_DENSE_TARGET_AND_MEMORY_PLAN_FROZEN`
3. `QWEN_DENSE_ARTIFACT_QUALITY_PROVEN`
4. `COLIBRI_DENSE_QWEN_RUNTIME_V0_PROVEN`
5. dense runtime optimization / KV work as measurements demand
6. `QWEN_DENSE_NODE_FINAL_PROMOTION_PROVEN`
7. cross-model Forge into `COLIBRI_LOW_MEMORY_MODEL_RUNTIME_V1_FORGED`

Optional MoE structural-compression work may run as a bounded side lane when it directly helps the currently deployed service or yields evidence reusable by dense Qwen. It must not silently replace the dense destination.

---

## 11. Start condition

Do not begin destructive dense implementation until the current MoE control closes.

Preparation that is safe now:

- current dense-Qwen model census template;
- WeightStore interface design;
- memory-budget calculator design;
- low-bit container contract;
- deterministic reference harness design;
- benchmark/receipt schema;
- artifact namespace and rollback layout.

The first dense execution wave should start from a clean owned branch/worktree and must not modify the frozen MoE control/deployment.

Target first verdict:

`QWEN_DENSE_RUNTIME_CAMPAIGN_READY`
