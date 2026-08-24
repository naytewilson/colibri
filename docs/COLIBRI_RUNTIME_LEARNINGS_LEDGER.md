# Colibri Runtime Learnings Ledger

> Durable record of what the **runtime that actually executes models** has learned across Qwen3.6, OLMoE, Inkling, DeepSeek/Kimi-style engines, dense-model research, and the shared Colibri infrastructure.
>
> This is separate from model-compression research. A model can change while these runtime lessons remain valuable.

## 1. Current mental model

Colibri is not just a model loader. On RAM-constrained systems it is becoming a **low-memory neural weight runtime**:

`model adapter -> weight demand -> residency/admission -> RAM/page-cache/NVMe/GPU tiers -> quantized compute -> telemetry`.

For MoE models, the natural demand unit is usually `(layer, expert)` and the router determines which experts are needed.

For dense models, the natural unit is more likely `(layer, tensor/block)` and exact execution normally touches essentially every layer each token. That makes dense out-of-core serving a different scheduling problem even when it can reuse the same lifetime, tier, quantization, telemetry, and promotion machinery.

**North star:** Colibri should ultimately run both oversized MoE and dense Qwen-family models without duplicating the storage/runtime stack.

---

## 2. Proven reusable lessons

### Weight ownership must have lease/pinning semantics

The shared `ColiExpertStore` already exposes `(layer, expert)` lookup, release, prefetch, stats, resident bytes, capacity bytes, and a lease contract.

Inkling exposed a severe correctness failure in a model-specific cache: a slot pointer could remain live while LRU eviction repurposed the slot for another expert, causing silent computation with the wrong weights. The repaired engine processes routed experts in bounded rounds so an expert cannot be evicted while in use.

**Runtime law:** no cache optimization may permit an admitted weight buffer to change identity before its consumer releases it. This law applies to dense blocks as strongly as to MoE experts.

### Duplicate physical admission must coalesce

Qwen3.6 proved that two logical requests for the same `(layer, expert)` must not independently fetch and publish duplicate physical copies. In-flight admission visibility/coalescing materially reduced duplicate residency, bytes read, admission time, and MoE wall.

**Runtime law:** distinguish logical weight requests from physical admissions and make duplicate admission impossible by construction.

### Batch/window acquisition is a scheduler problem

Qwen3.6 BATCH acquisition proved that prefill can derive a set of expert requests, deduplicate them, reserve them, and issue bounded parallel loads while preserving route/output semantics.

Dense exact execution does not have expert routing, but it does have predictable layer order. That suggests a different reusable scheduler shape: current-layer lease plus bounded next-layer/window prefetch, possibly double-buffered.

**Runtime law:** expose an admission scheduler above the raw store rather than making every model reinvent I/O orchestration.

### Device queue depth has a measured saturation point

On the current Qwen node/storage regime, effective expert-load concurrency saturated around 7–8 streams; QD16 worsened wall time and doubled I/O work/queue pressure.

**Runtime law:** concurrency is measured hardware state, not a monotonic tuning knob. Worker count belongs in a hardware/runtime profile and must be re-measured after model/hardware changes.

### OS page cache is a real tier

OLMoE changed from `pread + fadvise(DONTNEED)` on every expert read to retaining pages in the OS page cache by default, with an opt-out for RAM-tight hosts. This means an application-cache miss does not necessarily need to become a physical disk miss.

**Runtime law:** model the OS page cache as a distinct tier/effect when measuring physical I/O; do not equate application-cache miss with NVMe access.

### Dense exact serving has a harder I/O law

A conventional dense transformer generally requires essentially all layer weights for every generated token. Therefore naive "read every missing dense layer from SSD every token" can become storage-bandwidth-bound immediately.

**Runtime law:** for dense models, prefer making the low-bit artifact resident first. Out-of-core dense scheduling is a fallback for the remainder, not a reason to stream weights unnecessarily.

Before implementing a dense streaming design, estimate and then measure:

`physical_weight_bytes_per_generated_token / sustainable_storage_bytes_per_second`

If that lower bound already exceeds the desired token time, the design is rejected before optimization theater begins.

### Deterministic layer order is an asset for dense models

Unlike routed MoE expert access, exact dense layer execution has highly predictable ordering.

**Runtime opportunity:** current-layer execution can overlap bounded prefetch of the next layer/block/window. Dense mode should exploit sequential order, contiguous read bundling, and double-buffer/window residency rather than pretending dense blocks are random experts.

### Exact optimization and quality-changing optimization must be separated

Colibri/Inkling support TOPP-style routed-expert trimming as an opt-in quality lever. It can materially reduce expert loads, but changes computation and therefore cannot be presented as a free exact-runtime speedup.

Dense activation sparsity, neuron skipping, structured pruning, or conditional execution have the same classification issue.

**Runtime law:** exact-serving optimizations and approximate/quality-changing serving policies need separate benchmark and promotion lanes.

### Quantization belongs to runtime format dispatch, not one model

Across Qwen/Inkling/Kimi/other engines, runtime support now spans multiple low-bit formats and different dense/expert precision mixes. The engine must account for real stored bytes, scale metadata, and per-format compute rather than assuming one global bit-width.

For a dense Qwen on 16 GB, this becomes even more important: low-bit weights are useful only if Colibri computes directly from them instead of expanding the whole model back to f16/f32 in RAM.

**Runtime law:** capacity planning and telemetry must be format-aware at tensor/block granularity.

### Batch size can amortize dense weight movement but cannot define interactive success

Throughput-oriented out-of-core dense systems can amortize a weight read across many sequences. Interactive local serving cannot rely on giant batches to hide a fundamentally bad per-request storage budget.

**Runtime law:** report batch throughput separately from interactive single/few-request latency and decode tok/s.

---

## 3. Existing generic substrate

Existing shared pieces include:

- `c/expert_store.h`: model-shaped `(layer, expert)` store interface, leases, prefetch, stats;
- `c/expert_store_registry.*`: pluggable expert-store backend concept;
- family/model registry and per-family planning geometry;
- shared tensor/container readers;
- quantized kernels/format support;
- route tracing / telemetry / hardware probes;
- CLI/gateway family selection and common serving surfaces.

These are the beginnings of a reusable low-memory inference substrate.

The current generic surface is still biased toward experts. Dense support should extend the abstraction coherently rather than create a disconnected second runtime.

---

## 4. Known architecture debt

The generic runtime is **not fully model-neutral yet**.

- `expert_store_registry.h` still imports DeepSeek-V4-specific configuration/types.
- Qwen3.6 still owns important admission/cache/pilot/BATCH/FUSED machinery inside `qwen36.c` rather than consuming a shared admission scheduler.
- generic telemetry still has engine-specific structural assumptions in places.
- not every model uses the same lease/admission semantics.
- there is no proven model-neutral dense `WeightStore`/block-residency contract yet.
- there is no proven dense sequential-prefetch scheduler yet.

Therefore "Colibri runtime generalized" is not yet a proven claim.

---

## 5. Runtime Forge target

After the Qwen MoE control is frozen and a dense Qwen candidate is selected/built, extract mechanisms into a shared runtime that supports both sparse and dense demand patterns.

Target decomposition:

1. **Model adapter**: geometry, tensor naming/layout, execution order, router semantics where applicable, architecture-specific compute.
2. **WeightStore**: model-neutral lookup/lease/release/prefetch for addressable weight units.
3. **ExpertStore adapter**: preserves `(layer, expert)` semantics on top of the shared lifetime/tier contract without losing expert-specific behavior.
4. **Admission scheduler**: in-flight reservation, dedupe, coalescing, bounded parallel acquisition, batch/window prefetch.
5. **Tier manager**: resident RAM, OS page cache observability, NVMe, optional GPU/other backend.
6. **Telemetry**: logical requests, physical admissions, bytes read, hit/miss by tier, latency, resident bytes, decode-window counters.
7. **Resource planner**: fixed model bytes, context/KV-growth bytes, workspaces, precision mix, available RAM, expected backing-store pressure.
8. **Format dispatch**: native low-bit kernels and metadata-aware byte accounting.
9. **Promotion harness**: source/artifact hashes, parity/quality, performance, deployment pointer, rollback.

Do not hide all of this inside one giant interface.

---

## 6. Dense-Qwen operating modes

### Mode A — RESIDENT_DENSE (preferred)

Goal: low-bit dense weights + runtime state + useful KV/context fit inside safe measured RAM headroom.

Behavior:

- weights stay resident;
- no per-token weight I/O;
- focus shifts to quantized matmul throughput, memory bandwidth, KV/context, thread scheduling, and accelerator use;
- storage is startup/load-time infrastructure, not decode-time infrastructure.

This is the preferred design on a 16 GB node whenever quality-valid quantization makes it possible.

### Mode B — WINDOWED_DENSE (fallback)

Use only when a selected dense model cannot safely fit resident.

Behavior:

- weights addressed by layer/tensor/block;
- current block leased/pinned until compute completes;
- bounded next-layer/window prefetch;
- double-buffer or small rolling resident window;
- contiguous read bundling where container layout permits;
- page-cache effects measured separately from physical disk;
- physical bytes/token measured directly;
- no claim of viability until storage lower bound and real benchmark both pass.

### Mode C — APPROX_DENSE (separate quality-changing lane)

Possible mechanisms include activation sparsity, structured neuron/block skipping, pruning, or learned hot/cold partitioning.

These may make a nominally dense model behave sparsely, but they change computation or model structure and therefore require their own quality/evaluation contract.

Never mix Mode C numbers into exact Mode A/B claims.

---

## 7. Why the Forge sequence changed

The user's desired long-term Qwen is dense. Therefore the Forge should not wait until every possible MoE compression experiment is exhausted.

Correct sequence:

1. freeze trustworthy full-size MoE Qwen control;
2. perform a fresh dense-Qwen model/capability census;
3. build the selected dense low-bit artifact;
4. establish RESIDENT_DENSE if possible, otherwise discriminate WINDOWED_DENSE viability;
5. re-measure bottlenecks on the dense artifact;
6. extract the mechanisms that survive both MoE and dense operation into shared Colibri infrastructure;
7. prove no regression with at least one existing second MoE engine such as Inkling/OLMoE.

This gives future models the final cross-architecture lessons rather than hard-coding Qwen3.6-MoE accidents into the platform.

---

## 8. Dense proof requirement

A model-general runtime is not proven by several MoE engines alone.

Completion now requires:

- one Qwen MoE control path;
- one dense Qwen path using the shared weight/tier/format/telemetry machinery;
- at least one independent MoE regression consumer to prove dense generalization did not break expert semantics.

Dense proof must show:

- exact model/revision and artifact hashes;
- actual low-bit resident bytes;
- whether weights are resident or windowed;
- KV/context memory growth;
- physical bytes/token;
- page-cache/physical-I/O evidence if windowed;
- TTFT/prefill/decode timing;
- quality/correctness provenance;
- rollback.

---

## 9. Research already relevant to dense Colibri

These are research directions, not local proof until implemented and measured:

- flash-backed dense inference with reuse/windowing and bundled reads;
- hot/cold neuron or block partitioning from activation sparsity;
- CPU/RAM/storage tier scheduling with batched amortization;
- low-bit dense quantization with outlier/sensitive-tensor protection;
- KV-cache compression once dense attention/context memory is measured;
- distillation/pruning when a native dense target cannot meet the 16 GB quality/speed envelope directly.

Before irreversible dense-model work, refresh the literature/model census using current 2026 sources rather than treating any named method as eternal best practice.

---

## 10. Current status

- Shared ExpertStore contract: **EXISTS**.
- Pluggable backend registry: **EXISTS, PARTLY ENGINE-COUPLED**.
- Qwen MoE exact-runtime lessons: **PROVEN IN MODEL-SPECIFIC PATH**.
- Cross-model MoE runtime lessons: **PROVEN IN SEVERAL ENGINES**.
- Model-neutral admission scheduler: **NOT YET FORGED**.
- Model-neutral WeightStore: **NOT YET FORGED**.
- Dense resident-fit Qwen path: **READY TO PLAN AFTER CURRENT CONTROL CLOSES**.
- Dense windowed runtime: **DESIGNED CONCEPTUALLY, UNMEASURED**.
- Dense activation-sparse lane: **RESEARCHED CONCEPTUALLY, QUALITY-CHANGING, UNMEASURED**.
- Qwen migrated onto fully generic runtime: **NOT YET PROVEN**.
- Dense-Qwen shared-runtime proof: **NOT YET PROVEN**.

Intermediate target:

`COLIBRI_DENSE_QWEN_RUNTIME_V0_PROVEN`

Eventual target:

`COLIBRI_LOW_MEMORY_MODEL_RUNTIME_V1_FORGED`

---

## 11. Update discipline

For each runtime experiment record:

- model/family/revision;
- exact source SHA;
- hardware/storage profile;
- resident/windowed mode;
- weight precision map;
- cache/admission configuration;
- logical weight/expert requests;
- physical admissions/reads;
- tier hits/misses when measurable;
- bytes read and bytes/token;
- admission/prefetch latency;
- KV/context memory;
- quality/parity class (exact vs approximate);
- TTFT / decode wall / tok/s effect;
- state: `PROVEN / FALSIFIED / BLOCKED / PROMOTED / SUPERSEDED`.

Negative results are retained.
