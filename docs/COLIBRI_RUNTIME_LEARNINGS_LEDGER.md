# Colibri Runtime Learnings Ledger

> Durable record of what the **runtime that actually executes models** has learned across Qwen3.6, OLMoE, Inkling, DeepSeek/Kimi-style engines, and the shared Colibri infrastructure.
>
> This is separate from model-compression research. A model can change while these runtime lessons remain valuable.

## 1. Current mental model

Colibri is not just a model loader. On RAM-constrained MoE systems it is effectively a **memory/storage/compute scheduler**:

`model adapter -> router -> expert admission/residency -> RAM/page-cache/NVMe/GPU tiers -> quantized compute -> telemetry`.

For oversized sparse models, storage hierarchy and admission policy are part of practical inference performance.

## 2. Proven reusable lessons

### Expert ownership must have lease/pinning semantics

The shared `ColiExpertStore` already exposes `(layer, expert)` lookup, release, prefetch, stats, resident bytes, capacity bytes, and a lease contract.

Inkling exposed a severe correctness failure in a model-specific cache: a slot pointer could remain live while LRU eviction repurposed the slot for another expert, causing silent computation with the wrong weights. The repaired engine processes routed experts in bounded rounds so an expert cannot be evicted while in use.

**Runtime law:** no cache optimization may permit an admitted expert buffer to change identity before its consumer releases it.

### Duplicate physical admission must coalesce

Qwen3.6 proved that two logical requests for the same `(layer, expert)` must not independently fetch and publish duplicate physical copies. In-flight admission visibility/coalescing materially reduced duplicate residency, bytes read, admission time, and MoE wall.

**Runtime law:** distinguish logical expert requests from physical admissions and make duplicate admission impossible by construction.

### Batch acquisition is a scheduler problem, not eight independent lookups

Qwen3.6 BATCH acquisition proved that prefill can derive a set of expert requests, deduplicate them, reserve them, and issue bounded parallel loads while preserving route/output semantics.

**Runtime law:** expose a batch/admission scheduler above the raw store lookup rather than making each model reinvent one-expert-at-a-time acquisition.

### Device queue depth has a measured saturation point

On the current Qwen node/storage regime, effective expert-load concurrency saturated around 7–8 streams; QD16 worsened wall time and doubled I/O work/queue pressure.

**Runtime law:** concurrency is measured hardware state, not a monotonic tuning knob. Worker count belongs in a hardware/runtime profile and must be re-measured after model/hardware changes.

### OS page cache is a real tier

OLMoE changed from `pread + fadvise(DONTNEED)` on every expert read to retaining pages in the OS page cache by default, with an opt-out for RAM-tight hosts. This means an LRU miss in the application cache does not necessarily need to become a physical disk miss.

**Runtime law:** model the OS page cache as a distinct tier/effect when measuring physical I/O; do not equate application-cache miss with NVMe access.

### Exact optimization and quality-changing optimization must be separated

Colibri/Inkling support TOPP-style routed-expert trimming as an opt-in quality lever. It can materially reduce expert loads, but changes computation and therefore cannot be presented as a free exact-runtime speedup.

**Runtime law:** exact-serving optimizations and approximate/quality-changing serving policies need separate benchmark and promotion lanes.

### Quantization belongs to runtime format dispatch, not one model

Across Qwen/Inkling/Kimi/other engines, runtime support now spans multiple low-bit formats and different dense/expert precision mixes. The engine must account for real stored bytes, scale metadata, and per-format compute rather than assuming one global bit-width.

**Runtime law:** capacity planning and telemetry must be format-aware at tensor/expert granularity.

## 3. Existing generic substrate

Existing shared pieces include:

- `c/expert_store.h`: model-shaped `(layer, expert)` store interface, leases, prefetch, stats;
- `c/expert_store_registry.*`: pluggable expert-store backend concept;
- family/model registry and per-family planning geometry;
- shared tensor/container readers;
- quantized kernels/format support;
- route tracing / telemetry / hardware probes;
- CLI/gateway family selection and common serving surfaces.

These are the beginnings of a reusable out-of-core inference substrate.

## 4. Known architecture debt

The generic runtime is **not fully model-neutral yet**.

- `expert_store_registry.h` still imports DeepSeek-V4-specific configuration/types.
- Qwen3.6 still owns important admission/cache/pilot/BATCH/FUSED machinery inside `qwen36.c` rather than consuming a shared admission scheduler.
- generic telemetry still has engine-specific structural assumptions in places.
- not every model uses the same lease/admission semantics.

Therefore "Colibri runtime generalized" is not yet a proven claim.

## 5. Runtime Forge target

After the Qwen full-size control is frozen and the structural successor is proven, extract the mechanisms that survive both regimes into a shared runtime:

1. **Model adapter**: geometry, tensor naming/layout, router semantics, architecture-specific compute.
2. **ExpertStore**: lookup/lease/release, storage backend, residency and capacity.
3. **Admission scheduler**: in-flight reservation, dedupe, coalescing, bounded parallel acquisition, batch acquisition/prefetch.
4. **Tier manager**: resident RAM, OS page cache observability, NVMe, optional GPU/other backend.
5. **Telemetry**: logical requests, physical admissions, bytes read, hit/miss by tier, latency, resident bytes, decode-window counters.
6. **Resource planner**: fixed model bytes, context-growth bytes, expert-pool bytes, available RAM, precision mix, expected backing-store pressure.
7. **Promotion harness**: source/artifact hashes, parity/quality, performance, deployment pointer, rollback.

## 6. Why Forge follows the compressed Qwen successor

Do not freeze a universal abstraction around an intermediate full-size Qwen experiment.

Correct sequence:

1. freeze trustworthy full-size Qwen control;
2. structurally compress Qwen;
3. re-tune the runtime for the changed residency regime;
4. identify which mechanisms survive both full-size and compressed regimes;
5. Forge those into shared Colibri infrastructure;
6. prove reuse with a second model.

This gives future models the final lessons rather than yesterday's implementation accident.

## 7. Second-model proof requirement

A generic runtime is not proven generic by Qwen alone.

Completion requires at least one second existing MoE engine to consume/exercise the same shared ExpertStore/admission contract without copying Qwen cache/admission machinery.

Inkling/OLMoE are especially useful regression oracles because they have already exposed:

- eviction/lifetime correctness hazards;
- extreme RAM-constrained expert streaming;
- page-cache behavior;
- approximate expert trimming as a distinct quality lane.

## 8. Current status

- Shared ExpertStore contract: **EXISTS**.
- Pluggable backend registry: **EXISTS, PARTLY ENGINE-COUPLED**.
- Qwen exact-runtime lessons: **PROVEN IN MODEL-SPECIFIC PATH**.
- Cross-model runtime lessons: **PROVEN IN SEVERAL ENGINES**.
- Model-neutral admission scheduler: **NOT YET FORGED**.
- Qwen migrated onto fully generic expert runtime: **NOT YET PROVEN**.
- Second-model shared-runtime proof: **NOT YET PROVEN**.

Target eventual verdict:

`COLIBRI_OUT_OF_CORE_MOE_RUNTIME_V1_FORGED`

## 9. Update discipline

For each runtime experiment record:

- model/family;
- exact source SHA;
- hardware/storage profile;
- cache/admission configuration;
- logical expert requests;
- physical admissions;
- tier hits/misses when measurable;
- bytes read;
- admission latency;
- quality/parity class (exact vs approximate);
- wall/tok/s effect;
- state: `PROVEN / FALSIFIED / BLOCKED / PROMOTED / SUPERSEDED`.

Negative results are retained.