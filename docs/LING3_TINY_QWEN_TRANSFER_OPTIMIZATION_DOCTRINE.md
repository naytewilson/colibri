# Ling 3.0 Tiny — QWEN36 Transfer + Maximum-Throughput Doctrine

## Mission

Port Ling 3.0 Tiny into Colibri as a **correct first, then aggressively throughput-tuned** resident-capable hybrid MoE engine for the Dell target.

The performance goal is not merely to make Ling run. The goal is to carry forward every QWEN36 runtime lesson that survives the architecture/regime change, remove avoidable dispatch and orchestration overhead, keep low-bit weights native, and measure the true resident-model ceiling on the Dell.

QWEN36 remains an immutable control campaign and must be resumable. Ling is a temporary main quest, not a rewrite of QWEN36 history.

## Governing QWEN36 evidence to read before optimization

Workers must inspect current source truth plus these durable records before inventing new mechanisms:

- `docs/QWEN36_CAMPAIGN_LEDGER.md` on the durable QWEN36 ledger line;
- `docs/QWEN36_RESEARCH_APPLICATION_LEDGER.md`;
- `docs/COLIBRI_RUNTIME_LEARNINGS_LEDGER.md`;
- QWEN36 frozen production source `f04359aab31a388cc47d36e96c3ea36400061cb6` unless live source truth supersedes it;
- exact-next routing-reuse experiment `09e5bf13b758fb2e18f1e68679f2ff6a761a2f75` as diagnostic evidence, not automatic production authority.

Historical QWEN36 evidence is transferable only by mechanism. Do not transplant host-specific constants or performance claims into Ling without remeasurement.

## Primary optimization objective

After correctness is sealed, optimize **end-to-end tokens/sec and latency on the Dell**, not isolated kernel microbenchmarks.

Required decomposition:

`token wall = attention/KDA + routing + expert/shared MLP compute + LM head + dispatch/orchestration + memory movement + synchronization + any physical I/O`

The worker must instrument enough of this equation to identify the current largest wall before each optimization wave.

A local improvement that increases another component or fails to move end-to-end tok/s is not a throughput win.

## QWEN36 lessons that MUST be applied or explicitly falsified for Ling

### 1. Native low-bit residency is the preferred regime

QWEN36 proved that physical representation matters. Ling should consume the official INT4 checkpoint directly when cleanly possible, or use a deterministic Colibri-native packed artifact.

Rules:

- keep weights packed through compute wherever kernels support it;
- do not expand the model wholesale to FP16/F32 merely because conversion is easy;
- account for scales/metadata in real resident bytes;
- pre-resolve tensor locations/metadata at startup so decode does not perform name-based tensor discovery;
- if the complete useful low-bit model fits with runtime/context headroom, prefer full residency and make decode-time model-weight I/O approach zero.

### 2. Duplicate physical admission must be impossible

QWEN36's in-flight coalescing removed duplicate resident inserts and materially reduced physical loads, admitted bytes, admission wall, MoE wall, and RSS in its measured regime.

If Ling is fully resident, this should collapse mostly into a startup invariant rather than a hot decode feature.

If any weights remain lazy/windowed:

- one authoritative in-flight identity per weight/expert;
- reserve -> load -> publish / abort;
- duplicate logical requests coalesce;
- never publish partially loaded bytes;
- logical requests and physical loads must be counted separately.

Do not recreate independent competing in-flight state.

### 3. Eliminate dispatch/bookkeeping from the resident hot path

For Ling, full or near-full residency changes the boss: storage admission may disappear, exposing CPU dispatch and compute overhead that QWEN36 could hide behind I/O.

The resident decode path should be designed so the common hit case does **not** repeatedly pay for machinery whose answer is already known.

Discriminate and remove where safe:

- mutex/lock operations on immutable resident expert lookup;
- linear cache scans when expert pointers can be directly indexed;
- repeated residency tests for permanently resident tensors;
- per-expert `malloc/free`;
- per-token or per-layer temporary allocation;
- repeated tensor-name formatting/hash lookup;
- repeated parsing of static quant metadata;
- repeated construction/destruction of thread work;
- redundant function-layer dispatch that prevents useful compiler inlining/vectorization;
- unnecessary copies between packed resident weights and compute kernels;
- repeated routing transformations whose exact result was already derived earlier in the same pass.

Do not remove correctness/lifetime protection merely to shave calls. Separate a proven immutable-resident fast path from fallback admission logic if necessary.

### 4. Routing reuse must be re-tested in Ling's resident regime

QWEN36 exact-next commit `09e5bf13...` proved that when BATCH prefill has already derived bit-identical top-k selections, the row pass can consume those selections instead of recomputing softmax/group-selection/top-k.

That experiment was not a decisive QWEN36 production wall-clock win in its I/O-dominated regime, but Ling may be different because its resident path can expose routing/dispatch CPU work.

For Ling:

- never compute the same exact routing selection twice in one token/prefill pass without measuring a reason;
- preserve bit-identical semantics for exact mode;
- reuse already-computed route IDs/weights across admission and compute stages;
- benchmark routing reuse independently and end-to-end;
- if no end-to-end win remains, bank it as negative knowledge rather than retaining complexity by faith.

### 5. Batch acquisition becomes batch preparation in a resident engine

QWEN36 BATCH acquisition deduplicated expert demand and orchestrated bounded parallel loads before row execution.

For fully resident Ling, there may be no physical acquisition to batch, but the reusable idea remains:

- derive route sets once for a prefill block;
- deduplicate/organize work when this reduces dispatch;
- group expert work to improve cache locality and amortize setup;
- reuse routing results in compute;
- avoid turning a resident engine into an artificial cache scheduler.

For partial residency, retain the full reserve/dedupe/bounded-load semantics.

### 6. Fuse physical operations only when the physical layout earns it

QWEN36 `COLI_FUSED_LOAD` merged adjacent scale+weight reads when the container proved they were contiguous in the same file.

For Ling:

- if startup or fallback I/O remains, census actual physical layout first;
- fuse contiguous reads/copies only where byte adjacency is proven;
- consider packing metadata/weights so one physical operation feeds one expert/block where that improves startup or refill wall;
- do not invent a FUSED flag that performs no useful work in a fully resident steady state.

### 7. Concurrency is measured, never inherited

QWEN36 measured effective storage concurrency around 7–8 streams and falsified QD16 for that exact node/storage regime.

That number is **not** a Ling constant.

Measure independently on the Dell:

- OpenMP/thread count;
- resident expert parallelism;
- prefill parallelism;
- any startup/physical-load concurrency;
- affinity/scheduling where useful.

Search for the throughput knee and the regression point. Do not assume more threads/workers monotonically improve performance.

### 8. Cache capacity experiments do not transfer blindly

QWEN36's simple uniform cap increase was a negative result. For resident Ling, cache-cap tuning may be irrelevant entirely.

Therefore:

- first prove whether all routed weights can remain resident;
- if yes, delete cache-miss optimization from the steady-state critical path rather than tuning it forever;
- if no, use real per-layer working sets and byte budgets, not arbitrary uniform-cap escalation.

### 9. OS page cache is a measured tier, not magical RAM

If Ling requires fallback storage reads, distinguish:

- application resident hit;
- OS page-cache-served read;
- physical storage read.

Do not equate an application miss with NVMe I/O.

If Ling is fully resident, this becomes startup/fallback telemetry rather than decode policy.

### 10. Lease/lifetime correctness survives every optimization

Inkling/QWEN36 runtime history proved that a weight buffer must not change identity while compute holds it.

Resident fast paths may simplify this dramatically, but cannot weaken it.

For immutable permanently resident experts, pointer identity can itself be the lifetime guarantee. For evictable paths, leases/pins remain mandatory.

### 11. Exact and quality-changing lanes stay separate

Throughput tuning first targets exact execution:

- layout;
- dispatch;
- routing reuse;
- fusion;
- native quant kernels;
- thread scheduling;
- prefill organization;
- residency;
- memory locality.

Only after a strong exact baseline may Ling evaluate quality-changing levers such as routed-expert TOPP, pruning, expert merging, conditional execution, or more aggressive mixed precision.

Those require their own quality corpus, artifact identity, and promotion label.

### 12. Negative QWEN36 knowledge is part of the transfer

Do not casually repeat these QWEN36 dead ends:

- bigger uniform cache cap as a generic answer;
- deeper queue depth as a monotonic speed knob;
- theoretical storage-bandwidth subtraction masquerading as a wall-clock prediction;
- baseline-matching replay as proof of counterfactual policy validity;
- argmax-prune exact-runtime ideas already falsified in QWEN36;
- speculative/MTP work without actual model support.

A changed Ling regime may justify re-testing a mechanism only when the causal assumptions have changed and the discriminator is cheap.

## Maximum-throughput resident hot-path program

Once exact Ling parity is proven, perform a deliberate dispatch/compute census.

### A. Build the zero-I/O decode control

Target steady-state invariant where memory permits:

- all dense/core weights resident;
- all routed expert packed weights resident;
- direct `[layer][expert]` or equivalent stable pointer indexing;
- no decode-time tensor file reads;
- no decode-time tensor-name lookup;
- no decode-time allocation required by ordinary token execution;
- no cache eviction/admission on the resident route;
- exact same model outputs as the validated reference lane.

Measure and assert physical model-weight reads after warmup.

### B. Instrument dispatch cost explicitly

Add timers/counters that can isolate at least:

- KDA/MLA wall;
- router projection;
- route selection/group top-k;
- expert dispatch/setup;
- gate/up/down expert math;
- shared expert math;
- residual/norm/activation work;
- LM head;
- thread/parallel-region overhead where measurable;
- memcpy/packing/unpacking/conversion;
- miscellaneous residual token wall.

Keep instrumentation optional and low-overhead when disabled.

### C. Minimize expert-call overhead

After profiling, test coherent exact variants such as:

- one routed-expert work loop that consumes the already-selected top-8 without repeated generic lookup;
- stable direct resident pointers;
- grouped/fused gate+up traversal where the stored format and kernels permit it;
- fused or tightly coupled activation + down projection where mathematically exact and beneficial;
- batching selected expert setup before compute;
- reuse of scratch buffers across experts/layers/tokens;
- avoid barriers between independent work unless required;
- parallelize at the granularity that wins on the Dell rather than by aesthetic symmetry.

Do not call source-level fusion a win until end-to-end wall moves.

### D. Optimize native quant kernels for the actual Dell CPU

Discover the Dell CPU/ISA live.

Then tune the native packed path accordingly:

- compile flags/ISA dispatch;
- vectorized packed INT4 dot/matmul;
- activation quantization only if its cost is repaid;
- group-scale access locality;
- contiguous packed expert layout;
- prefetch/cache-line behavior where measured;
- thread count and affinity.

No generic x86 assumption is allowed to replace live CPU census.

### E. Prefill and decode are separate optimization problems

Report and optimize both separately.

Prefill may benefit from:

- larger matrix kernels;
- route-set preparation;
- grouped expert work;
- bounded parallelism;
- routing reuse.

Decode may benefit more from:

- direct pointers;
- tiny dispatch;
- persistent scratch;
- low synchronization;
- single-token specialized kernels;
- resident expert compute.

Do not let a large prefill throughput gain hide slower interactive decode.

## Research/application queue inherited from QWEN36

Every line below must be classified for Ling as `APPLY`, `DISCRIMINATE`, `DEFER`, `FALSIFIED_FOR_THIS_REGIME`, or `NOT_APPLICABLE` with evidence.

### Exact-runtime / infrastructure

- mixed low-bit precision by tensor/expert sensitivity;
- native packed low-bit execution;
- resident-vs-streamed byte accounting;
- in-flight load coalescing;
- FUSED contiguous I/O;
- BATCH request/acquisition preparation;
- routing-result reuse;
- expert parallelism;
- page-cache observability;
- hardware concurrency sweep;
- detailed admission/dispatch/compute timers;
- route/output parity and accounting conservation;
- deterministic trace/replay fixtures where they can answer a real policy question;
- promotion/rollback with source+artifact hashes.

### Model-shaping / quality-changing research

After exact throughput baseline only:

- progressive expert pruning;
- diversity-aware expert scoring;
- expert merging / partial-preservation merging;
- joint structural pruning + mixed precision;
- layer/expert sensitivity mapping;
- knowledge-distillation recovery if a smaller successor is later desired;
- routed-expert TOPP/conditional execution as an approximate serving lane;
- KV compression only if Ling's measured MLA/context state becomes a meaningful memory or bandwidth wall.

Do not perform these because they existed in a QWEN ledger. Rank them from Ling's measured bottleneck.

## Benchmark and promotion law

Every serious optimization must preserve an A/B packet with:

- exact source SHA;
- exact model/artifact/container hash;
- Dell hardware/OS/CPU/storage identity;
- precision map;
- thread/affinity/env configuration;
- prompt/corpus hash;
- prefill and decode token counts;
- cold/warm protocol;
- tok/s per run, not only an average;
- TTFT/prefill wall;
- decode tok/s and token wall;
- phase decomposition;
- RSS/resident bytes;
- logical expert requests;
- physical loads and bytes after warmup;
- correctness/token/logit/hidden-state discriminator;
- quality corpus when computation changes;
- explicit `PROMOTED`, `REJECTED`, `SUPERSEDED`, or `SCREENING_ONLY` result.

Maintain a frozen comparable Ling serving benchmark so the optimization ladder is real rather than a collection of incomparable anecdotes.

## Optimization ladder

The intended order is:

1. exact architecture/tensor parity;
2. native low-bit artifact correctness;
3. fully resident or maximum-residency control;
4. zero/near-zero decode-time model-weight I/O;
5. phase instrumentation;
6. dispatch removal and direct resident fast path;
7. routing reuse / prefill work reuse;
8. expert compute/kernel tuning;
9. thread/concurrency/affinity sweep;
10. prefill specialization;
11. residual bottleneck attack from measured profile;
12. only then quality-changing compression/conditional-compute research.

Repeat measurement after every meaningful step. The bottleneck is allowed to move.

## Success criterion

The final Ling path should not merely be "faster than QWEN36." That comparison is structurally expected from the smaller active model and different residency regime.

Success is:

**Ling 3.0 Tiny reaches a correctness-proven, low-bit, resident-oriented Colibri implementation whose remaining Dell token wall is measured, whose avoidable dispatch/orchestration overhead has been aggressively removed, and whose throughput is close to the practical ceiling of the actual CPU/memory architecture under the chosen quality contract.**

Before closeout answer:

> What did QWEN36 teach us that materially changed Ling's implementation, and what did Ling teach us that should flow back into QWEN36/Colibri after the pause ends?
