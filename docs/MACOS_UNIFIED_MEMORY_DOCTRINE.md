# macOS / Apple Silicon Unified-Memory Doctrine

> Platform-specific memory law for Colibri, dense-Qwen, ANE, Core ML, Metal, and local-model campaigns on Apple Silicon.
>
> **Do not import Windows-style RAM budgeting assumptions into macOS.** Memory capacity decisions must be based on the actual Apple unified-memory system and measured workload behavior.

## 1. Core rule

On Apple Silicon, physical memory is a **unified pool** shared by CPU workloads, GPU allocations, system services, caches, compressed memory, and accelerator-related runtime state.

Therefore this arithmetic is not an authoritative admission law:

`total RAM - currently used RAM = safe model bytes`

Nor is `free RAM` alone a useful success metric.

The useful question is:

> Can this exact workload sustain its intended context, generation length, accelerator use, and concurrency with stable memory pressure and acceptable latency, without pathological compression, swap churn, paging, or eviction of useful model state?

## 2. macOS memory states that matter

Capture and distinguish when feasible:

- total physical unified memory;
- application-resident memory;
- wired/non-pageable memory;
- compressed memory and compressor growth;
- file-backed/cacheable pages;
- purgeable/reclaimable state when observable;
- swap usage **and swap-in/swap-out activity**, not just the current swap-file size;
- page faults / paging behavior when useful;
- memory-pressure state/trend;
- GPU/Metal memory use or working-set information when available;
- model/runtime RSS or equivalent process-level footprint;
- KV/cache/workspace growth with context;
- accelerator-related allocations when measurable.

A large file cache is not automatically "wasted RAM". Cached/file-backed pages may be useful and reclaimable. Conversely, a process fitting below physical RAM does not prove it is healthy if the system is compressing or swapping aggressively under sustained decode.

## 3. Unified memory is not free extra capacity

Unified memory removes a separate CPU-vs-GPU VRAM pool and can avoid copies, which is valuable.

It does **not** mean:

- CPU RAM and GPU RAM should be budgeted independently and added together;
- the entire advertised physical-memory capacity is safely available to model weights;
- compression/swap can be treated as zero-cost capacity;
- a model that launches once is resident-fit.

CPU, GPU, system, model weights, KV, workspaces, caches, and other allocations compete for one physical pool.

## 4. Compression and swap doctrine

macOS memory compression is a normal operating mechanism. Its mere presence does not imply failure.

Likewise, some swap usage can exist on an otherwise usable machine.

But for latency-sensitive inference, classify **behavior**, not ideology:

- stable bounded compression with no material latency regression may be acceptable;
- static historical swap allocation is not itself evidence of active thrashing;
- growing compression, repeated swap-in/out, major page churn, or decode stalls correlated with pressure are evidence that the working set is too aggressive;
- sustained model-weight paging during decode must be measured as an execution cost, not hidden behind "macOS manages memory automatically".

Do not call a model resident-fit merely because macOS prevented OOM by compressing or swapping it.

## 5. Admission decision for a model/runtime configuration

Prefer a **measured pressure envelope** over a fixed Windows-style free-RAM threshold.

For each candidate, test at minimum:

1. cold launch;
2. warm steady state;
3. intended prompt/context size;
4. intended generation length;
5. repeated turns / soak behavior;
6. intended accelerator configuration;
7. intended concurrency if greater than one request.

Record memory and latency throughout the run, not only at process start or exit.

A candidate is healthy only if its pressure/latency behavior remains bounded and reproducible across the intended workload.

## 6. Suggested evidence classes

### `APPLE_MEMORY_RESIDENT_HEALTHY`

- intended workload completes repeatedly;
- memory pressure remains healthy/stable;
- compression is bounded;
- no material active swap churn attributable to inference;
- no repeated weight paging that dominates token latency;
- context/KV growth leaves reproducible headroom.

### `APPLE_MEMORY_PRESSURE_TOLERABLE`

- workload completes and latency remains acceptable;
- measurable compression or limited paging occurs;
- behavior is bounded and does not worsen across a soak run;
- explicitly document the trade.

### `APPLE_MEMORY_SWAP_BOUND`

- active swap/page churn materially contributes to TTFT/decode latency;
- working set no longer behaves like resident inference;
- treat storage as an execution tier and optimize or reduce the model accordingly.

### `APPLE_MEMORY_UNSTABLE`

- pressure escalates across turns;
- latency collapses or allocation failures occur;
- memory state is non-reproducible or unsafe for the intended service.

## 7. Dense-Qwen consequence

For dense Qwen on Apple Silicon, prefer this hierarchy:

1. native low-bit **fully resident** execution with sufficient KV/workspace/system headroom;
2. mostly resident execution with a deliberately measured small spill/window;
3. out-of-core/windowed execution only when storage arithmetic and measured pressure support it.

Do not set the dense target from parameter count or file size alone.

The real budget includes:

`low-bit weights + quant metadata + KV + activations/workspaces + tokenizer/runtime + GPU/accelerator allocations + system pressure headroom`

and is validated by the pressure envelope above.

## 8. ANE / Core ML consequence

ANE/Core ML campaigns must not assume that accelerator execution means model memory is outside the unified-memory budget.

Where allocator behavior is opaque, label it `UNKNOWN` rather than pretending to know the exact split.

Use process/system memory observations, package/load behavior, repeated execution, and pressure changes as evidence. Avoid inventing hidden ANE memory accounting.

## 9. Cross-platform rule

This doctrine applies to macOS / Apple Silicon targets.

For Linux or Windows nodes, use that platform's actual VM/page-cache/swap/accelerator behavior instead.

Do not universalize macOS heuristics either.

**Platform truth outranks generic RAM folklore.**

## 10. Ox Alpha instruction

When a Colibri/Qwen/ANE design question involves RAM, residency, cache size, model fit, context headroom, swap, or accelerator memory:

1. identify the target OS/hardware first;
2. use platform-native memory semantics;
3. on Apple Silicon, reason from unified memory + measured memory pressure, compression, swap activity, cache behavior, accelerator sharing, and sustained latency;
4. reject Windows-style `free RAM` budgeting as the sole admission criterion;
5. distinguish `fits`, `runs`, `resident`, `pressure-stable`, and `fast` as separate claims;
6. bank the measurements and exact workload used to justify any threshold.
