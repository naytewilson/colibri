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

## 5. Project swap-headroom admission law

For the current ANVIL / Colibri / ANE Mac campaigns, **internal-disk free space is an independent preflight gate because it is the practical headroom for macOS swap and other temporary system allocations under memory pressure.**

Established operational rule:

`internal free disk < 50 GB -> block memory-aggressive live runs`

`internal free disk >= 50 GB -> storage/swap admission satisfied; memory health must still be measured during the run`

This 50 GB rule is **not** a RAM-sizing formula and does not imply that macOS will consume 50 GB of swap. It is a conservative campaign admission law derived from the failure mode repeatedly encountered in practice: memory-heavy model/ANE runs become unsafe or non-reproducible when the internal system volume lacks enough room for swap growth and system working headroom.

Do not bypass, lower, or reinterpret this guard merely because:

- Activity Monitor shows nominally available memory;
- the model fits at launch;
- compression is currently low;
- an external volume has ample free space;
- the expected steady-state RSS appears below physical RAM.

macOS swap is system-managed on the internal/system volume; external model-storage capacity is not a substitute for internal swap headroom.

Conversely, satisfying the 50 GB gate does **not** prove the model/runtime configuration is healthy. It only admits the experiment. Sustained pressure, compression, swap activity, paging, latency, context growth, and accelerator sharing still determine whether the configuration is viable.

If a future campaign has strong repeated evidence supporting a different threshold on a different machine/OS/toolchain, record that as a new platform-specific law with provenance rather than silently weakening this one.

## 6. Admission decision for a model/runtime configuration

Use two separate gates:

### Gate A — storage/swap admission

- require the established project-specific internal-free-space threshold (currently 50 GB on the relevant Mac campaigns);
- fail closed before a heavy live run if the threshold is not met.

### Gate B — measured unified-memory health

For an admitted candidate, test at minimum:

1. cold launch;
2. warm steady state;
3. intended prompt/context size;
4. intended generation length;
5. repeated turns / soak behavior;
6. intended accelerator configuration;
7. intended concurrency if greater than one request.

Record memory and latency throughout the run, not only at process start or exit.

A candidate is healthy only if its pressure/latency behavior remains bounded and reproducible across the intended workload.

## 7. Suggested evidence classes

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

### `APPLE_SWAP_HEADROOM_BLOCKED`

- internal free space is below the established campaign admission threshold;
- live memory-aggressive execution is not admitted regardless of apparent instantaneous RAM headroom;
- no performance or placement conclusion may be drawn from a skipped live run.

## 8. Dense-Qwen consequence

For dense Qwen on Apple Silicon, prefer this hierarchy:

1. native low-bit **fully resident** execution with sufficient KV/workspace/system headroom;
2. mostly resident execution with a deliberately measured small spill/window;
3. out-of-core/windowed execution only when storage arithmetic and measured pressure support it.

Before any heavy dense-Qwen live run, satisfy the project swap-headroom gate first.

Do not set the dense target from parameter count or file size alone.

The real budget includes:

`low-bit weights + quant metadata + KV + activations/workspaces + tokenizer/runtime + GPU/accelerator allocations + system pressure headroom`

and is validated by the pressure envelope above.

## 9. ANE / Core ML consequence

ANE/Core ML campaigns must not assume that accelerator execution means model memory is outside the unified-memory budget.

Where allocator behavior is opaque, label it `UNKNOWN` rather than pretending to know the exact split.

Use process/system memory observations, package/load behavior, repeated execution, and pressure changes as evidence. Avoid inventing hidden ANE memory accounting.

The established internal-free-space gate remains in force before memory-heavy live ANE/Core ML experiments because swap/system-volume exhaustion is an execution blocker independent of ANE placement.

## 10. Cross-platform rule

This doctrine applies to macOS / Apple Silicon targets.

For Linux or Windows nodes, use that platform's actual VM/page-cache/swap/accelerator behavior instead.

Do not universalize macOS heuristics either.

**Platform truth outranks generic RAM folklore.**

## 11. Ox Alpha instruction

When a Colibri/Qwen/ANE design question involves RAM, residency, cache size, model fit, context headroom, swap, or accelerator memory:

1. identify the target OS/hardware first;
2. use platform-native memory semantics;
3. on Apple Silicon, preserve the established 50 GB internal-free-space swap-headroom preflight unless live campaign source truth explicitly supersedes it;
4. treat that 50 GB threshold as an experiment-admission guard, **not** as a RAM model or prediction of swap use;
5. after admission, reason from unified memory + measured memory pressure, compression, swap activity, cache behavior, accelerator sharing, and sustained latency;
6. reject Windows-style `free RAM` budgeting as the sole admission criterion;
7. distinguish `fits`, `runs`, `resident`, `pressure-stable`, and `fast` as separate claims;
8. bank the measurements and exact workload used to justify any threshold change.
