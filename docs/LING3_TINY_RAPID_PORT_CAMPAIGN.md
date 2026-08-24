# Ling 3.0 Tiny Rapid Colibri Port Campaign

## Mission

Temporarily pause new QWEN36 optimization work and test a faster local-serving regime on the Dell target by porting `inclusionAI/Ling-3.0-tiny` into Colibri.

The point is not to replace the frozen QWEN36 control. Preserve QWEN36 artifacts, refs, deployment state, and receipts so that campaign can resume unchanged.

This is not only a bring-up campaign. After correctness, Ling must be tuned with the same evidence-driven intensity used on QWEN36: remove avoidable dispatch/orchestration, reuse exact work, keep low-bit weights native, collapse decode-time physical model-weight I/O where resident fit allows it, and chase the measured end-to-end token wall until the Dell reaches a practical throughput ceiling.

## Mandatory throughput doctrine

Before optimization, read and follow:

`docs/LING3_TINY_QWEN_TRANSFER_OPTIMIZATION_DOCTRINE.md`

That doctrine imports all QWEN36 runtime/research lessons that may transfer and requires each to be explicitly classified for Ling as `APPLY`, `DISCRIMINATE`, `DEFER`, `FALSIFIED_FOR_THIS_REGIME`, or `NOT_APPLICABLE`.

The worker must also inspect the durable QWEN36 campaign/research/runtime ledgers rather than reconstructing those lessons from memory.

## Current source truth to verify at worker startup

Repository: `naytewilson/colibri`

Campaign branch: `feat/ling3-tiny-colibri`

This branch was created from live `main` on 2026-08-23. Workers must fetch and verify its actual HEAD before editing.

## Authoritative model facts

From the current InclusionAI model/config:

- model type: `bailing_hybrid` / `BailingMoeV3ForCausalLM`
- total parameters: 7.9B
- activated parameters: 1.3B per token
- 24 layers total: 18 KDA + 6 gated MLA in a 3:1 pattern
- hidden size: 1536
- heads: 16
- KDA / attention head dim: 128
- first dense layer count: 1
- 128 routed experts
- top-8 routed experts per token
- 1 shared expert
- routed expert intermediate size: 512
- dense intermediate size: 4608
- q LoRA rank: 256
- kv LoRA rank: 512
- MLA qk dimensions: 128 no-RoPE + 64 RoPE
- MLA value dim: 128
- short convolution kernel size: 4
- RMSNorm epsilon: 1e-6
- sigmoid router with grouped top-k selection and expert bias
- official BF16, FP8, and INT4 checkpoints are provided

Do not substitute third-party conversions as the governing source for architecture or tensor names.

## Why this may be a short port

Colibri `main` already contains a pure-C Kimi K3 engine with:

- Kimi Delta Attention recurrence
- short causal conv4 on q/k/v
- gated per-head output RMSNorm
- gated MLA
- recurrent KDA state
- MLA latent cache machinery
- native low-bit resident-weight kernels and common safetensors/tokenizer support

That is unusually close to Ling's hybrid attention stack.

However, K3 is NOT a drop-in Ling implementation. Important differences include:

- K3 uses NoPE MLA while Ling uses partial RoPE in MLA
- K3 uses AttnRes; Ling's governing config does not establish that same residual architecture
- K3 uses Stable LatentMoE; Ling uses BailingMoeV3 routed/shared expert MLPs
- K3 expert counts, top-k, dimensions, and quantization format differ materially

Therefore the rapid path is **mechanism composition**, not copying `kimi_k3.c` wholesale:

1. reuse/adapt K3 KDA math and recurrent-state machinery;
2. adapt K3 MLA to Ling's q/kv low-rank and partial-RoPE contract;
3. reuse the existing standard routed/shared-expert machinery and low-bit kernels from the closest Colibri engine rather than K3's LatentMoE;
4. keep the whole useful model resident when the actual Dell memory census permits it;
5. only add streaming/cache machinery if measured resident fit requires it.

## Main Quest

Get a correct native-C Ling 3.0 Tiny decode path on the Dell as quickly as possible, then drive the exact low-bit resident path toward maximum measured token throughput.

### Wave 0: live census

- inspect the official snapshot/config/tokenizer and exact tensor names/shapes/dtypes;
- inspect `c/kimi_k3.c`, `c/inkling.c`, `c/olmoe.c`, `c/qwen36.c`, `c/colibri.c`, `c/st.h`, `c/tok.h`, `c/quant.h`, and build machinery;
- inspect QWEN36 campaign/research/runtime ledgers and the Ling transfer doctrine;
- discover the Dell's current RAM/CPU/ISA/storage state from the machine itself;
- do not infer the target from the development host;
- produce a QWEN36-to-Ling transfer matrix before inventing new optimization machinery.

### Wave 1: exact tiny/oracle path

- create a dedicated `c/ling3.c` or equivalent clean model adapter/engine;
- implement config parsing and tensor lookup from the official snapshot;
- port KDA and MLA math under Ling's actual config;
- implement BailingMoeV3 dense layer + sigmoid grouped top-k MoE + shared expert;
- add raw-id inference and deterministic trace/logit hooks;
- validate against a small reference fixture before optimizing;
- preserve a slow/clear correctness path if useful as an oracle for later fast-path work.

### Wave 2: resident low-bit path

- consume the official INT4 checkpoint directly if its compressed-tensors layout is compatible with a clean native path;
- otherwise perform the smallest deterministic one-time conversion needed for Colibri's native packed format;
- keep low-bit weights low-bit through compute where possible;
- pre-resolve tensor metadata/pointers at startup;
- prefer full residency over expert streaming for this model when measured memory allows it;
- if fully resident, establish a steady-state decode invariant with zero/near-zero model-weight physical reads after warmup;
- build direct immutable resident expert/tensor indexing rather than paying cache/admission machinery on every hit when that machinery is no longer needed.

### Wave 3: Dell baseline + phase decomposition

Measure on the actual Dell target:

- cold start
- prompt processing / TTFT
- decode tok/s
- RSS / resident footprint
- routed expert logical requests vs physical loads
- model-weight physical bytes after warmup
- short and sustained generation
- context growth
- CPU utilization
- thread count / affinity sweep
- KDA/MLA wall
- routing projection + grouped top-k wall
- expert dispatch/setup wall
- routed expert compute wall
- shared expert wall
- LM head wall
- memcpy/packing/conversion wall
- residual/unattributed token wall

Do not project performance from other runtimes or machines.

### Wave 4: dispatch + exact-throughput attack

Follow the QWEN36 transfer doctrine and attack the largest measured wall.

At minimum discriminate:

- direct resident expert pointers vs generic lookup/cache-hit path;
- removal of hot-path mutex/lookup/bookkeeping that is redundant under immutable full residency;
- persistent scratch/workspaces vs per-token/per-layer allocation;
- route-result reuse so admission/preparation and compute never derive identical grouped top-k twice;
- grouped/batched top-8 expert setup;
- gate/up/down fusion or tighter traversal where mathematically exact and supported by the packed layout;
- activation/down-projection coupling where exact and measurable;
- packed INT4 kernel/vectorization improvements for the Dell's actual ISA;
- persistent OpenMP/thread-team and scheduling/affinity choices;
- prefill-specific route/work preparation distinct from decode specialization;
- contiguous/fused startup/refill operations only if physical layout and residual I/O make them relevant.

Do not retain an optimization because it sounds sophisticated. It must move end-to-end throughput or clearly remove a measured prerequisite wall.

### Wave 5: residual frontier

Once the obvious resident/dispatch/kernel walls are exhausted:

- re-profile;
- identify the new dominant component;
- continue exact-runtime optimization while wins remain coherent;
- only then open quality-changing research such as mixed sensitivity precision, routed TOPP, expert pruning/merging, joint pruning+mixed precision, or other QWEN36 research lines whose causal assumptions fit Ling;
- KV compression is earned only if Ling's measured context/KV state is actually a meaningful wall.

## QWEN36 transfer requirements

The Ling campaign must carry forward, adapt, or explicitly reject with evidence:

- native low-bit packed execution;
- exact physical-byte accounting;
- duplicate in-flight load coalescing if any lazy/streamed path survives;
- FUSED contiguous I/O where actual layout makes it useful;
- BATCH route/request preparation where it reduces prefill dispatch or physical admission;
- routing-result reuse from the W3a exact-next lesson;
- expert parallelism and worker-count sweeps;
- measured concurrency knees rather than inherited queue-depth constants;
- OS page-cache observability for any fallback storage path;
- lease/lifetime correctness;
- detailed phase telemetry;
- deterministic route/output parity and accounting conservation;
- frozen comparable benchmark protocol;
- negative knowledge from QWEN36 so dead ends are not casually repeated;
- source/artifact hashes, promotion state, and rollback discipline.

QWEN36 hardware-specific numbers are evidence about mechanisms, not Ling tuning constants.

## Acceptance gates

A rapid-port victory requires:

1. governing model/config/tensor census banked;
2. deterministic reference parity on a small fixture or equivalent strong logit/hidden-state discriminator;
3. native Colibri build on the Linux target with no new Python runtime dependency;
4. official or deterministically converted low-bit artifact with exact provenance/hash;
5. reproducible Dell decode benchmark;
6. QWEN36-to-Ling research transfer matrix completed;
7. physical model-weight reads after warmup measured and driven to zero/near-zero if full residency is feasible;
8. dispatch/compute phase decomposition banked;
9. at least one deliberate exact-throughput optimization wave after the first working baseline;
10. final exact path benchmarked under a frozen cold/warm protocol;
11. QWEN36 production/control state left untouched and resumable.

## Anti-drift rules

- Do not use llama.cpp as the implementation path or performance authority.
- Do not rewrite K3 into Ling by leaving K3-specific AttnRes/LatentMoE semantics in place.
- Do not add disk streaming because Colibri historically streams experts. First measure whether Ling can be fully resident.
- Do not keep generic cache/admission dispatch in a proven fully resident hot path merely for architectural symmetry.
- Do not assume HP/Mac memory behavior describes the Dell target.
- Do not inherit QWEN36's measured worker count, queue depth, cache caps, or storage timings as Ling constants.
- Do not introduce Python as a runtime dependency on the Linux compute node.
- Do not claim speed until measured on the Dell.
- Do not optimize against an unfrozen benchmark while reporting percentage wins.
- Do not mix exact-runtime wins with quality-changing approximations.

## Expected high-value finding

This campaign is a controlled regime change from QWEN36's storage-virtualized 35B-class MoE to a 7.9B-total / 1.3B-active hybrid MoE that may be resident on the Dell. It should tell us both:

1. how much performance Colibri gains when decode stops paying the large expert-storage residency tax; and
2. what the next bottleneck becomes when storage is no longer allowed to hide routing, dispatch, synchronization, kernel, and memory-bandwidth costs.

Before closeout, record:

- what QWEN36 taught us that materially changed Ling's implementation;
- what QWEN36 research did not transfer and why;
- what Ling taught us that should flow back into QWEN36/Colibri when the pause ends;
- what future KDA/MLA/MoE ports should never have to rediscover.
