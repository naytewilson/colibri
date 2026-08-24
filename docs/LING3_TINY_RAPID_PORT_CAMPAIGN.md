# Ling 3.0 Tiny Rapid Colibri Port Campaign

## Mission

Temporarily pause new QWEN36 optimization work and test a faster local-serving regime on the Dell target by porting `inclusionAI/Ling-3.0-tiny` into Colibri.

The point is not to replace the frozen QWEN36 control. Preserve QWEN36 artifacts, refs, deployment state, and receipts so that campaign can resume unchanged.

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

Get a correct native-C Ling 3.0 Tiny decode path on the Dell as quickly as possible, then measure the resident-model ceiling.

### Wave 0: live census

- inspect the official snapshot/config/tokenizer and exact tensor names/shapes/dtypes;
- inspect `c/kimi_k3.c`, `c/inkling.c`, `c/colibri.c`, `c/st.h`, `c/tok.h`, `c/quant.h`, and build machinery;
- discover the Dell's current RAM/CPU/storage state from the machine itself;
- do not infer the target from the development host.

### Wave 1: exact tiny/oracle path

- create a dedicated `c/ling3.c` or equivalent clean model adapter/engine;
- implement config parsing and tensor lookup from the official snapshot;
- port KDA and MLA math under Ling's actual config;
- implement BailingMoeV3 dense layer + sigmoid grouped top-k MoE + shared expert;
- add raw-id inference and deterministic trace/logit hooks;
- validate against a small reference fixture before optimizing.

### Wave 2: resident low-bit path

- consume the official INT4 checkpoint directly if its compressed-tensors layout is compatible with a clean native path;
- otherwise perform the smallest deterministic one-time conversion needed for Colibri's native packed format;
- keep low-bit weights low-bit through compute where possible;
- prefer full residency over expert streaming for this model when measured memory allows it.

### Wave 3: Dell performance

Measure on the actual Dell target:

- cold start
- prompt processing
- decode tok/s
- RSS / resident footprint
- routed expert hit/load behavior, which ideally collapses to no decode-time physical weight reads in the fully resident path
- short and sustained generation
- context growth
- CPU utilization and thread scaling

Do not project performance from other runtimes or machines.

## Acceptance gates

A rapid-port victory requires:

1. governing model/config/tensor census banked;
2. deterministic reference parity on a small fixture or equivalent strong logit/hidden-state discriminator;
3. native Colibri build on the Linux target with no new Python runtime dependency;
4. official or deterministically converted low-bit artifact with exact provenance/hash;
5. reproducible Dell decode benchmark;
6. QWEN36 production/control state left untouched and resumable.

## Anti-drift rules

- Do not use llama.cpp as the implementation path or performance authority.
- Do not rewrite K3 into Ling by leaving K3-specific AttnRes/LatentMoE semantics in place.
- Do not add disk streaming because Colibri historically streams experts. First measure whether Ling can be fully resident.
- Do not assume HP/Mac memory behavior describes the Dell target.
- Do not introduce Python as a runtime dependency on the Linux compute node.
- Do not claim speed until measured on the Dell.

## Expected high-value finding

This campaign is a controlled regime change from QWEN36's storage-virtualized 35B-class MoE to a 7.9B-total / 1.3B-active hybrid MoE that may be resident on the Dell. It should tell us how much performance Colibri gains when decode stops paying the large expert-storage residency tax.

Before closeout, record what this port teaches that KDA/MLA/MoE ports should never need to rediscover.
