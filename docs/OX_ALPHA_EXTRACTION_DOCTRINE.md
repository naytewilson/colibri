# Ox Alpha Research-Extraction Doctrine

> Durable operating rule for high-capability agent usage across Colibri / Qwen / ANE-adjacent campaigns.

## Mission

Use Ox Alpha on the highest-leverage problems: architecture forks, repo-scale synthesis, source-truth archaeology, causal debugging, benchmark design, novel runtime/model research, and coherent multi-step Forge waves.

Do not spend its best context on rote edits that a cheaper worker can execute after the design is settled.

## Mandatory tasking posture

Each Ox Alpha wave should receive:

- the full mission boundary rather than a micro-patch;
- exact live source refs and evidence labels;
- owned and protected surfaces;
- explicit acceptance evidence and kill conditions;
- authority to inspect broadly, follow causal discoveries, make adjacent owned changes, add tests/fixtures/docs, and finish a coherent wave;
- a requirement to report exact refs, changed files, commands/results, residual uncertainty, and the next highest-value wave.

Do not force arbitrary stop points after individual functions/files when the remaining work is part of the same validated objective.

## Extraction requirement

Every valuable discovery must be harvested into at least one durable asset when appropriate:

- reusable implementation;
- regression or adversarial test;
- benchmark/discriminator;
- runtime/model contract;
- ledger/receipt entry;
- explicit falsified-path record;
- reusable agent prompt or checklist;
- architecture note tied to source evidence.

A useful conclusion that exists only in chat output is not fully banked.

## Novel/uncommon findings

When Ox Alpha encounters a potentially novel or under-documented ANE/runtime/model behavior:

1. state the observation separately from the hypothesis;
2. design the cheapest strong discriminator;
3. reproduce when feasible;
4. record exact hardware/toolchain/model/source provenance;
5. capture negative controls;
6. identify whether the finding generalizes or is regime-bound;
7. bank it in GitHub before moving on;
8. avoid claiming novelty until external literature/source comparison supports it.

The objective is to extract every defensible insight that can compound future work, especially findings that may not yet be common ANE/runtime practice, without turning speculation into folklore.

## Delegation hierarchy

Prefer Ox Alpha for:

- architectural decisions whose mistakes would cascade;
- research synthesis that changes implementation direction;
- hard source-truth reconciliation;
- cross-model/runtime generalization;
- experiment design where falsifiability matters;
- difficult correctness/performance investigations.

Prefer cheaper/faster workers after Ox Alpha has fixed the architecture for:

- mechanical refactors;
- repetitive migrations;
- formatting/cleanup;
- broad but straightforward test expansion;
- routine documentation propagation.

## Review discipline

Ox Alpha output is never accepted because it sounds sophisticated.

Controller review must still separate:

- PROVEN
- OBSERVED
- INFERRED
- UNKNOWN/BLOCKED
- FALSIFIED

Plans and battle reports should be reviewed for hidden coupling, accidental scope shrinkage, missing negative controls, non-comparable benchmarks, source-truth drift, and claims not backed by artifacts/tests.

## Platform memory doctrine

Any Ox Alpha wave involving RAM, cache capacity, model fit, context headroom, swap, page cache, accelerator memory, or residency must first identify the target operating system and hardware.

For macOS / Apple Silicon, Ox Alpha must follow `docs/MACOS_UNIFIED_MEMORY_DOCTRINE.md`.

In particular:

- preserve the established **50 GB internal-free-space preflight** as a swap/system-volume headroom admission law for the current Mac campaigns;
- if internal free disk is below that threshold, fail closed before a memory-aggressive live run rather than arguing from instantaneous RAM availability;
- treat the 50 GB threshold as an **experiment-admission guard**, not a claim that 50 GB of swap will be used and not a model-RAM sizing formula;
- after the run is admitted, reason from unified memory, memory pressure, compression, active swap/page churn, file-backed/cacheable pages, CPU/GPU/accelerator sharing, context growth, and sustained latency;
- do **not** import Windows-style `free RAM` budgeting as the primary admission law;
- distinguish `fits`, `runs`, `resident`, `pressure-stable`, and `fast` as separate claims;
- do not treat compression or swap as zero-cost extra capacity;
- do not infer hidden ANE/Core ML allocator behavior when it is not measurable;
- bank the exact workload and measurements used to justify any threshold change.

External model-storage space does not substitute for internal macOS swap/system-volume headroom. Satisfying the 50 GB gate also does not prove memory health; it only admits the experiment.

For Linux/Windows nodes, use those platforms' actual VM/cache/swap semantics instead. Platform truth outranks generic RAM folklore.

## Current Colibri application

For the current runtime Forge, Ox Alpha is authorized to reason at repo scale and should explicitly harvest:

- model-independent admission/residency primitives;
- cross-model cache correctness laws;
- storage-tier and page-cache behavior;
- dense-Qwen-compatible abstractions;
- negative findings from failed policies;
- reusable synthetic contract tests;
- promotion/rollback methodology.

Do not lock the shared runtime permanently to MoE-specific `(layer, expert)` identity if the same lower-level lifetime/tier machinery can support the planned dense `WeightStore` path.

## Standing rule

Before closing every Ox Alpha wave, ask:

> What did this wave learn that the next model, next hardware target, next ANE investigation, or cheaper worker should never have to rediscover?

Bank the answer.
