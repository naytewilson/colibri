# QWEN36 BATCH Wedge Post-Freeze Addendum

Date: 2026-08-24

## Status

The frozen QWEN36 control identity remains frozen. A later diagnostic lane proved a real liveness defect in the BATCH prefill pre-pass when the number of unique missing experts for a layer exceeds the number of currently reservable cache slots.

This addendum does not mutate the frozen artifact or retroactively rewrite its benchmark/quality receipts. It records a newly proven limitation and the required successor-hotfix process.

## Proven causal mechanism

Diagnostic branch:

`diag/qwen36-batch-wedge-20260823`

Diagnostic fix commit:

`a0f0fd127a2f51d41a81e4547ce8086009864ce9`

Starting diagnostic lineage:

`4579d41f6cba79ec7071e242470f3245ea32ba15`

The diagnostic evidence established that the two-phase BATCH pre-pass reserved unique misses before finishing/publishing any of them. Once every cache slot became an unpublished reservation (`eid == -1`), the next reservation entered `expert_acquire()` and waited for a publish that was sequenced after the reserve loop itself. This is a single-thread ordering self-deadlock, not a storage, cwd, allocator, OpenMP, or pilot-thread race.

The causal evidence included a native backtrace in `expert_acquire`, live cache state with all slots in-flight, a capacity lever (`cap <= 128` hangs while larger capacity passes for the captured prompt), and absence of storage reads during the wedge.

## Important controller review of the diagnostic patch

The causal mechanism is accepted, but `a0f0fd1` is not automatically a production promotion candidate.

Its drain condition is:

`if (m->cache[layer].n >= m->cache[layer].cap && nl > 0) flush`

`LCache.n` is an allocated-slot count and does not fall after the cache reaches capacity. Therefore, once a layer cache is full, this condition can flush after every subsequent reserved miss even when many published/evictable slots remain. That preserves liveness but may serialize the BATCH load tail or entire warm-cache miss sequence and destroy useful admission concurrency.

The statement that the diagnostic patch keeps the single-flush fast path unchanged whenever the unique set "fits in cache" is therefore too broad. It is only obviously true while the cache has not already reached its allocated-slot capacity.

A clean successor fix must drain only when another reservation would otherwise block because no immediately reservable/published slot remains, or use an equivalent non-blocking/WOULD_BLOCK reservation protocol. It must preserve the original parallel load shape when sufficient reservable slots exist.

## Bounded-wait diagnostic hardening

The diagnostic lineage also contains earlier bounded waits and a scratch-lease fallback introduced before the true mechanism was known. Those paths were not causal for this wedge. The scratch fallback deliberately leaks a private expert slot on timeout.

Do not promote that hardening into the frozen production lineage merely because it exists on the diagnostic branch. It requires separate lifetime/accounting justification or removal from the clean hotfix candidate.

## Clean hotfix lane

Controller-created branch:

`fix/qwen36-batch-wedge-clean-20260824`

Base:

`f04359aab31a388cc47d36e96c3ea36400061cb6`

This branch exists so the actual liveness repair can be implemented directly on the frozen production-source lineage without importing unrelated `exact-next` experiments, stale diagnostic theories, or the non-causal bounded-wait/scratch-lease changes.

## Required successor evidence

Before any deployment successor is promoted, require at least:

- deterministic reproduction of the old wedge on the frozen control;
- clean causal fix on the `f04359a` lineage;
- cap-below-working-set regression (for example cap64) that terminates and preserves output;
- warm full-cache discriminator proving BATCH does not collapse to one-at-a-time finishes merely because `LCache.n == cap`;
- W9/accounting conservation and output/routing parity;
- canonical FUSED/BATCH benchmark repeated under the frozen protocol to prove no material throughput regression;
- RSS bound;
- source/binary hashes and rollback receipt;
- independent verifier review before deployment.

## Control-freeze consequence

`QWEN36_FULLSIZE_CONTROL_FROZEN` remains an identity/measurement freeze for comparison. It must no longer be interpreted as a claim that the frozen BATCH artifact is liveness-safe for every prompt/cache-capacity combination.

If a clean successor is promoted, preserve the original frozen artifact as the historical control and record the successor as a distinct production artifact rather than rewriting history.

## Cross-model lesson

For Ling and future Colibri runtimes:

> Never reserve work from an exhaustible pool while sequencing all publishers after the code that can block waiting for those publishers.

Also require artifact identity for negative trace evidence: a missing marker is meaningful only when the trace is proven to come from the exact binary under adjudication.
