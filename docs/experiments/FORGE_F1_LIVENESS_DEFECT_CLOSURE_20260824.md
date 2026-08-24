# FORGE F1 — Liveness & Lease Defect Closure (Lane A, 2026-08-24)

**Branch**: `forge/f1-ooc-moe-runtime` (started at verifier-confirmed
`486f557dc8e1c4bfb10ff7382dcb7c10a7602cd7`)
**Verifier evidence**: `verify/forge-f1-20260824 @ 5689298b89eb6bfe5d517262e6fd0088b59c6882`,
receipt `docs/experiments/VERIFIER_RECEIPT_FORGE_F1_20260824.md` (read from the
verifier branch; never merged).
**Scope guard**: no merge, no deploy, frozen QWEN36 control `f04359a` untouched,
verifier branch untouched, OLMoE real E2E left to Lane B.

## 1. F1-LIVE-1 — root cause

`c/expert_backend_pread.c` `pbs_pick_victim` stage 3 spun forever when no
RESIDENT victim existed (every slot `PBS_RESERVED`), rescanning only
RESIDENT slots. A concurrent abort that freed a slot (`PBS_FREE`) was never
observed by the spinner: pool-full was treated as permanently-no-capacity.
Engine callers held `g_xorder_mx` across `coli_expert_reserve`
(qwen36.c demand + PILOT paths), while publication needed the same mutex —
two threads racing reserves into an exhausted pool deadlocked. Verifier
reproducer: probe facet A ("FAIL H: deadlock timeout", exit 42).

## 2. Falsified prior coverage claim

Operating-laws §4 claimed `tests/test_expert_backend_pread.c (all-reserved
spin drain)` covered the law — grep-proven NO such test existed anywhere.
The claim is retracted in `forge_f1_operating_laws.md`; real coverage now
lives in `c/tests/test_pread_liveness.c`.

## 3. Final liveness architecture (generic backend)

Claim search is non-blocking and complete per attempt:
1. reject duplicate identity (reserving/resident → BUSY);
2. scan existing FREE buffers first;
3. grow while `n < cap`;
4. choose legal RESIDENT victim (stage 1 unpinned LRU, stage 2 any LRU;
   RESERVED buffers are never stolen);
5. every slot RESERVED → explicit transient result instead of spinning:
   `pbs_claim` reports saturation, `pbs_reserve` returns
   `COLI_EXPERT_ERR_SATURATED` (new additive code −6; −5 reserved for
   admission load failures). Governing laws hold: FREE-after-abort becomes
   claimable and RESIDENT-after-publish permits progress on the very next
   retry, because retry re-runs the FULL search after wake. Advisory
   prefetch skips under saturation (holds no lease, must not contend with
   demand). `would_evict` −2 semantics unchanged (preview of "claim would
   report transient saturation").

## 4. QWEN36 lock-order repair

`expert_acquire` (demand): reserve() moved OUTSIDE `g_xorder_mx`. BUSY and
SATURATED share one bounded drain-wait loop — sleep unlocked, recheck
residency under the lock (serve as hit if a publisher landed), retry the
reserve outside it. `pilot_realload`: probe under lock, reserve outside;
SATURATED drops the speculation (speculative prefetch yields to demand).
Loads were already outside all ordering locks; publish stays under
`g_xorder_mx` with trace emission (publish itself never blocks).

## 5. Engine-level saturation discriminator

`g_demand_coalesce_waits` counts demand acquisitions that entered the
BUSY/SATURATED drain-wait; PILOT keeps `g_pilot_coalesce_skips` for BUSY.
SATURATED on the PILOT path is silent-by-design drop (speculation policy),
visible indirectly via store stats (`aborts`/`publishes` deltas). Generic
admission surfaces SATURATED through `coli_admission_acquire`'s coalescing
window (same budget/valve as BUSY).

## 6. OLMoE lock-order audit — REPAIRED

Same hazard pattern found in `olmoe.c` (`expert_get` held `g_xorder_mx`
across reserve attempts incl. the coalesce loop; its own `pilot_realload`
held it across reserve). Applied the identical repair (reserve outside the
engine lock; BUSY/SATURATED drain-wait with residency recheck; goto-under-
lock removed). Classification: **REPAIRED** (source-level; real OLMoE E2E
remains Lane B's scope).

## 7. acquire_batch hidden lease leak — fixed

Old leasing phase performed fresh `coli_expert_lookup` per caller entry and
never released/transferred the leases already held by successful jobs'
views → one leaked lease per unique key per batch (slot lease_count stuck,
destroy contract violated). New ownership rule: each successful unique
job's lease TRANSFERS to its first caller occurrence (internal view cleared
after transfer); duplicate entries take their own fresh lookups; failed
jobs have nothing to transfer; an OOM fallback releases job leases instead
of leaking. Header contract updated.

## 8. Destroy contract resolution

Docs said "debug builds assert"; implementation printed and destroyed
anyway. One truthful contract now: destroy requires zero active reservations
AND zero active leases; debug builds assert BEFORE freeing
(`pbs_destroy`); release builds retain the documented caller precondition
(no ABI change). Direct isolated test: fork probe in
`test_pread_liveness.c` — child holds a lease across destroy and must NOT
exit cleanly (skipped loudly under NDEBUG/release builds).

## 9. Promotion gate self-containment — repaired

`promote.sh` invoked `c/qwen36_trace_replay` without building it (fresh
clone ⇒ PROMGATE_RED until manually built; builder GREEN depended on a
stale untracked binary). The build step now compiles
`qwen36 olmoe qwen36_trace_replay` before any gate runs. Fresh-DUMP
fail-closed parity checks untouched.

## 10. Durable tests added

`c/tests/test_pread_liveness.c` (Makefile-gated, real safetensors fixtures,
real pread backend):
- H1 ABORT→FREE ×100: fully-RESERVED tiny pool (2 slots), bounded waiter,
  owner aborts, waiter's next retry claims the freed slot; classification
  (SATURATED observed) asserted; counters balance (aborts=3, publishes=0,
  active=0).
- H2 PUBLISH→PROGRESS ×100: owner publishes, waiter legally evicts the
  RESIDENT copy (A miss afterwards), other owner's RESERVED buffer still
  publishes (no theft); counters balance (publishes=2, aborts=1).
- Batch lease transfer on the REAL backend: unique+duplicate keys (loads==2
  for 4 entries), mixed success/failure injection; zero leases survive
  release (debug destroy asserts enforce it).
- Destroy-contract fork probe (debug builds).

## 11. Validation record (this worktree, macOS arm64)

| Check | Result |
|---|---|
| make test-c (TEST_EXCLUDE test_uring test_qwen36_i4_kernel) | rc=0, 78 suites ok |
| H1 | PASS ×100 |
| H2 | PASS ×100 |
| ASan+UBSan (liveness/pread/admission/reservation/store-ops) | clean, rc=0 |
| TSAN liveness ×100+100 | clean after atomicizing the test's poll flags |
| engines build qwen36/olmoe/qwen36_trace_replay | ok |
| promote.sh clean-checkout cycle | see receipt section below |

## 12. Targeted reverification pointers (Lane C)

Re-run from the verifier receipt: gate B (promote.sh self-contained green ×2),
gate C (old-gate discriminator still fails closed), full-pool liveness facets
(now H1/H2 above), five-arm tiny parity, trace semantic gate, replay
self-consistency, accounting conservation, sanitizers, defaults-off arm.
OLMoE real E2E remains BLOCKED-by-scope (no canonical container) — not part
of this lane's completion criteria.

## Verdict

`FORGE_F1_LIVENESS_AND_LEASE_DEFECTS_FIXED_READY_FOR_TARGETED_REVERIFICATION`
