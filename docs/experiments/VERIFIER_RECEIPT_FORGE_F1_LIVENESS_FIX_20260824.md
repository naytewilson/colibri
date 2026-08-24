# FORGE F1 — TARGETED INDEPENDENT REVERIFIER RECEIPT (LIVENESS FIX, 2026-08-24)

**Verifier**: fresh independent targeted verifier (ox-alpha / Kilo), Phase-1 packet
consumed (ANVIL handoff #55809), no builder state reused
**Candidate**: `forge/f1-ooc-moe-runtime` @ `0b789e06eb6448cd16bf904b23cc66e565f44295`
**Lineage**: `f04359a` (frozen baseline) → … → `486f557` (verifier-confirmed defective
candidate) → `f75719d` (backend liveness) → `b1e0709` (engine lock-order) → `b7d3faf`
(admission lease/tests) → `f25d3c6` (promgate/docs) → `0b789e0` (real-container evidence).
Ancestry `486f557 ∈ 0b789e0` PROVEN (`git merge-base --is-ancestor`).
**Prior evidence**: `verify/forge-f1-20260824 @ 5689298` (consumed, never merged).
**Method**: fresh clone `/var/folders/.../T/kilo/f1-reverify/colibri` (clean at 0b789e0),
verifier-owned probes compiled from repo sources outside the repo, node-02 fresh clone
for Stage C. Candidate branch never mutated; no merge; no deploy.

## Census (486f557..0b789e0)

13 files, +841/−107 — exactly the controller's list; no unexpected surfaces:
`c/Makefile` (+3, liveness-test rule) · `c/admission.{c,h}` (lease transfer +
SATURATED coalescing) · `c/expert_backend_pread.{c,h}` (F1-LIVE-1 fix) ·
`c/expert_store.h` (ERR_SATURATED contract + destroy-contract wording) ·
`c/olmoe.c` + `c/qwen36.c` (lock-order) · `c/tests/test_pread_liveness.c` (new,
421 lines) · `docs/experiments/{promote.sh, FORGE_F1_LIVENESS_DEFECT_CLOSURE…,
forge_f1_handoff_stageA.md, forge_f1_operating_laws.md}`.

## Gate results

| Gate | Check | Expected | Observed | Verdict |
|---|---|---|---|---|
| A | origin == repair SHA | 0b789e0 | exact | PASS |
| A | 486f557 ancestor | yes | yes | PASS |
| A | clean checkout | 0 dirty | 0 | PASS |
| A | census | 5 commits / 13 files | exact | PASS |
| B | replay binary preexists | absent | absent | PASS |
| B | promgate run 1 (fixtures moved aside → full regeneration) | PROMGATE_LOCAL_GREEN | GREEN, rc=0, ref 0b789e0; script builds engines + qwen36_trace_replay itself; test-c PASS (incl. new liveness test); five arms bit-identical (fresh 1280B dumps, rc b/f=1/1 known fixture property) | PASS |
| B | promgate run 2 (replay binary removed, tree clean) | rebuild + GREEN | GREEN, rc=0 | PASS |
| C | stale-dump discriminator (stale artifacts seeded + candidate stub exit 126) | repaired gate refuses | 5× "no fresh DUMP" FAILs + trace-replay FAIL, PROMGATE_RED, exit 1; binary restored sha256-identical; tree clean | STALE_ARTIFACT_FALSE_GREEN_CLOSED = PROVEN |
| H1 | ABORT→FREE (verifier probe v2, independent of Lane A's test) | no hang, SATURATED classified, drain-on-retry, accounting balanced | 100/100 iters green (bounded alarm 20 s; zero hangs) | PASS |
| H2 | PUBLISH→PROGRESS (probe v2) | legal RESIDENT eviction, no RESERVED steal, balanced | 100/100 iters green | PASS |
| H1/H2 | Lane A's test_pread_liveness.c (cross-check) | ok | ok 100×100, rc=0; ASan/UBSan build ok; node TSAN run ok | PASS |
| H2- | old probe facet A on UNREPAIRED 486f557 (discrimination control, Phase 1) | hang | exit 42 "FAIL H: deadlock timeout" — the defect the repair removes | CONTROL PROVEN |
| H3 | QWEN lock-order source audit | no g_xorder_mx held across reserve/publish-wait | expert_acquire: unlock→reserve→poll-unlocked→recheck-under-lock (all paths balance, full-file verified); pilot_realload probe-under-lock/reserve-unlocked; publish under lock is non-blocking; SATURATED drops speculation | PASS |
| H3 | runtime saturation discriminator | progress under forced saturation | repaired engine cap=2 (vs cap=16): completes, token stream identical to cap16 line; PILOT+ASYNC saturation exercised; exit-time crash = pre-existing (below) | PASS |
| O | OLMoE lock-order audit | reserve outside engine mutex | expert_get + pilot_realload repaired identically to qwen36 (full-file verified, incl. coalesced hit `hits++/miss--` once-only counting); `would_evict` guard read-only, default PILOT_EVICT_GUARD=1 preserved; zero `loading[]`/`LCache` reintroduction (rg-proven); shared backend via xstore | PASS |
| O | OLMoE affected tests | green | test_olmoe_matmul_q, test_olmoe_serve_framing, test_olmoe_atomic_output.py green in test-c; olmoe builds | PASS |
| L | batch: unique+dup keys | transfer-once, per-entry leases | 4 views leased, loads==2, dedupe_removed==2, release→destroy clean (debug assert would fire on leak) | PASS |
| L | batch: partial load failure | failed entries cleared, no leak | ok==2, failed views NULL, successful leased, clean destroy | PASS |
| L | batch: parallel + mixed | caller-order leases | test_admission green (incl. parallel batch) under ASan/UBSan | PASS |
| L | OOM fallback path | releases job leases | source-audited only (calloc-failure branch) — runtime injection impractical | INFERRED (labeled) |
| D | destroy contract | header ≡ runtime; debug assert before free | expert_store.h:117-118 ≡ :270-272 ≡ pbs_destroy (assert(reservations==0 && live_leases==0) BEFORE unlock/free); fork probe: live-lease destroy → SIGABRT (sig 6), never clean exit; zero-live destroy passes in every H iteration | PASS |
| S | ASan+UBSan | no findings | vp_v2 + liveness + admission + pread + reservation suites: 5/5 rc=0, zero findings | PASS |
| S | TSAN | if practical | gcc/TSAN on node-02: liveness 100×100 rc=0, ZERO warnings | PROVEN (not UNMEASURED) |
| Q1 | tiny five-arm parity vs fresh f04359a build | bit-identical fresh dumps | ×2 promgate runs: all 5 arms PASS (default/PILOT/FUSED/ASYNC/BATCH); claim class REGRESSION_EQUIVALENCE (rc 1/1 fixture property) | PASS |
| Q2 | trace/accounting conservation | conserved, no double charge | 320 DEMAND rows = 279 hits + 41 misses; seq strictly monotone (344 rows, 0 regressions); residency state machine: 0 double INSERTs, 0 orphan hits; SATURATED/coalesced retries never emit extra physical rows; v4 stream E/R/C/B = 64/320/280/1 (matches prior verifier's PILOT-arm distribution); replay TRACE_REPLAY_SELF_CONSISTENT | PASS |
| Q2 | defaults-off | opt-in preserved | promgate default arm (env-cleared) bit-identical; repair diff introduces no new env knobs; SATURATED path reachable only via exhaustion, no mode silently enabled | PASS |
| Q3 | node-02 pins | 7606939241b64e82… / 831b27539e462cf8… | exact match both | PASS |
| Q3 | binary hashes (gcc -O3 -march=native -fopenmp -pthread, fresh node clones) | controller refs | candidate 057ff93237f529c… ≡ builder ref; baseline 6387cfe6acfd9e… ≡ frozen ref (bit-reproducible builds) | PASS |
| Q3 | paired parity run 1 (b1↔c1) + run 2 (b2↔c2) | exact deterministic output identity (banked normalization) | every diff line classified: (a) timing/RSS noise; (b) run-noise counters (hits/misses ±5-11, coalesce waits/skips) — same magnitude baseline-vs-baseline (b1↔b2 self-check); (c) source-proven DEAD TELEMETRY (below); zero unexplained lines; zero generated-content differences | PASS |
| Q3 | candidate self-checks | c1↔c1p↔c2 consistent | only pilot-race noise counters drift | PASS |
| Q4 | same-window perf | no gross regression | warm medians (turns 2–4): base {3.71, 4.56} cand {3.91, 4.46, 4.42}; same-window b2↔c2 = 4.56 vs 4.46 (−2.2%, within noise; baseline self-spread 3.71↔4.56) | NO REGRESSION |

## DEAD-TELEMETRY CLASSIFICATION (Q3 detail — source-proven, NOT a repair regression)

Baseline (f04359a, private cache) prints `[expert_io] demand/pilot loads`, `Admitted
bytes INT3/INT4`, and `packed INT4/INT3 expert CPU residency active` from
`load_expert_merged`. On the forge branch — at BOTH 486f557 and 0b789e0, unchanged by
the repair (121-line un-elided qwen36.c diff inspected) — `load_expert_merged` is
called ONLY from the speculative shadow ring (`spec_loader`), gated by
`COLI_SPEC_DEPTH>0` (default 0). Loads otherwise flow through
`qw_store_load → coli_expert_backend_pread_load` (store-side counters). Candidate
therefore prints 0 loads / 0 admitted bytes / no packed banners; `ALLOCATED EXPERT
SLOTS` reports the container census (10240 = 8704 INT3 + 1536 INT4) vs baseline's
private-cap basis (5120 = 128×40). All four lines are architecture-level reporting
differences present before the repair; flagged to Lane A as follow-up telemetry
rewiring (non-blocking, cosmetic-to-accounting only).

## PRE-EXISTING DEFECT REPORTED (out of repair scope)

Exit-time SIGSEGV in `pilot_worker` (teardown race vs main-thread free) at tiny-cap
PILOT configurations: baseline f04359a 2/10 vs candidate 1/10 runs (rc 139, after all
output/DUMP written; tokens always correct). Crash report
`~/Library/Logs/DiagnosticReports/qwen36-2026-08-24-145019.ips` (faulting frame
`pilot_worker`). Present on the frozen baseline at ≥ the candidate's rate → not
repair-introduced. ASan build did not reproduce in 20 runs (timing-shifted).
Recommended follow-up for Lane A: join pilot workers before store/model teardown.

## Verdict rules applied

All targeted repaired-surface gates pass. OLMoE real E2E explicitly out of scope
(Lane B2). No OLMoE status appended.

## FINAL VERDICT

`FORGE_F1_CORE_AND_QWEN36_INDEPENDENTLY_REVERIFIED_AFTER_LIVENESS_FIX`

## Evidence inventory

- Verifier clone: `/var/folders/.../T/kilo/f1-reverify/colibri` @ 0b789e0, clean, unpushed
- Packet: `…/T/kilo/f1-reverify/packet/` — VERIFIER_PLAN.md (Phase 1), verifier_probe.c
  (v1, discrimination control), verifier_probe_v2.c + vp_v2_run.log (100×100 green),
  gateC_discriminator.sh + gateC_reject.log, promgate_run1/run2.log,
  b1/b2/c1/c2/c1p logs, q36_full.diff (un-elided repair diff), prior_evidence/
- Node: fresh clone built, hashed, run, then REMOVED (`/home/nayte/verif_f1_liveness_0b789e0`,
  45 MB cleaned); all logs pulled to the Mac packet first; no bulk pulled
- Crash report: `qwen36-2026-08-24-145019.ips` (macOS DiagnosticReports)
- ANVIL ledger: handoff #55809 (Phase 1), memory #3086 (checkpoint)
