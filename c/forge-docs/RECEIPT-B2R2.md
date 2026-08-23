# RECEIPT-B2R2 — semantic closure repair (trace clock domain + registry extraction)

- branch `forge/b2-traces-profiler` (colibri fork) · parent `fbcaf88…` (B2R) · grandparent `32902b6…` (B2) · checkpoint pin `4917db5a…` intact
- responds to: independent re-verification **B2_REPAIR_REQUIRED** (event #55064), defects D3 (DISTINCT_MEASUREMENTS trace clock) + D4 (ONTOLOGY_EXTRACTION_SCOPE_DEFECT)
- scope: exactly `c/qwen36.c`, `c/forge_profile.sh`, this receipt; `c/forge_arms_persistent_serving.sh` untouched

## R3 — trace clock-domain separation
- TOKEN-BOUNDARY rows are now pure metadata: `{ev, i(process-wide), gen, gi(in-generation), first_in_generation, wall_delta_ms}` — **no metric field**; wall state resets at every generate() (`forge_begin_generation()` anchored at decode-loop entry); first boundary of EVERY generation emits `wall_delta_ms:null`; intervals can never span prefill gaps.
- ENGINE STEP rows are new: emitted at the OUTER completed-step site — `{ev:"engine_step", i==g_tm_dec_tokens (completed decode identity), gen, metric:"forge.runtime.step_ms_per_token", engine_step_ms=exact step() duration, cumulative_engine_step_ms_per_token=g_tm_step/g_tm_dec_tokens}`. The canonical id is attached HERE and nowhere else. No value is manufactured from boundary timestamps.

## R4 — two-phase registry extraction
Phase 1: parse ONLY the `public static let metrics: [MetricDefinition]` array slice (start anchored, termination proven) → symbols.
Phase 2: resolve each symbol via anchored `static let <sym> = try! MetricID("forge.…")` — exactly-one-match enforced; shape-checked; duplicates fail.
Fail-closed exit 5 on: missing/empty/unhashable source; unprovable start/termination; zero ids; unresolved/ambiguous symbol; bad id shape; duplicate ids. Embedded allow-list REMOVED. Temp slices live beside OUT on the evidence root (no internal temp).

## EVIDENCE LEDGER
| # | Test | Exit | Observed |
|---|------|------|----------|
| 1 | clean rebuild from pinned base | 0 | binary sha256-prefix `f2748ce9248cecf4` |
| 2 | instrumentation OFF | 0 | rc=0, zero forge stderr, no side effects |
| 3 | fresh multi-turn trace | 0 | 505/505 jq-parse = 252 token_boundary + 252 engine_step + 1 timing_summary |
| 3a | generation resets | — | gens 1–4 detected; first_in_generation=true ×4 at gi=1, all wall_delta null; valued deltas n=248 (=252−4), max 315.2 ms — prefill-spanning giants eliminated (was 21,236 ms) |
| 3b | index binding | — | boundary i sequence == engine i sequence byte-identical (trace index N == completed engine step N) |
| 3c | cumulative recompute | — | mean(engine_step_ms[1..i]) == cumulative field on 252/252 rows |
| 3d | same-run agreement | — | final trace cumulative_engine_step_ms_per_token == profile forge.runtime.step_ms_per_token == 147.8796 ms (exact) |
| 4 | R4 parity, canonical | 0 | ontology_source_identity b8d50f2d121289b16150a6319c9a048ce75437705588b62c0bd9475069938bb6; registry_ids=21; keys=10 pass |
| 5 | R4 adversarial bank | mixed as designed | A canonical PASS(0) · B registry-removal-only FAIL(1) ✓ · C comment-leak FAIL(1) ✓ bypass CLOSED · D fake-comment id absent from extracted set (extracted=21, fake-in-set=0) ✓ · E substitution strings contribute no raw ids (21==registry) ✓ · F missing exit5 ✓ · G empty exit5 ✓ · H unresolved symbol exit5 ✓ |
| 6 | guard selftest | 0 | ALL 5 PASSED |
| 7 | guarded persistent-serving arm | 0 | VALID wall=109s cpu=709.4s faults=0 maxrss=11.3GB real workload |
| 8 | band recompute (fresh same-run profile) | — | moe 61.5 (60.5) · dn 23.2 (24.9) · lmh 8.7 (8.9) · attn 5.6 (5.7) · step 147.9 ms/tok — WITHIN band |
| 9 | hygiene | 0 | git diff --cached --check CLEAN |

## ARTIFACT COORDINATES (approved node evidence root)
/home/nayte/bench_results/forge_b2r2_20260822/{profile.json, trace.jsonl, wrapper.out, run.log, arm.log, ont/*} · no /tmp usage anywhere in this tranche.

## GOVERNANCE
B2 remains RUNNING — builder does not self-promote. Handoff to independent verifier carries new HEAD, R3/R4 proofs, artifact coordinates.
