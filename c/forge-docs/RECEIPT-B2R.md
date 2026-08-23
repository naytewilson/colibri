# RECEIPT-B2R — trace-semantics + ontology-freshness repair of B2

- branch: `forge/b2-traces-profiler` (colibri fork) · parent: `32902b6276ed07477a6a63da961b6d4fc5b3fbcf` (B2) · grandparent/checkpoint pin: `4917db5a36ae5fb73e1dc519847ac1fff13043e4`
- responds to: independent verifier verdict **B2_REPAIR_REQUIRED** (ANVIL event #54967), defects D1 (TRACE_SEMANTICS_DEFECT) + D2 (STALE_PARITY_GATE_DEFECT)
- scope honored: only `c/qwen36.c` + `c/forge_profile.sh` touched; `c/forge_arms_persistent_serving.sh` untouched (inspection showed no need)

## R1 — trace timing semantics (D1)

ORIGINAL DEFECT: token_boundary rows emitted `g_tm_step/g_tm_dec_tokens` (lagged cumulative mean) under key `"step_ms"`; row i=1 fabricated `0.0000`; key name diverged from claimed metric id `forge.runtime.step_ms_per_token`.

FIX: two explicitly distinct concepts per row:
- `wall_delta_ms` — raw observation (metadata-class, NOT an ontology metric): actual wall interval between consecutive decode-step completions, measured boundary-to-boundary around step(); `null` on the first boundary (no prior boundary — never fabricated).
- `cumulative_step_ms_per_token` — cumulative mean of COMPLETED intervals, associated with canonical id `forge.runtime.step_ms_per_token` (kept in the `metric` field); recomputes exactly as mean(wall_delta_ms[2..i]); `null` on first boundary.

## R2 — ontology freshness (D2)

ORIGINAL DEFECT: wrapper embedded a verbatim ontology allow-list copy; could silently pass after MetricOntology.swift drift.

FIX: parity allow-list is DERIVED at run time from the canonical source (`grep -oE '"forge\.[a-z0-9_.]+"'`). Resolution order: `$FORGE_ONTOLOGY_SRC`, then `$HOME/.forge-ontology/MetricOntology.swift`; **fail-closed (exit 5)** when absent — no embedded fallback exists, so nothing non-canonical can satisfy the gate. Output stamps:
- `ontology_source=/home/nayte/.forge-ontology/MetricOntology.swift`
- `ontology_source_identity=b8d50f2d121289b16150a6319c9a048ce75437705588b62c0bd9475069938bb6` (byte-identical to Mac canonical @aa95397)

## EVIDENCE LEDGER

| # | Test | Command shape | Exit | Observed |
|---|------|---------------|------|----------|
| 1 | clean rebuild | regenerate qwen36.c from 4917db5 + B2R block; `make -C c qwen36` | 0 | binary sha256-prefix `175dd55d90d0cb22` |
| 2 | instrumentation OFF | env-stripped serving run | 0 | zero `[forge]` stderr lines; no profile/trace side-effect files |
| 3 | fresh trace | 4-turn/64-token run w/ COLI_FORGE_TRACE | 0 | 253/253 rows jq-parse; 252 boundaries strictly monotonic; i=1 both fields null; cumulative recomputes exactly on 251/251 valued rows (`mean(wall_delta[2..i])`); closes with timing_summary |
| 4 | real ontology parity | wrapper w/ canonical source | 0 | 10/10 keys; identity stamped |
| 5 | stale-ontology negative | disposable copy minus `forge.moe.total_ms_per_token`, FORGE_ONTOLOGY_SRC override | **1** | `ONTOLOGY_PARITY_FAIL: 'forge.moe.total_ms_per_token' not in canonical ontology /tmp/stale_ont.swift` — gate fails closed |
| 6 | guard selftest | `anvil-bench selftest` on node | 0 | ALL 5 PASSED |
| 7 | guarded persistent-serving arm | `forge_arms_persistent_serving.sh` | 0 | VALID: wall=108s(≥20) cpu=703.5s(≥1) faults=0(≤50k) exit=0 G2-row matched maxrss=11.3GB real workload |
| 8 | band recompute (fresh run) | shares from /tmp/b2r_p2.json metrics | — | moe 61.1% (ref 60.5) · deltanet 23.5% (24.9) · lm_head 8.8% (8.9) · attention 5.7% (5.7) · step 147.1 ms/tok (ref ~147.96) — all within band |
| 9 | hygiene | `git diff --cached --check` | 0 | CLEAN |

## TRACE ARTIFACT COORDINATES
- `/tmp/b2r_trace.jsonl` (253 rows; fresh post-repair) · `/tmp/b2r_profile.json` (parity-gated) · guarded-arm evidence `/home/nayte/ane-hot/colibri-qwen36/forge-b2-persistent-serving-20260822T204412/`

## GIT
- one follow-up commit on `forge/b2-traces-profiler` (32902b6 NOT rewritten) · files: `c/qwen36.c`, `c/forge_profile.sh`, `c/forge-docs/RECEIPT-B2R.md` · pushed non-force to `fork/forge/b2-traces-profiler`

## GOVERNANCE
- B2 remains **RUNNING** in canonical DAG (`b2-live-qwen-serving`) — builder does not self-promote.
- Handoff to independent verifier for re-verification (R1/R2 proof above; artifacts listed).
