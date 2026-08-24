# QWEN36 FUSED+BATCH PROMOTION-GATE RECEIPT — 2×2 FACTORIAL (2026-08-23)

**HEAD**: feat/qwen36-async-expert-batch @ `653ec63`+telemetry/K repairs (this tranche, pushed after factorial) · production `tournament-p1@5dc15f4` untouched · both flags default OFF
**Binary**: one binary, all arms; node-02 seat-guarded; 20 runs = 5 counterbalanced blocks × {A,B,C,D}

## Repairs earned before measurement (W9/W10)
- K-bound: BATCH gated `K<=8` fail-closed (bidx[8] width).
- Telemetry: `g_acq_silent` mode — batch preload does full cache-state work with ZERO DEMAND emission/counters; loads classify to pilot buckets; canonical per-row pass owns every event exactly once. Post-repair parity EXACT re-proven.

## Factorial (total generation wall, s; lower better)
| Arm | mean | min–max | ΔvsA | hit% | RSS GB |
|---|---|---|---|---|---|
| A | 70.48 | 70.0–71.0 | — | 90.9 | 9.09 |
| B FUSED | 65.70 | 65.3–66.0 | **−6.8%** | 91.0 | 9.10 |
| C BATCH | 59.78 | 59.5–60.2 | **−15.2%** | **95.8** | 9.58 |
| D both | 56.76 | 56.5–57.0 | **−19.5%** | 95.8 | 9.58 |

Effects: fused=−4.78s, batch=−10.70s, combined=−13.72s, interaction=+1.76s (mildly sub-additive). Dispersion <1.5% ⇒ effects ≫ noise (CASE 2-leaning: BATCH_DOMINANT with material FUSED secondary).

## Prefill/decode decomposition (W6)
- Prefill-side dominates total-wall gain (BATCH mechanism; turn1 −27% class).
- Decode step() 252-tok: A 172.3 → D 163.2 ms/tok (**−5.3%**) via FUSED (~4ms) + BATCH residual warmth: prefill preload persists → hit rate 90.9→95.8% ⇒ fewer early-decode misses.
- Metric honesty (W15): original "+16.6%" was wall-based incl prefill; steady-state decode acceleration is the −5.3% component. Both are real production benefits, separately labeled.

## Exactness / safety
Parity EXACT same-binary OFF-vs-each-flag (probe) post-repair; units ctx/i4/mixed all pass; routing untouched by construction (pre-pass on scratch copy; canonical loop authoritative); RSS regression +0.49GB bounded (batch staging + residency diversity); no OOM/swap events in 20 runs; logs at bench_results/qwen36_frontier/run_B*.log + factorial_results.csv.

## Interpretation: CASE 2 — BATCH_DOMINANT (primary promotion candidate), FUSED secondary (robust −6.8%, keep default-OFF opt-in or bundle).

## VERDICT (exactly one): `QWEN36_FUSED_BATCH_EXACT_GAIN_READY_FOR_INDEPENDENT_VERIFY`
Next: fresh independent verifier reproduces ≥2 blocks + parity + telemetry audit; then controller promotion decision. Default flip NOT authorized here.
