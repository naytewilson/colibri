# FORGE F1 — INTEGRATION BOUNDARY AUDIT R1 (independent recheck, 2026-08-24)

Fresh Lane D operator recheck of
`FORGE_F1_INTEGRATION_BOUNDARY_AUDIT_20260824.md` (banked at `c20f9e1`).
Every check below was re-run from live repo state and **can fail**. The original
receipt file is left byte-stable; corrections live here.

## Independent verification battery

| # | Check | Expected | Observed | Verdict |
|---|---|---|---|---|
| V1 | `git diff --stat 0b789e0..c20f9e1` docs-only | 2 doc files only | receipt + CSV, 924 insertions, no runtime path | PASS |
| V2 | ahead/behind `ecade075...0b789e0` | 0 / 792 | `0	792` | PASS |
| V3 | merge-base(ecade075, 0b789e0) | ecade075 (= old main fully contained) | `ecade075cfc2eae684097ea7de5570c3786ce199` | PASS |
| V4 | census CSV rows + header | 793 lines | 793 | PASS |
| V5 | class histogram sums; UNKNOWN count | Σ=792; UNKNOWN=0 | 561+76+66+37+33+9+8+1+1 = 792; UNKNOWN=0 | PASS |
| V6 | tracked binary-ish artifacts in full delta (`*.bin/o/so/dylib/a/exe/mlmodelc/mlpackage/hwx/spv`) | none unexplained | exactly one: `c/shaders/qmatmul.spv` — see correction C1 | PASS with note |
| V7 | origin/main live SHA | post-promotion state per recipe H2 | ls-remote: `0b789e06eb6448cd16bf904b23cc66e565f44295` | PASS |

## Corrections

**C1 (precision):** the audit says "`.spv` shaders are upstream-intentional
sources" (plural). Live tree truth: **exactly one**, `c/shaders/qmatmul.spv`,
added by `d4bf041` (2026-07-30, "vulkan: fmt=7 MXFP4 decode … for Kimi K3
experts") — an upstream Vulkan-engine feature commit in the campaign lineage,
referenced from `c/Makefile`. Classification unchanged (intentional build-input
source, not a generated artifact); only the count is corrected.

## Post-audit execution OBSERVED (this closes the loop on Mission H's recipe)

The promotion recipe (H2) has been **executed at origin** after `c20f9e1`:

| Artifact | Live state |
|---|---|
| `main` (origin) | ff'd to `0b789e06eb6448cd16bf904b23cc66e565f44295` |
| `pre-forge-f1-promotion-20260824` (rollback anchor) | `ecade075cfc2eae684097ea7de5570c3786ce199` |
| `post-forge-f1-promotion-20260824` | `0b789e06eb6448cd16bf904b23cc66e565f44295` |
| Lane C evidence | `verify/forge-f1-liveness-fix-20260824 @ 8cc8b1e` (`FORGE_F1_CORE_AND_QWEN36_INDEPENDENTLY_REVERIFIED_AFTER_LIVENESS_FIX`) |
| Lane B2 evidence intact (append-only) | `5d9f449` is ancestor of `verify/olmoe-forge-e2e-20260824`; Lane B3 independent rerun appended at tip `f5466e7` (byte-parity reproduced by fresh operator from clean rebuilds) |
| Sequenced follow-up present, not consumed | `fix/forge-f1-qwen-worker-telemetry-20260824 @ d3f130d` |

No protected surface was mutated by this recheck: verifier branches append-only,
candidate untouched, no deploy/symlink change, local `main` reconciled ff-only to
origin.

## R1 verdict

All audit claims independently confirmed; two precision notes above.
`FORGE_F1_COHERENT_LINEAGE_PROMOTION_BOUNDARY_PROVEN` stands, and promotion is
OBSERVED EXECUTED. Remaining sequenced work is exactly Mission H's
post-promotion reconciliation list (rebase `feat/qwen36-exact-next`,
`diag/qwen36-batch-wedge-20260823`, `docs/qwen36-control-freeze-20260823`;
retire/rebase stale `feat/qwen36-cuda-vram-tier`; land
`fix/forge-f1-qwen-worker-telemetry-20260824` with its targeted reruns).
