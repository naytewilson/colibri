# FORGE F1 — INTEGRATION BOUNDARY AUDIT (Lane D, 2026-08-24)

**Lane**: final integration-boundary audit after evidence convergence
**Candidate**: `forge/f1-ooc-moe-runtime @ 0b789e06eb6448cd16bf904b23cc66e565f44295`
**main at audit time**: `ecade075cfc2eae684097ea7de5570c3786ce199` (2026-07-30, PR #650)
**Joined evidence consumed** (not rerun): Lane C
`verify/forge-f1-liveness-fix-20260824 @ 8cc8b1e1288c811dd65f1d4d3e8334863c10641a`
(`FORGE_F1_CORE_AND_QWEN36_INDEPENDENTLY_REVERIFIED_AFTER_LIVENESS_FIX`) + Lane B2
`verify/olmoe-forge-e2e-20260824 @ 5d9f44968e130a21789a2cb9a27465c33159fc64`
(`OLMOE_FORGE_REAL_E2E_GATE_REPROVEN_ON_LIVENESS_REPAIR`)
**No merge, no deploy, no production mutation.** Candidate/verifier branches untouched.

## Mission A — production lineage (PROVEN, live repo)

| Relationship | Value |
|---|---|
| merge-base(main, candidate) | `ecade075` = main itself → main fully contained |
| ahead / behind | 792 / 0 |
| main..`f04359a` | **765 commits** = Qwen3.6 production campaign (2026-07-18 → 2026-08-23) |
| `f04359a`..candidate | **27 commits** = Forge F1 (22, `518965f`..`486f557`) + liveness repair (5, `f75719d`..`0b789e0`) |
| merge commits in delta | 254 (branch-based campaign workflow) |
| `f04359a` | tip of `feat/qwen36-async-expert-batch` — the promoted production mixed-baseline anchor (anvil_core ledger: "Promoted to Production Mixed Baseline"); frozen control for BOTH verifier lanes |
| Contained campaign branches | `feat/qwen36-async-expert-batch @ f04359a` ✓, `feat/qwen36-cache-policy-tournament-p1 @ 5dc15f4` ✓, `feat/qwen36-packed-int4-cpu-residency @ 4917db5` ✓ |
| Parallel lines NOT in candidate | `feat/qwen36-exact-next @ 4579d41` (3 ahead of f04359a, runtime), `diag/qwen36-batch-wedge-20260823 @ a0f0fd1` (4 ahead, runtime), `docs/qwen36-control-freeze-20260823 @ 80c3eed` (13 ahead of mid-campaign, docs/process), `feat/qwen36-cuda-vram-tier @ 06d27cf` (stale, Jul-30, behind 1017) — post-promotion rebase candidates, not contamination |
| `66d4660`, `486f557` | in candidate (Forge Stage-A handoff / verifier-confirmed pre-repair candidate) |

## Mission B — commit census (machine-readable, `forge_f1_integration_commit_census.csv`)

792 rows, programmatic (segment + dominant-subsystem + subject rules; Forge segment
hand-verified). Class histogram:

| Class | Commits |
|---|---|
| UNRELATED_BUT_INTENDED_PRODUCTION | 561 (upstream repo surface: CLI/web/GPU/docker/other engines — the project's own multi-engine evolution carried by campaign merges) |
| GENERIC_RUNTIME_PREREQUISITE_REQUIRED | 76 (shared ExpertStore lineage incl. DeepSeek-V4-originated store, tools, build) |
| TEST_VALIDATION_REQUIRED | 66 |
| CAMPAIGN_EVIDENCE_ONLY | 37 (docs/experiments receipts) |
| QWEN36_REQUIRED | 33 |
| FORGE_CORE_REQUIRED | 9 |
| DOC_RECEIPT_ONLY | 8 |
| OLMOE_REQUIRED | 1 (Stage-7 port) |
| GENERATED_OR_ACCIDENTAL | 1 (`dda965f` — the untrack-binaries remediation itself) |
| UNKNOWN_REVIEW_REQUIRED | **0** |

## Mission C — file-surface census (348 files, main..0b789e0)

TESTS 108 · DEEPSEEK 51 · CONVERTERS_TOOLS 50 · OTHER_SOURCE 36 · OTHER/upstream 33 ·
DOCS_RECEIPTS 31 · QWEN36 16 · SHARED_STORE 12 · OLMOE 6 · BUILD_CI 5 ·
**GENERATED_BINARY 0**. Zero deletions. Adds are legitimate new sources
(`admission.{c,h}`, `deepseek_v4.c`, backends, tools). `c/coli` is a Python CLI
source (verified), not a binary.

## Mission D — integration class: **D3 (coherent lineage promotion)**

Evidence:
1. main (Jul-30) predates the entire production campaign; production QWEN36 truth
   runs from the campaign lineage, not from main (anvil_core ledger promotion record;
   `f04359a` = promoted mixed baseline = frozen control of both verifier lanes).
2. The campaign's own production branches are contained in the candidate (A.3).
3. Forge F1 was built directly on that production baseline and independently
   reverified at `0b789e0` (Lane C) — OLMoE E2E reproven on the same SHA (Lane B2).
4. Census shows a coherent evolution: zero UNKNOWN, zero tracked artifacts, zero
   deletions, no contamination markers.
5. behind = 0 → promotion is a fast-forward; reconstruction onto stale main (D1)
   would discard the promoted production campaign and is evidence-contrary.
D1 rejected (main stale, would orphan production lineage). D2 subsumed (the "newer
accepted production lineage" IS the candidate's inherited history — same object).
D4 rejected (census shows no contamination).

## Mission E — trial integration

D3 genuinely proven → **no synthetic reconstruction** (per lane card). The validated
integration object is the candidate itself at `0b789e0`; this audit adds docs-only
commits on `audit/forge-f1-integration-boundary-20260824`.

## Mission F — validation at the promotion boundary

| Check | Result |
|---|---|
| `make -C c test-c` | PASS at `0b789e0` (Lane C promgate ×2) |
| ExpertStore/backend/admission tests | PASS (Lane C: pread/reservation/admission/liveness suites, local + sanitizers) |
| H1/H2 liveness | PASS ×100+×100 (verifier probe v2 + Lane A suite, ASan/UBSan/TSAN) |
| QWEN36 build / OLMoE build / trace-replay build | PASS (Lane C promgate: script builds all three itself, ×2 from clean prerequisites) |
| DeepSeek shared-store compatibility | **PASS (this lane)**: `test_deepseek_v4` ok, `test_v4_ownership` ok at `0b789e0` |
| Promotion script local gate | PROMGATE_LOCAL_GREEN ×2 (Lane C) |
| Tracked generated-artifact census | 0 binary artifacts tracked in tree (`.spv` shaders are upstream-intentional sources) |
| `git diff --check` | clean (this lane, at `0b789e0`) |
| Real QWEN36 Stage C | PASS (Lane C on node-02: pins verified, bit-reproducible binaries, paired deterministic parity, NO REGRESSION) |
| Real OLMoE E2E | PASS (Lane B2, same SHA) |

## Mission G — evidence transfer matrix

This audit's commits are **docs-only** (receipt + census CSV). Runtime source at the
audit-branch tip is byte-identical to `0b789e0` for all production surfaces.

| Evidence | Changed runtime files vs `0b789e0` | Transfer class |
|---|---|---|
| Lane C / QWEN (`8cc8b1e`) | none | **TRANSFERS_EXACTLY** |
| Lane B2 / OLMoE (`5d9f449`) | none | **TRANSFERS_EXACTLY** |
| Both → post-promotion main (ff to `0b789e0`) | none (fast-forward introduces no source change) | **TRANSFERS_EXACTLY** |

The isolated follow-up branch `fix/forge-f1-qwen-worker-telemetry-20260824`
(pilot_worker teardown join + dead-telemetry rewiring) was NOT consumed; when it
lands it must rerun per its own surface classes: shared pread/reserve/publish
change → Lane C targeted rerun (H1/H2 + tiny parity minimum), QWEN-only lifecycle →
OLMoE transfers, QWEN targeted rerun.

## Mission H — FINAL PROMOTION RECIPE (H2 — coherent lineage promotion)

- **Promote**: `main` ← `0b789e06eb6448cd16bf904b23cc66e565f44295` via fast-forward
  (`git checkout main && git merge --ff-only 0b789e06eb6448cd16bf904b23cc66e565f44295`
  then `git push origin main`). Zero source delta vs verified state; behind=0 makes
  conflicts impossible.
- **Proof of inherited lineage intent**: anvil_core promotion ledger (Qwen3.6-35B-A3B
  mixed baseline promoted, PPL 5.39, warm 5.33 tok/s ≥ 4.5 target); `f04359a` = frozen
  control of both verifier lanes; campaign production branches contained (Mission A);
  census coherence (Mission B/C).
- **Rollback anchor**: pre-promotion `main @ ecade075cfc2eae684097ea7de5570c3786ce199`
  (tag recommended: `pre-forge-f1-promotion`); functional runtime rollback:
  `f04359aab31a388cc47d36e96c3ea36400061cb6` (last pre-Forge production baseline).
- **Post-promotion reconciliation (sequenced AFTER promotion, separate lanes)**:
  rebase `feat/qwen36-exact-next`, `diag/qwen36-batch-wedge-20260823`,
  `docs/qwen36-control-freeze-20260823` onto promoted main; retire or rebase stale
  `feat/qwen36-cuda-vram-tier`; land `fix/forge-f1-qwen-worker-telemetry-20260824`
  with its targeted reruns per Mission G rule.
- **Explicitly NOT this lane**: moving main, deploy, production symlinks — controller
  executes the recipe.

## FINAL VERDICT

`FORGE_F1_COHERENT_LINEAGE_PROMOTION_BOUNDARY_PROVEN`
