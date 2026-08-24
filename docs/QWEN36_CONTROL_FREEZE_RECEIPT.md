# QWEN36 Full-Size Control Freeze Receipt

Date: 2026-08-23

## Controller verdict

`QWEN36_FULLSIZE_CONTROL_FROZEN`

This is a **control-freeze** verdict, not a claim that a pre-registered numeric quality threshold was passed.

The exact QWEN36 production artifact is frozen as the campaign control because its source, binary, deployment, rollback, output-parity, performance, and quality-provenance identities are now sealed. No pre-existing numeric quality acceptance threshold was found for this artifact class, so none is invented or applied retroactively after observing the measurements.

## GitHub-proven immutable refs

- Production source branch: `feat/qwen36-async-expert-batch`
- Production source SHA: `f04359aab31a388cc47d36e96c3ea36400061cb6`
- Controller re-verification: the branch was still identical to that SHA at freeze review.
- Provenance-safe quality tooling branch: `fix/qwen36-quality-artifact-binding-20260823`
- Quality tooling SHA: `59c487b3c926ed19f58ea89ab7f8bc9002602c15`
- Controller re-verification: the quality-tooling branch was still identical to that SHA at freeze review.

## Receipt-supported node evidence

The final node-side quality closure report recorded:

- production executable: `/home/nayte/ane-deploy/qwen36/promotions/f04359a_20260823T2204Z/qwen36`
- production binary SHA-256: `6387cfe6acfd9e4110bdcafc31d7418dbedf8eabd27e75cc187a547dddbc4421`
- corpus: `/home/nayte/prompts/heldout_eval.json`
- corpus SHA-256: `357cb4d8557008a691264119eb5ffa70079ab3c14599c61e3dee0bb37a0f4152`
- corpus size: 1142 bytes
- corpus shape: one held-out sample, 20 prompt tokens, 131 full tokens, 111 scored tokens, no domain field
- immutable packet: `~/ane-deploy/qwen36/quality/f04359a_20260824T0236Z/`
- deployment receipt: `~/ane-deploy/qwen36/DEPLOYMENT_RECEIPT_20260823_final.md`
- ANVIL decisions reported: `#55504`, `#55516`, `#55518`, `#55525`, `#55527`

Reported four-arm measurements on the exact promoted binary:

| Arm | TF-NLL | PPL |
| --- | ---: | ---: |
| Uniform INT3 | 2.6310 | 13.89 |
| Mixed-Low production | 2.5345 | 12.61 |
| Mixed-Med | 2.5888 | 13.31 |
| Uniform INT4 | 2.4933 | 12.10 |

The packet reportedly re-hashed the binary and corpus before/after evaluation and froze all packet files read-only.

## Epistemic classification

### PROVEN

- GitHub production source ref remains frozen at `f04359a...`.
- GitHub provenance-safe quality tooling remains frozen at `59c487...`.
- The campaign ledger already warned that historical quality provenance was unresolved and required new measurements to be recorded separately with hashes.

### RECEIPT-SUPPORTED

- exact node deployment binary identity and path;
- sealed single-sample quality corpus identity;
- four-arm TF-NLL/PPL measurements;
- packet immutability and node-side hash verification;
- deployment/rollback linkage and ANVIL decision chain.

### NOT CLAIMED

- no claim that this is a multidomain corpus;
- no claim that a numeric quality threshold existed before this run;
- no claim that the measured packet formally passed a threshold selected in advance;
- no claim that historical 8-domain PPL numbers share this corpus provenance.

## Why no post-hoc threshold is registered here

The quality-closure worker proposed choosing a rule after seeing the sealed measurements, including margins such as `M=0.09` versus INT3 or `D=0.05` from INT4. Applying such a rule retroactively would violate the campaign's explicit no-retrospective-threshold-shopping constraint.

A future candidate-quality policy may be defined prospectively, before the candidate measurements are inspected, and may then use this frozen QWEN36 control packet as a baseline/reference. The frozen control does not need to be re-run merely to manufacture a historical threshold that never existed.

## Campaign consequence

The QWEN36 full-size control is now frozen and leaves the critical path. New QWEN36 optimization work is not required for control closure. Existing diagnostic or research side lanes may continue only when they have independent value and must not mutate the frozen production control.

The active product/runtime Main Quest may proceed using this exact QWEN36 artifact and its sealed measurements as the control reference.
