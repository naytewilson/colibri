# FORGE F1 — Handoff: Stage A landed, Stage B/C next

Date: 2026-08-23 · Branch: `forge/f1-ooc-moe-runtime` @ `a54f834` (base `f04359a`)
Worktree: `~/ANVIL-worktrees/colibri-forge-f1`

## Landed (commits in order)

| Commit | Content |
|---|---|
| `518965f` | Phase 0: container pinning ledger + mechanism census (`docs/experiments/`) |
| `e8cc76e` | Phase 1: expert_store.h v1.1 reservation surface + tests |
| `35e9475` | Phase 2: registry de-coupling, ColiExpertStoreDescriptor v2 open |
| `61b819a` | Phase 3: shared pread/LRU backend + 14-case unit matrix (real safetensors fixtures built in C) |
| `77c117c` | Phase 4: admission scheduler policy layer (`c/admission.{c,h}`) |
| `3bd496e` | Phase 5: engine-agnostic telemetry writers (v3/v4 byte-pinned), qwen36 emitters rerouted |
| `a54f834` | **Phase 6 Stage A**: qwen36 live cache on the shared store; bit-identical A/B vs baseline |

## PROVEN gates for a54f834 (all run on Mac this session)

- logits BIT-IDENTICAL vs frozen-commit binary across 5 arms:
  default / PILOT=1 / COLI_FUSED_LOAD=1 / COLI_EXPERT_ASYNC=1 / COLI_BATCH_ACQ=1
  (tiny container `/tmp/kilo/qwen36_tiny_i8`, ref `/tmp/kilo/ref_qwen36.json`,
  regen recipe: `tools/make_qwen36_tiny.py --out … --emit-ref …` then
  `tools/convert_qwen36.py --model … --out …_i8 --ebits 8`; run
  `SNAP=<i8> ./qwen36 16 8 ref_qwen36.json`).
  NOTE: this fixture diverges from the HF oracle at token 6 IDENTICALLY on
  baseline and port — pre-existing fixture/reference property, not ours.
- `qwen36_trace_replay`: TRACE_REPLAY_SELF_CONSISTENT ×3 under PILOT=1.
- `qwen36_policy_sim` consumes the captured v4 stream.
- full `make test-c` rc=0 (`TEST_EXCLUDE="test_uring test_qwen36_i4_kernel"`
  locally; i4_kernel is x86-only, verified failing at clean f04359a on arm64).
- admission + preread units additionally clean under ASan+UBSan.

## Key mechanics the next agent must not break

- Trace-order invariant: every slot-table mutation and its v3 row share
  `g_xorder_mx` (loads stay outside). Replay gate fails if violated.
- Views are INVALIDATED by release() — capture `slot_hint`/bytes/format
  BEFORE `coli_expert_release`.
- PRELOAD publishes carry NO lease; never release a view you were not given.
- Eviction notifications run on the evicting thread inside its own reserve()
  → thread-local victim capture is exact.
- DeepSeek auto backend self-registers from deepseek_v4.c constructor;
  registry is engine-free (no weak-symbol tricks — Mach-O ld rejects them).

## Next (exact unfinished deliverable)

1. **Stage B** (plan task 13): route pilot prefetch + batch acquisition +
   async decode through `c/admission.h` with knobs default-OFF. Gates:
   tiny parity bit-identical again, replay SELF_CONSISTENT, policy-sim deltas
   within noise.
2. **Stage C** (task 14): node-02 real-container parity + A/B on pinned
   container `/home/nayte/models/qwen36_mixed_low` (hashes in
   `docs/experiments/forge_f1_container_pinning_20260823.md`); protocol =
   corpus_persistent_serving cap128 ebits4 EP=1 PILOT=1 OMP=8 TEMP=0 N_NEW=64,
   anchor 4.98 tok/s warm median, promgate factorial layout (20 runs).
   Build on node: needs Makefile deps for qwen36 target there too
   (expert_backend_pread.c admission.c expert_store_registry.c already in
   the rule).
3. Phase 7: olmoe.c second-consumer port + delete its duplicate machinery
   (~500-line class; census section 2 maps it; keep PILOT_EVICT_GUARD as
   scheduler policy via admission config or engine-side hook).
4. Phase 8 deliverables incl. copying the promgate receipt from ANVIL
   commit `2f41cdc` into colibri docs with attribution.
5. Phase 9: push branch + Battle Report.

## Environment notes

- anvil-hub events logged: #55493/#55494 (phase 0), #55515 (phases 1-2),
  #55544 (phase 3), #55554 (phases 4-5). Bank stage C receipts when measured.
- rtk tee logs hold full grep output when RTK truncates:
  `$HOME/Library/Application Support/rtk/tee/`.

---

## CONTINUATION CLOSEOUT (2026-08-24)

- Stage B: batch+async finish concurrency routed through shared
  coli_admission_run_parallel; widths = baseline rules; gates green (ca7e9e4).
- Stage C: node-02 mixed_low A/B COMPLETE after two real-container fixes
  (b2c337b slots_per_layer; 2620ca0 expert_numel classification — gdb-proven
  INT8-misclassification segfault). Outputs IDENTICAL base-vs-forge;
  tok/s within run variance (base medians 3.87–4.66, forge 4.37–4.50 under
  sibling-contention conditions; frozen anchor 4.98 measured on an idle box).
- Phase 7: olmoe second-consumer port landed, duplicate class deleted,
  LFRU guard preserved via would_evict preview; unit gates green (531e8be).
- Phase 8: operating laws + ownership map, promote.sh gate
  (PROMGATE_LOCAL_GREEN at 333a3ee), promgate receipt banked from ANVIL.
- olmoe end-to-end vs real model: still UNMEASURED (node dependency).
- Verdict: FORGE_F1_STAGES_0_9_COMPLETE_READY_FOR_INDEPENDENT_VERIFICATION
