# FORGE F1 — Ownership Map + Operating Laws (Phase 8)

Branch: `forge/f1-ooc-moe-runtime` · Baseline: `f04359a`

## 1. Ownership map (per engine)

| Concern | QWEN36 | OLMoE | DeepSeek-V4 (legacy consumer) |
|---|---|---|---|
| Tensor names / layer remap | adapter: templates + `layer_map=active_of` | adapter: identity templates | adapter: `coli_dsv4_build_descriptor` |
| Quant formats / slot layout | backend classifies (INT8/INT4/INT3-g64); engine derives g/u/d pointers from blob offsets | backend INT8; engine derives g/u/d | v1 auto store (engine-owned, unchanged) |
| Physical residency/lease/in-flight | **store** (pread backend) | **store** | v1 store |
| Storage I/O | **backend** (pread/fused/drop) | **backend** (pread/drop) | v1 store |
| Eviction policy | store LRU (3-stage scan) | store LRU + **LFRU guard as engine prefetch veto** (`would_evict`) | n/a |
| Admission/prefetch policy | engine pilot ring + shared runner widths | engine pilot ring (single worker) | n/a |
| Coalescing | store BUSY identity + engine wait loop | store BUSY identity + engine wait loop | v1 |
| Telemetry | shared v3/v4 writers; engine policy emits rows | none (baseline had none) | route_trace.h |

**Single-source-of-truth rule**: residency, in-flight identity and eviction
decisions exist ONLY in the pread backend. Engines hold no parallel table
(verified: no `lc->slots`/`loading[]` remain in either engine).

## 2. Model-neutrality classification (post-Stage-A audit)

- GENERIC: expert_store.h v1.1/v1.2 ops+wrappers, ColiExpertCoreKey,
  reservation lifecycle, admission.{c,h}, expert_backend_pread.*, telemetry writers.
- ADAPTER-OWNED: descriptor name templates, `layer_map`, `slots_per_layer`,
  backend handles (`engine_context` etc.).
- LEGACY-COMPATIBILITY: registry v1 types + `open_selected`;
  `ColiExpertStore.gpu` (annotated in source: new backends must not touch it).
- LEAKED-MODEL-SPECIFIC into generic code: NONE found (grep-audited).

## 3. Operating laws (binding for future adapters)

1. The store is the only physical residency/admission truth. Engines never
   rebuild private cache tables behind the seam.
2. Reservation/lease lifetime is explicit: exactly one of publish/abort per
   reservation; views are invalid after release (capture slot_hint/bytes first).
3. No eviction or buffer reuse touches a RESERVED slot or a held lease.
4. One active reservation per key (BUSY); coalescing = wait for that publish,
   count once.
5. Loads run outside all ordering locks; mutation+trace pairs share one
   critical section (`g_xorder_mx` pattern).
6. Policy previews (`probe`, `would_evict`) are read-only and uncounted.
7. Defaults OFF until promotion evidence earns a change.
8. Parallelism width is a per-call-site decision; mechanics are shared
   (`coli_admission_run_parallel`), numbers belong to the engine protocol.
9. Telemetry observes; it never mutates routing/admission state.
10. Untrusted-container validation is ENGINE policy (fail closed at init).

## 4. Cross-model deadlock law (inherited from QWEN36 wedge)

Never reserve from an exhaustible pool while sequencing all publishers after a
possible wait on that same pool. Concretely tested: full-pool != unreservable;
a caller holding an unfinished reservation never waits on a publish only it can
perform. Covered by tests/test_expert_reservation.c (BUSY path) +
tests/test_admission.c (coalesce window with external publisher) +
tests/test_pread_liveness.c (H1 ABORT->FREE and H2 PUBLISH->PROGRESS on the
real backend, 100 iterations each; fully-RESERVED pools return the transient
COLI_EXPERT_ERR_SATURATED instead of spinning inside victim selection).

CORRECTION (2026-08-24, verifier F1-LIVE-1): an earlier revision claimed
coverage by "tests/test_expert_backend_pread.c (all-reserved spin drain)" —
NO such test existed; the claim is FALSIFIED. Worse, the then-current
`pbs_pick_victim` stage-3 spin could hang a reserve that another thread's
publish (blocked behind the caller-held engine lock) needed to drain: two
QWEN36 threads racing reserves into an exhausted pool deadlocked. The repair:
reserve() never blocks internally (explicit transient-saturation result;
retry above reconsiders FREE, growth, evictable RESIDENT, drained RESERVED),
and no engine thread holds `g_xorder_mx` across reserve() in qwen36.c or
olmoe.c.

## 5. Promotion gate

- promote.sh (this directory): builds both engines from an exact ref, runs
  make test-c + tiny parity A/B + replay gates, prints a promotion packet.
- Stage C real-container protocol: corpus_persistent_serving cap128 ebits4
  EP1 PILOT1 DENSE_I8 OMP8 N_NEW64 TEMP0; anchor 4.98 tok/s warm median
  (see forge_f1_container_pinning_20260823.md); factorial layout in the
  banked promgate receipt (QWEN36_FUSED_BATCH_PROMGATE_RECEIPT_20260823.md,
  copied here from ANVIL commit 2f41cdc with attribution).
- Rollback = redeploy frozen f04359a artifact; Forge lives on its branch
  until independent verification signs off.

## 6. Known-open items at builder closeout

- ~~Stage C node-02 real-container A/B: BLOCKED on cross-lane memory
  contention~~ **CLEARED / SUPERSEDED** (anvil blocker #55624 cleared by
  event #55760): A/B completed in a quiet window after two gdb-proven fixes
  (`b2c337b` explicit slots_per_layer budget; `2620ca0` adapter-declared
  expert_numel classification). Outputs IDENTICAL baseline-vs-forge across
  run pairs; tok/s within run variance under contention conditions.
- olmoe end-to-end vs real model: UNMEASURED (requires node-02; still open).
- HF tiny fixture 6/16: pre-existing baseline property (regression-
  equivalence discriminator only; identical on baseline and Forge). NOT an
  HF-oracle pass.
- Performance anchor: 4.98 tok/s warm median remains the HISTORICAL
  idle-node anchor from the frozen campaign receipt. Today's A/B medians
  (baseline 3.87–4.66, forge 4.37–4.50) are within-variance measurements
  under sibling-contention conditions — NOT a new absolute anchor.
- Promotion gate (`promote.sh`) is the LOCAL builder gate only. Independent
  verification is a separate, not-yet-started step. No deployment or merge
  has occurred; rollback reference remains frozen f04359a.
- Census provenance: the builder's closeout prose reported 19 commits / 31
  changed files total and 11 commits / 19 files continuation-only. Controller
  GitHub comparison at builder handoff `fcb95f2` instead resolved
  `f04359a..fcb95f2` as 20 commits / 28 changed files and
  `66d4660..fcb95f2` as 12 commits / 16 changed files. Treat the builder census
  as FALSIFIED; fresh verification must recompute exact counts from its live
  target ref rather than carrying either pre-verifier count forward.
