# FORGE F1 — INDEPENDENT VERIFIER RECEIPT (2026-08-24)

**Verifier**: fresh independent verifier (ox-alpha / Kilo), no builder state reused
**Target**: `forge/f1-ooc-moe-runtime` @ `486f557dc8e1c4bfb10ff7382dcb7c10a7602cd7`
**Baseline**: `f04359aab31a388cc47d36e96c3ea36400061cb6` · Handoff anchor `66d4660` · Builder closeout `fcb95f2`
**Method**: fresh clone (`/var/folders/lz/.../T/kilo/f1-verify/colibri`), all gates rerun from source

## Identity & lineage (recomputed)

| Check | Expected | Observed | Verdict |
|---|---|---|---|
| remote URL | naytewilson/colibri | https://github.com/naytewilson/colibri.git | PASS |
| branch / HEAD | forge/f1-ooc-moe-runtime @ 486f557 | exact | PASS |
| clean tree | 0 dirty | 0 | PASS |
| f04359a ancestor | yes | yes | PASS |
| 66d4660 ancestor | yes | yes | PASS |
| commits f04359a..HEAD | 22 | 22 | PASS |
| files f04359a..HEAD | 28 | 28 | PASS |
| commits 66d4660..HEAD | 14 | 14 | PASS |
| files 66d4660..HEAD | 16 | 16 | PASS |
| d5cc9c2 scope | promote.sh only | 1 file, 1 line | PASS |
| 486f557 scope | laws doc only | 1 file (+7/-3) | PASS |

## Gate results

| Gate | Verdict |
|---|---|
| MISSION B promote.sh (after building untracked-prereq tool) | PROMGATE_LOCAL_GREEN reproduced; 5 arms bit-identical (fresh 1280B dumps each) |
| MISSION C old-gate discriminator | OLD_GATE_STALE_ARTIFACT_FALSE_GREEN = PROVEN (old gate printed GREEN with both engines dead over seeded stale artifacts); repaired gate fail-closed PROVEN (single-side rc=126 → per-arm "no fresh DUMP" FAILs, RED, exit 1); binary restored sha256-identical |
| Store lease/reservation contract | wrapper+mock suite pass; real backend reserve/publish/abort/pin/batch verified in source + probe |
| Coalescing one physical load | PROVEN by probe: 8 racing callers → winners=1, coalesced=7, physical_loads Δ=1, bytes_read Δ=2048 |
| Full-pool liveness | PARTIAL: drain-via-publisher works (probe facet B green); **DEFECT F1-LIVE-1 below** |
| Model neutrality | generic surfaces carry zero model geometry/tensor names/qwen-olmoe code paths; DeepSeek refs are opaque legacy typedefs; `gpu` touched once (NULL init at construction) = LEGACY-COMPATIBILITY hygiene |
| Slot budget / expert_numel | slots_per_layer explicit-count wins; expert_numel authority fails closed ("not INT8/INT4/INT3-g64 classifiable"); adapters wire both (qwen36.c:1715-1718, olmoe.c:401-402) |
| QWEN36 private machinery deletion | `loading[]` zero hits both engines; LCache only in comment; env defaults preserved; K≤8/S≤512/cap≥2K gates intact |
| Tiny 5-arm parity | bit-identical logits ×5 arms (rc b/f=1/1, known fixture property = REGRESSION_EQUIVALENCE only) |
| Trace semantics | default arm: 322 rows semantic-identical (adm_ms aside); PILOT arm: v4 classes E/R/C/B byte-identical per stream (64/320/280/1); raw interleave delta = PHYSICAL_ADMISSION_ORDER_DIFFERENCE; replay SELF_CONSISTENT both sides |
| ASan+UBSan | reservation/backend/admission/telemetry suites rc=0 no findings; concurrency probe facet B clean |
| Failure injection | wrong-geometry/missing-tensor/bad-template/nonexistent-path fail closed at open w/ message; truncated container + short pread exit(1) via st-layer guards (fail-closed, loud); destroy-with-live-lease reports in debug |
| DeepSeek legacy compat | test_deepseek_v4 ok, test_v4_ownership ok, registry engine-free verified, v1 path intact |
| QWEN36 real-container Stage C | REPRODUCED INDEPENDENTLY on node-02 from verifier-built binaries: 2 paired runs/side, pinned `/home/nayte/models/qwen36_mixed_low` (config.json pin 7606939241b64e82… verified), protocol cap128 ebits4 EP=1 PILOT=1 DENSE_I8 OMP=8 TEMP=0 N_NEW=64; all four outputs byte-identical (sha256 b236ff277b9f47209f6d0f3d… after stripping timing/RSS telemetry) |
| Binary hashes (node gcc -O3 -march=native -fopenmp -pthread) | candidate 0d43a2e6600c993b… ; baseline 6387cfe6acfd9e41… |
| Same-window perf | warm medians base {4.05, 4.61} vs forge {4.54, 4.54} tok/s — forge within-window ≥ baseline; NO regression |
| 4.98 tok/s | remains HISTORICAL_IDLE_NODE_ANCHOR (both sides below it under today's window; not comparable as regression) |
| OLMoE source port | duplicate class genuinely deleted (no loading[]/LCache/slots[]); same shared backend; would_evict read-only guard; PILOT_EVICT_GUARD default preserved |
| OLMoE real E2E | **BLOCKED** — no canonical OLMoE model/container exists in source truth (node models/, /data archive, Mac volumes swept); converter tooling exists but no established deterministic invocation/oracle to reproduce; inventing one is out of verifier scope |
| Defaults-off | candidate == frozen baseline with all knobs unset (arm "" parity); every advanced arm opt-in ('1' required) |
| Production surfaces | untouched: no deploy symlinks, frozen refs, containers, or other branches modified; verifier worked only in temp dirs + new node clones (cleaned) |

## DEFECT F1-LIVE-1 (substrate liveness edge; reproducer held by verifier)

`pbs_pick_victim` stage-3 spin scans only PBS_RESIDENT victims. When EVERY slot is
PBS_RESERVED and a concurrent abort/failure frees a slot (PBS_FREE), the spinning
reserve never rescans FREE buffers and hangs despite capacity existing. Engine
callers hold `g_xorder_mx` across `coli_expert_reserve` (qwen36.c:2341→2365,
3360→3373) while publish needs the same mutex (3402/3447): two engine threads
racing reserves into an exhausted pool deadlock. Reproducer: verifier probe
facet A — pool cap 2/layer fully reserved → third reserve spins → abort frees a
slot → spinner never wakes ("FAIL H: deadlock timeout", exit 42).

Mitigating scope: production caps (128/layer) vs ≤~10 concurrent in-flight loads
make exhaustion unreachable today; engines exit(1) on load failure so FREE-via-
abort is rare; tiny-parity arms and Stage C never entered stage 3.

Related falsifications found:
- Operating-laws §4 claims "tests/test_expert_backend_pread.c (all-reserved spin
  drain)" covers this law — NO such test exists anywhere (grep-proven).
- expert_store.h says destroy() "debug builds assert" on live leases; actual
  implementation fprintf-reports then destroys anyway (doc/behavior mismatch).
- promote.sh is not self-contained on a fresh clone: it invokes `c/qwen36_trace_replay`
  without building it (untracked since dda965f); first independent run was
  PROMGATE_RED until the tool was built manually. Fail-closed direction, but
  builder's original GREEN depended on an untracked stale binary in the worktree.
- admission acquire_batch job-views hold leases that are never released
  (leak-by-design? undocumented); harmless to shipped engines which do not call
  acquire_batch (they use reserve/publish + run_parallel directly).

## VERDICT

`FORGE_F1_CORE_AND_QWEN36_INDEPENDENTLY_VERIFIED_DEFECT_FOUND_OLMOE_E2E_BLOCKED`

All core substrate laws, QWEN36 parity (tiny + real-container), trace semantics,
accounting, sanitizers, defaults-off, and receipts verified independently.
Blocked from full-platform verdict by: (1) DEFECT F1-LIVE-1 liveness edge +
falsified coverage claim, (2) OLMOE real E2E unavailable. No merge/deploy
performed; rollback reference remains frozen f04359a.

Controller next action: fix F1-LIVE-1 on the candidate branch (spin must rescan
FREE buffers, or claim-loop must re-enter instead of parking inside pick_victim;
plus add the missing spin-drain test and build qwen36_trace_replay in promote.sh),
then re-run this verification's gates B/C/H before promotion sequencing.
