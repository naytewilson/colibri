# FORGE F1 — QWEN36 WORKER LIFECYCLE + TELEMETRY HARDENING (2026-08-24)

**Lane**: parallel non-blocking builder (QWEN36 reliability/telemetry hardening)
**Branch**: `fix/forge-f1-qwen-worker-telemetry-20260824`
**Start (exact)**: `0b789e06eb6448cd16bf904b23cc66e565f44295` (verified Forge candidate; untouched)
**Verifier receipt consumed**: `verify/forge-f1-liveness-fix-20260824 @ 8cc8b1e1288c811dd65f1d4d3e8334863c10641a`
(`docs/experiments/VERIFIER_RECEIPT_FORGE_F1_LIVENESS_FIX_20260824.md`)
**Final head**: see GIT section. **Shared runtime mutated: NO** (production diff touches
`c/qwen36.c`, `c/Makefile`, `c/tests/test_qwen36_lifecycle.c` only).

---

## 1. Pre-fix crash reproduction (Mission D)

Bounded repeated-exit harness: `/tmp/kilo/qwen_telemetry_hardening/teardown_stress.sh`
(tiny i8 container `/tmp/kilo/qwen36_tiny_i8`, ref-mode generation to completion,
fd-level rc classification, 60 s hang watchdog via kill -9).

Binary under test built pristine at start SHA:
`qwen36_prefix_0b789e0` sha256 `a765420751e6c751761e2ea6c56bda5787268408a654e5dfffa32d6a9f33d9f7`.

| Arm | Runs | SIGSEGV | Hang | Notes |
|---|---|---|---|---|
| PILOT=1 W=8 cap=2 trace=1 | 20 | **2** | 0 | matches verifier rate (1–2/10) |
| PILOT=1 W=8 cap=2 trace=0 | 20 | **3** | 0 | |
| PILOT=0 control (W=8 cap=2) | 20 | **0** | 0 | crash is pilot-worker-gated |

Output-completion proof: full report (hit rate / coalesce diagnostics / speed) is on
stderr before the crash in every observed case; tokens always correct — same shape as
the verifier's observation. Canonical crash signature: verifier's
`~/Library/Logs/DiagnosticReports/qwen36-2026-08-24-145019.ips`, faulting frame
`pilot_worker` (`EXC_BAD_ACCESS / KERN_INVALID_ADDRESS`). Local stress runs did not
persist new `.ips` files (macOS report throttling); rc=139 classification is from the
harness.

## 2. Source root cause (lifecycle)

`ensure_pilot_worker_started()` created up to 8 `pthread_detach`ed `while(1)` workers
dereferencing global `pilot_m` (which points at `main()`'s stack `Model`). No stop
flag, no join, no shutdown protocol on any normal exit path. After `main()` returns,
workers keep claiming queue jobs and calling `pilot_realload(pilot_m, …)` — malloc,
store ops, and `trace_emit` into an already-`fclose`d stream — while libc tears down;
detached-thread use-after-teardown ⇒ SIGSEGV. The shadow `spec_loader` had the SAME
ownership defect (detached infinite loop dereferencing global `pilot_m`), so both
worker classes were fixed together.

## 3. Lifecycle architecture (Missions A/B/C)

One ownership law, one idempotent boundary:

- Worker handles stored: `g_pilot_thr[PILOT_MAX_WORKERS]` (joinable, never detached),
  `g_spec_thr` + `g_spec_thr_live`. Startup stays idempotent (`!pilot_m && !n`) and
  honours the full `COLI_PILOT_W` 1..8 range.
- Cooperative stop flag `g_workers_stop`; both worker loops observe it with ACQUIRE
  **before claiming new work**. In-flight jobs always run to their terminal
  publish/abort; queued-but-unclaimed entries are cancelled by construction.
- `background_workers_shutdown()`: atomic-exchange idempotence gate → join pilots →
  join spec loader → report joined/cancelled counts. Deterministic order; joins never
  hold a mutex a worker needs.
- Shutdown ordering vs streams: called immediately after generation completes on EVERY
  normal exit path (serve return, corpus end, PPL sample + single, normal generate,
  plus error-free early returns post-model_init), strictly BEFORE trace fclose /
  reports — no worker can touch model/store/FILE state afterwards.
- No process-exit timing used as synchronization; no sleeps added for correctness.

Post-fix repeated exits (same harness): PILOT trace/no-trace 60+60 = **120/120 clean,
0 SIGSEGV, 0 hang**; spec-depth arm 20/20 clean; ASan engine teardown 20/20 clean;
corpus multi-request exits clean (Mac tiny + node real container ×4).

## 4. Spec-loader audit → TWO additional pre-existing defects found & closed

Auditing the shadow path per Mission A ("do not fix only one worker class") exposed:

**(a) INT8-container spec staging SIGSEGV** — PROVEN pre-existing at `0b789e0`
(pre-fix binary reproduces **10/10** with `COLI_SPEC_DEPTH=2`, crash during first
decode, lldb: main-thread memmove from address 0x0 inside `expert_acquire`).
Root cause: `spec_serve()` assumed packed slots — it formatted the lease as INT4 for
any non-INT3 ring entry and memcpy'd from `spec_ring[r].s.w4`, which is NULL when the
container is unpacked INT8 (ring slot holds `g/u/d`). Hidden forever because
speculation is default-OFF and no prior gate ran with `COLI_SPEC_DEPTH>0`.
Fix: format-faithful lease (INT3→w3, INT4→w4, INT8→g), source pointers guarded.
Additionally the copy-out now runs under `g_pilot_mx`, closing the steal-during-copy
race against `spec_issue` ring restyling.
Post-fix: spec-depth arm 20/20 clean; speculation on/off produces **bit-identical**
logit DUMPs (routing-untouched invariant preserved).

**(b) Reservation-leak stall** — source-proven; unmasked once (a) was fixed.
In `expert_acquire`, a successful `reserve()` falls through into `miss_path`, where
`spec_serve()` could satisfy the acquisition from the shadow ring and return
`ACQ_HIT` — leaking the live reservation. Every later acquirer of that key then spins
forever in the BUSY coalesce poll (observed as a hang via live thread sample).
Fix: shadow serve is attempted only when `!ar->has_xres`.

## 5. Telemetry source-of-truth map (Missions E/F)

Inventory and disposition of every QWEN-printed metric whose meaning changed under
Forge (`load_expert_merged` is shadow/spec-only; demand flows through
`qw_store_load -> coli_expert_backend_pread_load`):

| Metric | Class | Disposition |
|---|---|---|
| `[expert_store] requests/logical/hits/misses/coalesced` | shared-store physical | NEW authoritative block (unconditional) |
| `[expert_store] physical_loads/admitted bytes(w+s)/admission_ms avg` | shared-store physical | NEW authoritative |
| `[expert_store] resident/capacity/publishes/aborts/reservations_active/pinned/prefetch_requests/bytes_read` | shared-store physical | NEW authoritative |
| "Admitted bytes (INT3/INT4)" (perf block) | legacy private-cache byte counters | REMOVED (dead zeros); replaced by store-derived line |
| `[expert_io] demand:` loads/bytes/ms | legacy private-sync demand loads | nonzero-only, relabeled `legacy demand sync-loads << UNEXPECTED under Forge store path` |
| `[expert_io] pilot:` loads/bytes/ms | spec-shadow private-slot staging | nonzero-only, relabeled `spec-shadow private-slot loads (… not store admissions)` |
| "Admission (demand NVMe I/O)" ms/token + Slot Acquisition Decomposition w/s split | engine timing of removed sync path | rewired to live store-load interval (`expert_finish` around `qw_store_load`); dead weight/scale ms split dropped |
| "packed INTx expert CPU residency active" banners | container residency truth | moved to first real DEMAND admission through the store (old site fired only for shadow loads); shadow-site banners relabeled `spec-shadow slot staging` |
| decode-window hit/miss, routed format mix, cache hit/miss by fmt, coalesce waits/skips | engine logical/routing | kept (truthful engine intent) |

No second physical-load counter invented: QWEN prints store stats verbatim.

### ALLOCATED EXPERT SLOTS basis fix (Mission F)

The single conflated line reported container census (n_layers × n_experts) next to
resident BYTES. Now separated:

```
EXPERT CONTAINER CENSUS (%llu experts: %d INT3, %d INT4, %d INT8)
CACHE SLOT CAPACITY (%llu slots: %d layers x cap budget): <capacity_bytes>
STORE RESIDENCY (live, pinned %llu): <resident_bytes>
```

Tiny run before → after example:

```
- ALLOCATED EXPERT SLOTS (64 slots: 0 INT3, 0 INT4): 106496 bytes ...
+ EXPERT CONTAINER CENSUS (64 experts: 0 INT3, 0 INT4, 64 INT8)
+ CACHE SLOT CAPACITY (16 slots: 8 layers x cap budget):   106496 bytes
+ STORE RESIDENCY (live, pinned 0):                        106496 bytes
```

(64 = container experts; 16 = 8 layers × cap 2 — previously conflated.)

Real-container after example (node-02, canonical mixed-low container):

```
[expert_store] requests 5472 (logical 5472) | hits 5472 misses 0 | coalesced 0
[expert_store] physical_loads 2848 | admitted 4140.75 MB (weights 3606.75 / scales 534.00) | admission 14452.1 ms (avg 5.07 ms/load)
[expert_store] resident 4140.75 of 8640.00 MB | publishes 2848 aborts 0 reservations_active 0 pinned 0 prefetch_requests 0 bytes_read 4140.75 MB
```

where the pre-fix build printed zero loads / zero admitted bytes / no residency truth.

## 6. Tests (Mission G)

New `c/tests/test_qwen36_lifecycle.c` (17 checks, all green; included in test-c):

- startup honours COLI_PILOT_W exactly; idempotent re-ensure; ownership bound to Model*
- shutdown sets flag; **queue state cannot be consumed after shutdown begins**
  (enqueue-after-stop ⇒ pilot_r frozen over 200 ms)
- second shutdown silent no-op (idempotence); 200× repeated shutdown non-blocking
- telemetry contracts vs stub store: zero loads ⇒ printed zero; defaults-off ⇒ NO fake
  demand/spec-shadow lines; printed numbers agree EXACTLY with stats(); nonzero shadow
  counter prints only under its truthful label + disclaimer

Integration discriminators (documented commands, receipt-run like promgate):
cap=2 churn (PILOT stress arms) ⇒ physical_loads/admitted_bytes nonzero; coalescing ⇒
trace replay conservation (below); slot-basis separation asserted by parsing the new
memory-accounting block.

## 7. Regression gates (Mission H)

| Gate | Result |
|---|---|
| `make -C c qwen36` / `qwen36_trace_replay` | PASS (clean build) |
| `make -C c -k test-c TEST_EXCLUDE="test_uring test_qwen36_i4_kernel"` (arm64 excludes) | PASS rc=0, 0 failures |
| H1/H2 liveness (`test_pread_liveness`) | ok 100×100, rc=0 |
| Tiny five-arm exact parity vs verified `0b789e0` build | 5/5 bit-identical DUMPs (default/PILOT/FUSED/ASYNC/BATCH), rc 1/1 known fixture property |
| Trace replay self-consistency | `TRACE_REPLAY_SELF_CONSISTENT` |
| Accounting conservation | INSERT rows 43 DEMAND + 21 PILOT = **64** == store `physical_loads` == `publishes`; aborts 0; reservations_active 0; insert bytes == admitted bytes == bytes_read (no double charge) |
| Defaults-off | five-arm default arm bit-identical; no new env knobs; contract tests pin zero-line behaviour |
| Repeated teardown stress | 120/120 PILOT + 20/20 spec-depth + corpus exits clean (§1/§3) |
| ASan + UBSan (`-fsanitize=address,undefined -fno-sanitize-recover=all`) | lifecycle suite rc=0; liveness suite rc=0; ENGINE 20/20 teardown + 20/20 spec-depth, ZERO findings (LeakSanitizer unsupported on macOS — noted, not skipped silently) |
| TSAN (gcc, node-02) | lifecycle test rc=0 + liveness 100×100 rc=0; ZERO `WARNING: ThreadSanitizer` |
| Real QWEN36 targeted paired parity | RUN — canonical container `/home/nayte/models/qwen36_mixed_low` on anvil-node-02; both engines built gcc `-O3 -march=native -fopenmp -pthread`: candidate sha256 `057ff93237f529c35c0f5f24cf0ee7278238796e3df320990b78b1f0fb94ff23` (**exactly reproduces controller-pinned builder hash**), hardened `ada1a8ce40c55386ceec9549e4a534c6c020c6021eb9f108ddaa60d816b38869`; same prompt, N_NEW=16, DUMP logits **BIT-IDENTICAL**; generated text identical |

Node hygiene: workspace `~/q36_tsan_20260824` (3.0 MB sources/binaries) created then
REMOVED; only small text logs pulled to Mac evidence dir. No bulk pulled.

## 8. Changed files

- `c/qwen36.c` (+206/−54): worker lifecycle block, stop-flag worker loops, spec_loader
  joinable handle, shutdown calls on all normal exit paths, telemetry rewire
  ([expert_store] block, legacy relabels, timing rewire, banner move),
  spec_serve format-faithful mutex-guarded lease, reservation-leak fix, memory
  accounting basis split
- `c/tests/test_qwen36_lifecycle.c` (new, ~200 lines): lifecycle + telemetry contracts
- `c/Makefile` (+3): test rule

Protected surfaces untouched: `expert_backend_pread.{c,h}`, `expert_store.h`,
`admission.{c,h}`, `olmoe.c`, verified candidate branch, verifier branches, OLMoE
evidence branch, main, production deployment.

## 9. Git

- Branch `fix/forge-f1-qwen-worker-telemetry-20260824`, started at exact
  `0b789e06eb6448cd16bf904b23cc66e565f44295`, pushed NON-FORCE to origin.

## 10. Task-class log

LOCAL_LIGHT (small C compiles + tiny-model CPU runs on Mac; real-model parity +
TSAN offloaded to anvil-node-02). Internal-disk telemetry at session start: 40 GiB
free — below the 50 GiB XL-admission floor, hence everything heavy stayed on
anvil-node-02; nothing multi-GB was created locally; scratch confined to
`/tmp/kilo/qwen_telemetry_hardening` (4.3 MB, retained as evidence packet).

---

## FINAL VERDICT

Lifecycle AND telemetry defects closed WITHOUT touching shared runtime — plus two
pre-existing QWEN-only spec-path defects (INT8 shadow segv, reservation-leak stall)
found by the mandated audit and closed within the same QWEN-only boundary:

`QWEN36_POST_FORGE_WORKER_LIFECYCLE_AND_TELEMETRY_HARDENED`

Evidence labels: reproduction rates, gate table rows, hashes, and conservation sums =
PROVEN (commands above re-runnable). Reservation-leak stall mechanism = source-proven
with observed hang; its trigger interleaving = INFERRED from control flow (not
reproduced on demand). macOS LeakSanitizer coverage = UNAVAILABLE (platform), ASan
non-leak modes fully green. Serve-mode startup/shutdown exercise = NOT RUN (gateway
wire protocol needs a client); lifecycle boundary is still installed on that path.
