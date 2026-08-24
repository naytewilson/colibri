# QWEN36 BATCH WEDGE — CLEAN HOTFIX ON FROZEN LINEAGE (2026-08-24)

Branch: `fix/qwen36-batch-wedge-clean-20260824` (base `f04359aab31a388cc47d36e96c3ea36400061cb6`, controller-created).
Candidate binary sha256: see commit message / verification request. Baseline rebuild of clean
f04359a reproduced the frozen promotion binary byte-exactly
(`6387cfe6acfd9e4110bdcafc31d7418dbedf8eabd27e75cc187a547dddbc4421`).

## Fix design (replaces diagnostic a0f0fd1 approach)

Non-blocking reservation protocol instead of slot-count heuristics:

1. New mode `ACQ_MODE_PRELOAD_NB` (+ `ACQ_WOULD_BLOCK` outcome). In
   `expert_acquire()`, only where the eviction path is about to enter the
   wait-for-publish loop (no reservable slot: everything pinned or an
   unpublished reservation), a non-blocking preload now reports
   `ACQ_WOULD_BLOCK` and returns without touching cache state or counters.
2. The BATCH pre-pass reserves through `PRELOAD_NB`. On `WOULD_BLOCK`:
   if it owns unfinished reservations (`nl>0`) it publishes them first
   (parallel flush, width<=8) and retries the SAME expert blocking; with
   `nl==0` the blockage is external and ordinary blocking semantics apply.
3. All mode guards changed from `!= ACQ_MODE_PRELOAD` to
   `== ACQ_MODE_DEMAND` (and their duals) so the new mode stays silent and
   un-counted exactly like `PRELOAD`; `expert_finish` treats both preload
   modes identically for the preload I/O hint and `g_pb_io_us`.

Invariant enforced: the pre-pass can never block on a publication that only
its own unfinished batch could perform. Concurrency is preserved up to the
true safe-reservation frontier — drains fire only when zero reservable slots
exist, never merely because `n == cap`.

Not imported from diagnostic branches: bounded 45s waits, scratch-lease
fallback, deliberate leaks, W3a routing reuse, STEPTRACE markers, stale
layout comments.

## Evidence

Mission A (frozen repro, SHA-bound to 6387cfe6…):
BATCH absent PASS / BATCH=0 PASS / BATCH=1 cap128 HANG(rc124) / cap256 PASS.

Mission E (candidate): cap128 x3 PASS; cap64 x2 PASS; flags-off PASS;
rich profile PASS; stdout byte-identical across all arms; MaxRSS ~10.60 GB
all arms (flat vs baseline).

Mission C/F (frozen serving protocol, 4-prompt corpus N64 OMP8 PILOT1,
counterbalanced two passes, candidate vs frozen promotion binary):
batch_wall ~9.7–10.7s with preload_io_sum ~69–77s on BOTH binaries =>
effective load concurrency ~7.1–7.3x everywhere, including warm-cache
prompts 2–4. Warm-cache full-cap discriminator: no serialization
(candidate prompt walls match frozen within noise). BATCH win preserved:
prompt1 wall 27.0s(off) -> 18.7s(bat) candidate vs 25.5s -> 19.4s frozen.

W9/accounting: cumulative demand acquisitions equal in total (109,120);
hit/miss split shifts by <=10 of 109k between ANY two arms INCLUDING two
frozen-binary arms (f_bat vs f_both differ by 10) — within the protocol's
own LRU victim-order variance; loads==misses==inserts mapping preserved
structurally (every reserved cell finished exactly once via ld[] list).
K<=8 / S<=512 fail-closed gate untouched (line 2634).

One arm (c_bat2, first pass) was SIGKILLed by the kernel global OOM killer
under unrelated node memory pressure (anon-rss 10.3 GB of 14 GB); rerun
completed cleanly rc=0. Not candidate-attributable.

## Regression artifact

cap64 BATCH=1 full-model reproducer terminates and matches control output
(banked as required regression); scripts: /tmp/hotfix_20260824/*.sh on
anvil-node-02.

## CONTROLLER CORRECTION (2026-08-24, post-1971084)

Source review of 1971084 found PB_FLUSH computed `int W_ = nl < 8 ? nl : 8`
but never applied it: `_Pragma("omp parallel for schedule(static)")` lacked
`num_threads(W_)`. With inherited OMP_NUM_THREADS > 8 the flush team could
therefore exceed the historical min(nl,8) software worker cap. OMP8-based
validation could not see this because the env cap already bounds the default
team at 8 — exactly why OMP16/32 discriminators were added.

Correction (commit e70c135): `_Pragma("omp parallel for schedule(static)
num_threads(W_)")`. W_ is consumed; worker count <= 8 regardless of inherited
OMP_NUM_THREADS; nl==1 stays serial; nl>1 stays parallel. Zero new compiler
warnings vs pristine f04359a baseline census (all remaining warnings identical).

Thread-environment discriminator on final binary: BATCH=1 cap128 terminates
with byte-identical stdout and flat MaxRSS at OMP_NUM_THREADS=1 / 8 / 16 / 32;
cap64 x2 and cap128 x2 PASS under OMP1; absent/0 PASS; rich profile PASS.

Performance confirmation (frozen serving protocol, tonight, final binary):
batch_wall 12.06-12.45s with preload_io_sum 87.1-89.9s => 7.2-7.3x effective
load concurrency; live frozen-binary control arm in the same window measured
batch_wall 10.97s / io_sum 79.0s (same 7.2x ratio) while frozen's own earlier
passes spanned 9.95-10.64s — absolute drift is node-load environment, ratio
is drift-resistant and unchanged. No sequentialization; RSS flat ~10.8-10.9 GB
serving profile.

Accounting/compat: make test-c rc=0 (all suites); W9 cumulative totals equal
(109,120 acquisitions both binaries; hit/miss split delta 7 within the frozen
code's own arm-to-arm variance); K<=8/S<=512 gate untouched; no new demand
telemetry from PRELOAD_NB (counters fire on ACQ_MODE_DEMAND only).

1971084 evidence above is preserved unmodified.
