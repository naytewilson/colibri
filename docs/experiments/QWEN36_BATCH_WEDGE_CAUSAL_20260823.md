# QWEN36 BATCH WEDGE — CAUSAL MECHANISM PROVEN AND FIXED (2026-08-23)

Branch: `diag/qwen36-batch-wedge-20260823-local` (from `fork/diag/qwen36-batch-wedge-20260823` @ 4579d41).
Banked binary rebuilt byte-exact: `dc60ea4204bfb1c5ce9560a41cd6a23bba10b5d22d7bb4e80b9722f0c9be142b`.
Frozen production binary (`6387cfe6…`) used read-only as cross-check; untouched.

## Root cause (PROVEN)

The COLI_BATCH_ACQ prefill pre-pass in `moe()` is **two-phase**: it *reserves*
every unique missing expert first (`slots[n++]`, `eid=-1`, `loading[eid]=slot`)
and only *finishes* (loads+publishes) them after the whole reserve loop. When
the number of unique routed experts in a prefill layer exceeds the layer cache
capacity (`argv[1]`, default launcher 128), the reserve phase fills **every**
slot with unpublished reservations; its own eviction path then classifies all
slots as "in-flight" and enters the wait-for-publish loop — whose publishers
are sequenced strictly after that same loop. Single-threaded, nobody can ever
publish: a self-deadlock.

Live state at wedge (gdb, L0): `n=128 cap=128`, **128/128 slots eid==-1**,
**128 loading[] entries set**, PC inside `expert_acquire` unlock→usleep(1ms)→lock
rescan loop (objdump 0x1fca0-0x1fd67, bound reg ebx=45000).

## Evidence chain

1. `[st] L0 pre-moe` marker FIRES before the hang (cap1.err) — falsifies the
   banked "zero [st] lines ⇒ before step()/pre-MoE" boundary; the BATCH code
   path DOES execute. (Prior wave's zero-marker trace predates its own binary:
   st.log 22:13 < build 22:15.)
2. Native backtrace of wedged process: `usleep ← expert_acquire ← moe ← step`.
3. Strace: pure clock_nanosleep(1ms) stream during wedge, zero preads.
4. Capacity lever: cap=64/128 HANGS; cap=160/256/512 PASS ⇒ unique-miss count
   of some prefill layer ∈ (128,160].
5. cwd does NOT participate: identical hangs from /home/nayte and src-cwd on
   BOTH the diag binary and frozen f04359a under explicit env. The banked
   "ON@src-cwd PASS ×24" cell did not reproduce tonight even with the richer
   launcher env — treat it as unproven.
6. Value-dependence: BATCH=1 wedges, BATCH=0/absent/dummy-var pass.

## Fix (mechanism-driven, minimal)

Drain-before-overflow in the reserve loop: before any reserve that would need
eviction (`cache.n >= cache.cap`), publish outstanding reservations via
`pb_finish_batch` (extracted W<=8 parallel finish). Final flush unchanged.
When uniques fit in cache the code executes exactly one flush — behavior
byte-identical to before. Also fixes malformed early-path diagnostic
(`timeouts=` had no %ld/arg) and a doubled trailing comment.

## Validation

| Arm | Env | Before | After patch |
|---|---|---|---|
| B ×3 | BATCH=1 OMP1 PILOT0 N8 home-cwd cap128 | rc=124 HANG | PASS ×3 |
| A control | flags off | PASS | PASS |
| cap64 | BATCH=1 cap=64 | HANG | PASS |
| rich | FUSED+EP+PILOT+WIDE+OMP8+BATCH | HANG | PASS |

Generated stdout byte-identical to flags-off control in every BATCH arm.
Bounded waits retained as defensive hardening (now never fire on this path;
diagnostics show waits=0 timeouts=0).

## Lessons recorded

- A "zero markers" claim is only as good as the binary that emitted them —
  verify artifact timestamps against the traced binary BEFORE anchoring a
  boundary conclusion on absence-of-output.
- Absence-of-marker proves only "not past marker", never "code path not
  executed" when later markers exist downstream of the suspect region.
- Two-phase acquire-then-publish over a bounded pool deadlocks at pool
  capacity even with ZERO threads racing: concurrency bugs do not require
  concurrency, only ordering.
- Environment/cwd "layout sensitivity" was a red herring here: with fully
  explicit env the defect is deterministic data-dependence (unique-miss count
  vs capacity). Always re-run the claimed environment matrix with env -i
  before accepting layout theories.
