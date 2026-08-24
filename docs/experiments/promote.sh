#!/usr/bin/env bash
# FORGE F1 promotion gate (builder-side). Verifies the working tree against
# the frozen baseline and prints a promotion packet. Never mutates deploys.
set -euo pipefail
cd "$(dirname "$0")/../.."

BASE=f04359aab31a388cc47d36e96c3ea36400061cb6
FAIL=0
say() { printf '[promgate] %s\n' "$*"; }
chk() { if eval "$2"; then say "PASS $1"; else say "FAIL $1"; FAIL=1; fi; }

chk "branch is forge/f1-ooc-moe-runtime" "[[ \"\$(git branch --show-current)\" == forge/f1-ooc-moe-runtime ]]"
chk "baseline is ancestor" "git merge-base --is-ancestor $BASE HEAD"
chk "clean tree" "[[ -z \"\$(git status --porcelain)\" ]]"

say "== build =="
make -C c qwen36 olmoe >/dev/null 2>&1 && say "PASS engines build" || { say "FAIL engines build"; FAIL=1; }

say "== unit gates (arm64 local excludes: test_uring, x86-only i4_kernel) =="
make -C c -k test-c TEST_EXCLUDE="test_uring test_qwen36_i4_kernel" >/dev/null 2>&1 \
  && say "PASS make test-c" || { say "FAIL make test-c"; FAIL=1; }

say "== tiny parity A/B vs frozen baseline =="
TDIR=/tmp/kilo/forge_promgate
mkdir -p "$TDIR"
if [[ ! -d /tmp/kilo/qwen36_tiny_i8 ]]; then
  (cd c && python3 tools/make_qwen36_tiny.py --out /tmp/kilo/qwen36_tiny --emit-ref /tmp/kilo/ref_qwen36.json >/dev/null 2>&1 \
    && python3 tools/convert_qwen36.py --model /tmp/kilo/qwen36_tiny --out /tmp/kilo/qwen36_tiny_i8 --ebits 8 >/dev/null 2>&1)
fi
for f in qwen36.c st.h json.h compat.h; do
  git show "$BASE:c/$f" > "$TDIR/$f" || { say "FAIL baseline extract $f"; exit 1; }
done
clang -O3 -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include "$TDIR"/qwen36.c -o "$TDIR/qwen36_base" -lm -L/opt/homebrew/opt/libomp/lib -lomp
for arm in "" "PILOT=1" "COLI_FUSED_LOAD=1" "COLI_EXPERT_ASYNC=1" "COLI_BATCH_ACQ=1"; do
  env SNAP=/tmp/kilo/qwen36_tiny_i8 OMP_NUM_THREADS=4 $arm DUMP="$TDIR/lb.f32" "$TDIR/qwen36_base" 16 8 /tmp/kilo/ref_qwen36.json >/dev/null 2>&1
  env SNAP=/tmp/kilo/qwen36_tiny_i8 OMP_NUM_THREADS=4 $arm DUMP="$TDIR/lp.f32" c/qwen36 16 8 /tmp/kilo/ref_qwen36.json >/dev/null 2>&1
  cmp -s "$TDIR/lb.f32" "$TDIR/lp.f32" && say "PASS logits identical [$arm]" || { say "FAIL logits [$arm]"; FAIL=1; }
done

say "== trace replay =="
SNAP=/tmp/kilo/qwen36_tiny_i8 OMP_NUM_THREADS=2 PILOT=1 COLI_MOE_TRACE="$TDIR/t3.tsv" c/qwen36 16 8 /tmp/kilo/ref_qwen36.json >/dev/null 2>&1 || true
c/qwen36_trace_replay "$TDIR/t3.tsv" 2>&1 | grep -q TRACE_REPLAY_SELF_CONSISTENT \
  && say "PASS trace replay self-consistent" || { say "FAIL trace replay"; FAIL=1; }

say ""
if [[ $FAIL -eq 0 ]]; then
  say "VERDICT: PROMGATE_LOCAL_GREEN — ref $(git rev-parse HEAD)"
  say "Stage C real-container A/B + independent verification still required before deploy."
else
  say "VERDICT: PROMGATE_RED — do not promote"
fi
exit $FAIL
