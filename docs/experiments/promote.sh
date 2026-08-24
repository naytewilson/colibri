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
# Self-containment: the trace-replay gate below invokes c/qwen36_trace_replay,
# so a fresh clean checkout must build every executable the gate runs.
make -C c qwen36 olmoe qwen36_trace_replay >/dev/null 2>&1 \
  && say "PASS engines + trace-replay build" || { say "FAIL engines + trace-replay build"; FAIL=1; }

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
i=0
for arm in "" "PILOT=1" "COLI_FUSED_LOAD=1" "COLI_EXPERT_ASYNC=1" "COLI_BATCH_ACQ=1"; do
  i=$((i+1))
  # FAIL-CLOSED parity: per-arm unique artifacts, pre-cleared; the known
  # tiny-fixture nonzero engine exit is permitted ONLY if that arm actually
  # produced a fresh nonempty DUMP. A stale/absent dump is a FAIL.
  lb="$TDIR/lb_$i.f32"; lp="$TDIR/lp_$i.f32"
  rm -f "$lb" "$lp"
  brc=0; frc=0
  env SNAP=/tmp/kilo/qwen36_tiny_i8 OMP_NUM_THREADS=4 $arm DUMP="$lb" "$TDIR/qwen36_base" 16 8 /tmp/kilo/ref_qwen36.json >/dev/null 2>&1 || brc=$?
  env SNAP=/tmp/kilo/qwen36_tiny_i8 OMP_NUM_THREADS=4 $arm DUMP="$lp" c/qwen36 16 8 /tmp/kilo/ref_qwen36.json >/dev/null 2>&1 || frc=$?
  fresh() { [[ -s "$1" && "$(stat -f %z "$1" 2>/dev/null || stat -c %s "$1")" -gt 0 ]]; }
  if ! fresh "$lb"; then say "FAIL [$arm] baseline produced no fresh DUMP (rc=$brc)"; FAIL=1; continue; fi
  if ! fresh "$lp"; then say "FAIL [$arm] forge produced no fresh DUMP (rc=$frc)"; FAIL=1; continue; fi
  cmp -s "$lb" "$lp" \
    && say "PASS logits identical [$arm] (rc b/f=$brc/$frc, $(stat -f %z "$lb" 2>/dev/null || stat -c %s "$lb")B)" \
    || { say "FAIL logits differ [$arm] (rc b/f=$brc/$frc)"; FAIL=1; }
done

say "== trace replay =="
rm -f "$TDIR/t3.tsv"
trc=0
SNAP=/tmp/kilo/qwen36_tiny_i8 OMP_NUM_THREADS=2 PILOT=1 COLI_MOE_TRACE="$TDIR/t3.tsv" c/qwen36 16 8 /tmp/kilo/ref_qwen36.json >/dev/null 2>&1 || trc=$?
if [[ -s "$TDIR/t3.tsv" ]] && c/qwen36_trace_replay "$TDIR/t3.tsv" 2>&1 | grep -q TRACE_REPLAY_SELF_CONSISTENT; then
  say "PASS trace replay self-consistent (fresh stream)"
else
  say "FAIL trace replay (missing/stale stream or inconsistent)"; FAIL=1
fi

say ""
if [[ $FAIL -eq 0 ]]; then
  say "VERDICT: PROMGATE_LOCAL_GREEN — ref $(git rev-parse HEAD)"
  say "Stage C real-container A/B is complete; fresh independent verification is still required before any merge or deploy."
else
  say "VERDICT: PROMGATE_RED — do not promote"
fi
exit $FAIL
