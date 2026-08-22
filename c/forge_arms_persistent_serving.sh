#!/bin/sh
# B2 (PKG traces-profiler-bench-wrappers): persistent-serving pilot arm,
# anvil-bench-guarded (ab_init / ab_preflight / ab_run / ab_finish).
# Device-of-record stamped. Fails closed if the guard is absent.
#
# usage: forge_arms_persistent_serving.sh <snap_dir> [cap_tokens]
set -eu

SNAP_DIR="${1:?usage: forge_arms_persistent_serving.sh <snap_dir> [cap_tokens]}"
CAP="${2:-64}"
CACHE="${CACHE:-128}"; EBITS="${EBITS:-4}"; N_NEW="${N_NEW:-64}"; WARMUP="${WARMUP:-128}"
CORPUS_FILE="${CORPUS_FILE:-/home/nayte/bench_results/p1_live_ab/corpus4.txt}"
WARMUP_TXT="${WARMUP_TXT:-/home/nayte/bench_results/p1_live_ab/warmup.txt}"

GUARD="${ANVIL_BENCH_GUARD:-$HOME/ANVIL/tools/anvil-bench/anvil_bench_guard.sh}"
[ -f "$GUARD" ] || { echo "B2-ARMS: bench guard missing at $GUARD (set ANVIL_BENCH_GUARD)" >&2; exit 2; }
# shellcheck disable=SC1090
. "$GUARD"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BIN="$SCRIPT_DIR/qwen36"
[ -x "$BIN" ] || { echo "B2-ARMS: $BIN not built" >&2; exit 2; }
[ -d "$SNAP_DIR" ] || { echo "B2-ARMS: snap dir missing: $SNAP_DIR" >&2; exit 2; }

# G2 result-row contract for this tool: the ontology-named profile artifact
# must actually be emitted by the serving run.
AB_RESULT_RE='^\[forge\] profile written:'
export AB_RESULT_RE

DEV=$(stat -c %d "$SNAP_DIR")
OUT="$(pwd)/forge-b2-persistent-serving-$(date +%Y%m%dT%H%M%S)"

ab_init "$OUT"
ab_preflight "$BIN"
echo "[b2-arms] device_of_record st_dev=$DEV model=$SNAP_DIR cap=$CAP"

ab_run "persistent-serving-pilot" \
  "env COLI_TIMERS=1 COLI_FORGE_TRACE='$OUT/trace.jsonl' COLI_FORGE_PROFILE='$OUT/profile.json' SNAP='$SNAP_DIR' CORPUS_FILE='$CORPUS_FILE' N_NEW='$N_NEW' WARMUP='$WARMUP' PILOT=1 COLI_EXPERT_PARALLEL=1 OMP_NUM_THREADS=\${OMP_NUM_THREADS:-8} LC_ALL=C.UTF-8 '$BIN' '$CACHE' '$EBITS' '$WARMUP_TXT' 2>&1"

ab_finish
