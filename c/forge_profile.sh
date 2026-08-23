#!/bin/sh
# B2 (PKG traces-profiler-bench-wrappers): ontology-named step-time profiler
# wrapper. Runs the serving binary under the canonical persistent-serving
# invocation (SNAP + corpus + warmup), captures forge_profile_v1 JSON via the
# COLI_FORGE_PROFILE hook, then verifies ONTOLOGY PARITY against the LIVE
# canonical ontology source. NO PYTHON.
#
# B2R/R2: the parity allow-list is DERIVED from the canonical
# MetricOntology.swift at run time — no embedded copy exists, so this gate
# cannot silently go stale. Fail-closed when the canonical source is absent.
#
# usage: forge_profile.sh <snap_dir> <out_json>
# env:   FORGE_ONTOLOGY_SRC  (default $HOME/.forge-ontology/MetricOntology.swift)
#        CACHE EBITS N_NEW WARMUP CORPUS_FILE WARMUP_TXT OMP_NUM_THREADS
set -eu

SNAP_DIR="${1:?usage: forge_profile.sh <snap_dir> <out_json>}"
OUT="${2:?output json path required}"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BIN="$SCRIPT_DIR/qwen36"
[ -x "$BIN" ] || { echo "forge_profile: $BIN not built (run make -C c qwen36)" >&2; exit 2; }
[ -d "$SNAP_DIR" ] || { echo "forge_profile: snap dir missing: $SNAP_DIR" >&2; exit 2; }

CACHE="${CACHE:-128}"; EBITS="${EBITS:-4}"; N_NEW="${N_NEW:-64}"; WARMUP="${WARMUP:-128}"
CORPUS_FILE="${CORPUS_FILE:-/home/nayte/bench_results/p1_live_ab/corpus4.txt}"
WARMUP_TXT="${WARMUP_TXT:-/home/nayte/bench_results/p1_live_ab/warmup.txt}"
[ -f "$CORPUS_FILE" ] || { echo "forge_profile: CORPUS_FILE missing: $CORPUS_FILE" >&2; exit 2; }
[ -f "$WARMUP_TXT" ] || { echo "forge_profile: WARMUP_TXT missing: $WARMUP_TXT" >&2; exit 2; }

# ---- R2: canonical ontology source, fail-closed --------------------------
ONT_SRC="${FORGE_ONTOLOGY_SRC:-$HOME/.forge-ontology/MetricOntology.swift}"
[ -f "$ONT_SRC" ] || {
  echo "ONTOLOGY_SOURCE_MISSING: $ONT_SRC — set FORGE_ONTOLOGY_SRC to the canonical tools/model-forge/Sources/ForgeCore/MetricOntology.swift (fail-closed; embedded fallback lists are NON-CANONICAL and not permitted to satisfy this gate)" >&2
  exit 5
}
ONT_SHA=$(sha256sum "$ONT_SRC" 2>/dev/null | awk '{print $1}')
[ -n "$ONT_SHA" ] || ONT_SHA=$(shasum -a 256 "$ONT_SRC" 2>/dev/null | awk '{print $1}')
[ -n "$ONT_SHA" ] || { echo "ONTOLOGY_HASH_FAILED for $ONT_SRC" >&2; exit 5; }
ONTOLOGY=$(grep -oE '"forge\.[a-z0-9_.]+"' "$ONT_SRC" | tr -d '"' | sort -u)
[ -n "$ONTOLOGY" ] || { echo "ONTOLOGY_EXTRACT_EMPTY from $ONT_SRC" >&2; exit 5; }
# --------------------------------------------------------------------------

DEV=$(stat -c %d "$SNAP_DIR")
OMP_N="${OMP_NUM_THREADS:-8}"

LOG="$(dirname "$OUT")/.forge_profile_run.log"
env COLI_TIMERS=1 COLI_FORGE_PROFILE="$OUT" \
    SNAP="$SNAP_DIR" CORPUS_FILE="$CORPUS_FILE" N_NEW="$N_NEW" WARMUP="$WARMUP" \
    PILOT="${PILOT:-1}" COLI_EXPERT_PARALLEL="${COLI_EXPERT_PARALLEL:-1}" \
    OMP_NUM_THREADS="$OMP_N" LC_ALL=C.UTF-8 \
    "$BIN" "$CACHE" "$EBITS" "$WARMUP_TXT" >"$LOG.stdout" 2>"$LOG" || {
  echo "forge_profile: serving binary exited nonzero; stderr tail:" >&2
  tail -5 "$LOG" >&2
  exit 4;
}
[ -s "$OUT" ] || { echo "forge_profile: no profile emitted at $OUT" >&2; exit 3; }

KEYS=$(sed -n 's/^    "\([a-z_.]*\)": .*/\1/p' "$OUT")
rc=0
while IFS= read -r k; do
  [ -z "$k" ] && continue
  printf '%s\n' "$ONTOLOGY" | grep -qx "$k" || {
    echo "ONTOLOGY_PARITY_FAIL: '$k' not in canonical ontology $ONT_SRC (kill-rule: never invent local names)" >&2; rc=1; }
done <<EOF
$KEYS
EOF

grep -q '"device_of_record"' "$OUT" || { echo "DEVICE_STAMP_MISSING in $OUT" >&2; rc=1; }

echo "ontology_source=$ONT_SRC"
echo "ontology_source_identity=$ONT_SHA"
echo "device_of_record_st_dev=$DEV omp_num_threads=$OMP_N keys=$(printf '%s\n' $KEYS | wc -l | tr -d ' ')"
exit $rc
