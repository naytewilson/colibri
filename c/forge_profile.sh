#!/bin/sh
# B2 (PKG traces-profiler-bench-wrappers): ontology-named step-time profiler
# wrapper. Runs the serving binary under the canonical persistent-serving
# invocation, captures forge_profile_v1 JSON via COLI_FORGE_PROFILE, then
# verifies ONTOLOGY PARITY against ids DERIVED from the canonical registry.
#
# B2R2/R4: two-phase structural extraction — membership comes from the actual
# `public static let metrics: [MetricDefinition]` array only. Quoted forge.*
# strings elsewhere in the source (comments, examples, substitution rules,
# unregistered MetricID declarations) are NOT membership and cannot satisfy
# this gate. Fail-closed (exit 5) on any unprovable parsing state. NO PYTHON.
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

fail5() { echo "ONTOLOGY_SOURCE_INVALID: $*" >&2; exit 5; }

WORKDIR=$(dirname "$OUT")

# ---- R4 PHASE 0: locate canonical source ---------------------------------
ONT_SRC="${FORGE_ONTOLOGY_SRC:-$HOME/.forge-ontology/MetricOntology.swift}"
[ -f "$ONT_SRC" ] || fail5 "source missing: $ONT_SRC (set FORGE_ONTOLOGY_SRC)"
[ -s "$ONT_SRC" ] || fail5 "source empty: $ONT_SRC"
ONT_SHA=$(sha256sum "$ONT_SRC" 2>/dev/null | awk '{print $1}')
[ -n "$ONT_SHA" ] || ONT_SHA=$(shasum -a 256 "$ONT_SRC" 2>/dev/null | awk '{print $1}')
[ -n "$ONT_SHA" ] || fail5 "cannot hash $ONT_SRC"

# ---- R4 PHASE 1: metrics registry slice ----------------------------------
REG_START=$(grep -n 'public static let metrics' "$ONT_SRC" | head -1 | cut -d: -f1)
[ -n "$REG_START" ] || fail5 "metrics registry start not found"
TAIL="$WORKDIR/.b2r2_regtail.$$"
sed -n "$((REG_START + 1)),\$p" "$ONT_SRC" > "$TAIL"
REL_END=$(awk '/^ *\] *$/{print NR; exit}' "$TAIL")
[ -n "$REL_END" ] || { rm -f "$TAIL"; fail5 "metrics registry termination not provable"; }
SLICE="$WORKDIR/.b2r2_slice.$$"
sed -n "1,${REL_END}p" "$TAIL" > "$SLICE"
rm -f "$TAIL"

SYMS=$(grep -oE 'id: [A-Za-z][A-Za-z0-9_]*' "$SLICE" | awk '{print $2}' | sort -u)
rm -f "$SLICE"
[ -n "$SYMS" ] || fail5 "zero MetricDefinition ids extracted from registry"

# ---- R4 PHASE 2: resolve registry symbols via anchored declarations -------
IDS_FILE="$WORKDIR/.b2r2_ids.$$"
: > "$IDS_FILE"
for s in $SYMS; do
  m=$(grep -cE "^ *(public |internal |private )?static let $s *= *try! MetricID\(\"forge\.[a-z0-9_.]+\"\)" "$ONT_SRC" || true)
  [ "$m" -eq 1 ] || { rm -f "$IDS_FILE"; fail5 "registry symbol '$s' resolves to $m MetricID declarations (need exactly 1)"; }
  id=$(grep -E "^ *(public |internal |private )?static let $s *= *try! MetricID\(\"" "$ONT_SRC" \
       | sed -E 's/.*MetricID\("([^"]+)"\).*/\1/')
  case "$id" in
    forge.[a-z0-9_.]*) ;;
    *) rm -f "$IDS_FILE"; fail5 "registry symbol '$s' resolved to non-canonical-shaped id '$id'" ;;
  esac
  printf '%s\n' "$id" >> "$IDS_FILE"
done

TOT=$(wc -l < "$IDS_FILE" | tr -d ' ')
UNIQ=$(sort -u "$IDS_FILE" | wc -l | tr -d ' ')
[ "$TOT" -ge 1 ] || { rm -f "$IDS_FILE"; fail5 "empty registry resolution"; }
[ "$UNIQ" -eq "$TOT" ] || { rm -f "$IDS_FILE"; fail5 "duplicate canonical ids in registry ($TOT entries, $UNIQ unique)"; }
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
  grep -qx "$k" "$IDS_FILE" || {
    echo "ONTOLOGY_PARITY_FAIL: '$k' is not a registered metric of $ONT_SRC (kill-rule: never invent local names)" >&2; rc=1; }
done <<EOF
$KEYS
EOF
rm -f "$IDS_FILE"

grep -q '"device_of_record"' "$OUT" || { echo "DEVICE_STAMP_MISSING in $OUT" >&2; rc=1; }

echo "ontology_source=$ONT_SRC"
echo "ontology_source_identity=$ONT_SHA"
echo "registry_ids=$TOT keys=$(printf '%s\n' $KEYS | wc -l | tr -d ' ') device_of_record_st_dev=$DEV omp_num_threads=$OMP_N"
exit $rc
