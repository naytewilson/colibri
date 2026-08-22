#!/bin/sh
# B2 (PKG traces-profiler-bench-wrappers): ontology-named step-time profiler
# wrapper. Runs the serving binary under the canonical persistent-serving
# invocation (SNAP + corpus + warmup, mirroring the P1 lane driver), captures
# forge_profile_v1 JSON via the COLI_FORGE_PROFILE hook, then verifies
# ONTOLOGY PARITY (every metrics key must exist in the canonical ontology)
# and the device-of-record stamp. NO PYTHON.
#
# usage: forge_profile.sh <snap_dir> <out_json>
# env overrides: CACHE(128) EBITS(4) N_NEW(64) WARMUP(128)
#                CORPUS_FILE WARMUP_TXT OMP_NUM_THREADS COLI_EXPERT_PARALLEL
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

# canonical ontology ids — source of truth:
# tools/model-forge/Sources/ForgeCore/MetricOntology.swift
ONTOLOGY="forge.attention.ms_per_token
forge.lm_head.ms_per_token
forge.memory.dram_bytes_per_token
forge.memory.logical_weight_bytes_per_token
forge.memory.peak_rss_gib
forge.mixer.deltanet_ms_per_token
forge.moe.admission_ms_per_token
forge.moe.demand_hit_rate
forge.moe.expert_addresses_pct
forge.moe.pilot_hit_rate
forge.moe.routed_compute_ms_per_token
forge.moe.routed_compute_pct
forge.moe.routed_int4_share_pct
forge.moe.slot_lookup_ms_per_token
forge.moe.total_ms_per_token
forge.quality.ppl
forge.runtime.step_ms_per_token.total
forge.runtime.step_ms_per_token
forge.storage.nvme_admission_ms_per_token
forge.storage.nvme_read_bytes_per_token
forge.wall.decode_tok_s"

KEYS=$(sed -n 's/^    "\([a-z_.]*\)": .*/\1/p' "$OUT")
rc=0
while IFS= read -r k; do
  [ -z "$k" ] && continue
  printf '%s\n' "$ONTOLOGY" | grep -qx "$k" || {
    echo "ONTOLOGY_PARITY_FAIL: '$k' not in MetricOntology (kill-rule: never invent local names)" >&2; rc=1; }
done <<EOF
$KEYS
EOF

grep -q '"device_of_record"' "$OUT" || { echo "DEVICE_STAMP_MISSING in $OUT" >&2; rc=1; }

echo "device_of_record_st_dev=$DEV omp_num_threads=$OMP_N keys=$(printf '%s\n' $KEYS | wc -l | tr -d ' ')"
exit $rc
