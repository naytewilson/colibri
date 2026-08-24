#!/bin/bash
set -euo pipefail

QWEN36_BIN="${QWEN36_BIN:?set QWEN36_BIN to the exact qwen36 binary under test}"
EXPECTED_BIN_SHA256="${EXPECTED_BIN_SHA256:-}"
EVAL_JSON="${EVAL_JSON:-/home/nayte/prompts/heldout_eval.json}"
OUT_DIR="${OUT_DIR:-/home/nayte/eval_quality_runs}"
mkdir -p "$OUT_DIR"

if [[ ! -x "$QWEN36_BIN" ]]; then
  echo "quality gate: binary is not executable: $QWEN36_BIN" >&2
  exit 2
fi
if [[ ! -f "$EVAL_JSON" ]]; then
  echo "quality gate: evaluation corpus missing: $EVAL_JSON" >&2
  exit 2
fi

BIN_SHA256="$(sha256sum "$QWEN36_BIN" | awk '{print $1}')"
EVAL_SHA256="$(sha256sum "$EVAL_JSON" | awk '{print $1}')"
if [[ -n "$EXPECTED_BIN_SHA256" && "$BIN_SHA256" != "$EXPECTED_BIN_SHA256" ]]; then
  echo "quality gate: binary hash mismatch" >&2
  echo "  expected: $EXPECTED_BIN_SHA256" >&2
  echo "  observed: $BIN_SHA256" >&2
  exit 2
fi

PROVENANCE="$OUT_DIR/quality_provenance.txt"
{
  echo "qwen36_bin=$QWEN36_BIN"
  echo "qwen36_bin_sha256=$BIN_SHA256"
  echo "eval_json=$EVAL_JSON"
  echo "eval_json_sha256=$EVAL_SHA256"
  echo "omp_num_threads=8"
  echo "coli_dense_i8=1"
  echo "pilot=1"
} | tee "$PROVENANCE"

echo "=== RUNNING COMPACT QUANTITATIVE QUALITY GATE (TF-NLL / PPL) ==="

run_arm() {
  local label="$1"
  local snap="$2"
  local out="$3"
  if [[ ! -d "$snap" ]]; then
    echo "quality gate: snapshot missing for $label: $snap" >&2
    exit 2
  fi
  echo "$label ($snap)..."
  PPL=1 SNAP="$snap" OMP_NUM_THREADS=8 COLI_DENSE_I8=1 PILOT=1 \
    "$QWEN36_BIN" 128 4 "$EVAL_JSON" > "$out" 2>&1
  grep -E "TF-NLL|Expert cache hit rate|Speed|PEAK RSS" "$out"
}

run_arm "[1/4] Evaluating Uniform INT3" "/home/nayte/models/qwen36_i3_gs64_clean" "$OUT_DIR/nll_uniform_int3.txt"
run_arm "[2/4] Evaluating Mixed-Low (15% INT4)" "/home/nayte/models/qwen36_mixed_low" "$OUT_DIR/nll_mixed_low.txt"
run_arm "[3/4] Evaluating Mixed-Med (30% INT4)" "/home/nayte/models/qwen36_mixed_med" "$OUT_DIR/nll_mixed_med.txt"
run_arm "[4/4] Evaluating Uniform INT4" "/home/nayte/models/qwen36_i4_gs64" "$OUT_DIR/nll_uniform_int4.txt"

echo "=== QUALITY GATE COMPLETE ==="
