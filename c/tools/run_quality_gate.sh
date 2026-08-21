#!/bin/bash
set -euo pipefail

EVAL_JSON="/home/nayte/prompts/heldout_eval.json"
OUT_DIR="/home/nayte/eval_quality_runs"
mkdir -p "$OUT_DIR"

echo "=== RUNNING COMPACT QUANTITATIVE QUALITY GATE (TF-NLL / PPL) ==="

# 1. Uniform INT3
echo "[1/4] Evaluating Uniform INT3 (/home/nayte/models/qwen36_i3_gs64_clean)..."
PPL=1 SNAP="/home/nayte/models/qwen36_i3_gs64_clean" OMP_NUM_THREADS=8 COLI_DENSE_I8=1 PILOT=1 /home/nayte/ane-hot/colibri-qwen36/c/qwen36 128 4 "$EVAL_JSON" > "$OUT_DIR/nll_uniform_int3.txt" 2>&1
cat "$OUT_DIR/nll_uniform_int3.txt" | grep -E "TF-NLL|Expert cache hit rate|Speed|PEAK RSS"

# 2. Mixed-Low (15% INT4)
echo "[2/4] Evaluating Mixed-Low (/home/nayte/models/qwen36_mixed_low)..."
PPL=1 SNAP="/home/nayte/models/qwen36_mixed_low" OMP_NUM_THREADS=8 COLI_DENSE_I8=1 PILOT=1 /home/nayte/ane-hot/colibri-qwen36/c/qwen36 128 4 "$EVAL_JSON" > "$OUT_DIR/nll_mixed_low.txt" 2>&1
cat "$OUT_DIR/nll_mixed_low.txt" | grep -E "TF-NLL|Expert cache hit rate|Speed|PEAK RSS"

# 3. Mixed-Med (30% INT4)
echo "[3/4] Evaluating Mixed-Med (/home/nayte/models/qwen36_mixed_med)..."
PPL=1 SNAP="/home/nayte/models/qwen36_mixed_med" OMP_NUM_THREADS=8 COLI_DENSE_I8=1 PILOT=1 /home/nayte/ane-hot/colibri-qwen36/c/qwen36 128 4 "$EVAL_JSON" > "$OUT_DIR/nll_mixed_med.txt" 2>&1
cat "$OUT_DIR/nll_mixed_med.txt" | grep -E "TF-NLL|Expert cache hit rate|Speed|PEAK RSS"

# 4. Uniform INT4
echo "[4/4] Evaluating Uniform INT4 (/home/nayte/models/qwen36_i4_gs64)..."
PPL=1 SNAP="/home/nayte/models/qwen36_i4_gs64" OMP_NUM_THREADS=8 COLI_DENSE_I8=1 PILOT=1 /home/nayte/ane-hot/colibri-qwen36/c/qwen36 128 4 "$EVAL_JSON" > "$OUT_DIR/nll_uniform_int4.txt" 2>&1
cat "$OUT_DIR/nll_uniform_int4.txt" | grep -E "TF-NLL|Expert cache hit rate|Speed|PEAK RSS"

echo "=== QUALITY GATE COMPLETE ==="
