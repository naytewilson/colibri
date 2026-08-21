#!/bin/bash
set -euo pipefail

MODEL_DIR="/home/nayte/models/qwen36_i3_gs64_clean"
PROMPTS_DIR="/home/nayte/prompts"
OUT_DIR="/home/nayte/census_runs"
mkdir -p "$OUT_DIR"

echo "=== Running Multi-Domain Routing Census Campaign ==="

for i in {1..9}; do
    pfile=$(ls "$PROMPTS_DIR"/p${i}_*.txt 2>/dev/null | head -n 1)
    if [ -n "$pfile" ]; then
        name=$(basename "$pfile" .txt)
        cout="$OUT_DIR/census_${name}.json"
        echo "[Domain $i/9] $name -> $cout"
        SNAP="$MODEL_DIR" CENSUS_OUT="$cout" OMP_NUM_THREADS=8 COLI_DENSE_I8=1 PILOT=1 COLI_TIMERS=1 TEMP=0 N_NEW=32             /home/nayte/ane-hot/colibri-qwen36/c/qwen36 128 4 "$pfile" > "$OUT_DIR/output_${name}.txt" 2> "$OUT_DIR/log_${name}.txt"
    fi
done

echo "=== Census Campaign Completed Successfully ==="
ls -lh "$OUT_DIR"/census_*.json
