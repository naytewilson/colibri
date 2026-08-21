#!/bin/bash
set -euo pipefail

OUT_DIR="/home/nayte/bench_mixed_results"
mkdir -p "$OUT_DIR"

MODELS=(
    "uniform_int3:/home/nayte/models/qwen36_i3_gs64_clean"
    "mixed_low_15pct:/home/nayte/models/qwen36_mixed_low"
    "uniform_int4:/home/nayte/models/qwen36_i4_gs64"
)

PROMPTS=(
    "p1_tech:/home/nayte/prompts/p1_tech.txt"
    "p3_code:/home/nayte/prompts/p3_code.txt"
    "p5_reasoning:/home/nayte/prompts/p5_reasoning.txt"
    "p9_json:/home/nayte/prompts/p9_json.txt"
)

echo "=== RUNNING 64-TOKEN SCORED A/B BENCHMARK CAMPAIGN ==="

for mentry in "${MODELS[@]}"; do
    mname="${mentry%%:*}"
    mdir="${mentry##*:}"
    echo "========================================================"
    echo "Testing Model Configuration: $mname ($mdir)"
    echo "========================================================"

    for pentry in "${PROMPTS[@]}"; do
        pname="${pentry%%:*}"
        pfile="${pentry##*:}"
        out_log="$OUT_DIR/bench_${mname}_${pname}.log"

        echo "[RUN] $mname on $pname -> $out_log"
        SNAP="$mdir" OMP_NUM_THREADS=8 COLI_DENSE_I8=1 PILOT=1 COLI_TIMERS=1 TEMP=0 N_NEW=64 \
            /home/nayte/ane-hot/colibri-qwen36/c/qwen36 128 4 "$pfile" > "$out_log" 2>&1

        grep -E "Speed|Warm decode|TTFT|Expert cache hit rate|Total generation|MoE total|DeltaNet|PEAK RSS" "$out_log" || true
    done
done

echo "=== RUNNING 4-TURN PERSISTENT SERVING SIMULATION (MIXED-LOW) ==="
turn_log="$OUT_DIR/persistent_serving_mixed_low.log"
SNAP="/home/nayte/models/qwen36_mixed_low" OMP_NUM_THREADS=8 COLI_DENSE_I8=1 PILOT=1 COLI_TIMERS=1 TEMP=0 N_NEW=64 \
    /home/nayte/ane-hot/colibri-qwen36/c/qwen36 128 4 \
    /home/nayte/prompts/p1_tech.txt \
    /home/nayte/prompts/p3_code.txt \
    /home/nayte/prompts/p5_reasoning.txt \
    /home/nayte/prompts/p9_json.txt > "$turn_log" 2>&1

grep -E "Turn|Speed|Warm decode|TTFT|Expert cache hit rate|Total generation|PEAK RSS" "$turn_log" || true

echo "=== BENCHMARK CAMPAIGN COMPLETE ==="
