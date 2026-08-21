#!/bin/bash
set -e

MODEL_I4="/home/nayte/models/qwen36_i4_gs64"
MODEL_I3_CLEAN="/data/ANVIL/models/qwen36_i3_gs64_clean"
BIN="/home/nayte/ane-hot/colibri-qwen36/c/qwen36"
RESULTS_DIR="/home/nayte/bench_results"

mkdir -p "$RESULTS_DIR"
mkdir -p /home/nayte/prompts

# Ensure corpus file exists for persistent serving
cat << 'EOF' > /home/nayte/prompts/corpus_persistent_serving.txt
Explain the concept of speculative decoding in large language models and how draft models reduce memory bandwidth bottlenecks.
===PROMPT===
Derive the matrix exponentiation algorithm for calculating the n-th Fibonacci number in O(log n) time.
===PROMPT===
Write a C function that uses Linux epoll to manage 10,000 non-blocking TCP socket connections with edge-triggered events.
===PROMPT===
Describe the historical evolution of liquid neural networks and continuous-time recurrent models.
EOF

echo "================================================================="
echo "=== 1. VERIFYING CONVERTER SHARDS (PHASE 3) ==="
echo "================================================================="
/home/nayte/ane-hot/colibri-qwen36/c/tools/verify_int3_converter "$MODEL_I3_CLEAN" 0
/home/nayte/ane-hot/colibri-qwen36/c/tools/verify_int3_converter "$MODEL_I3_CLEAN" 1
/home/nayte/ane-hot/colibri-qwen36/c/tools/verify_int3_converter "$MODEL_I3_CLEAN" 3

echo ""
echo "================================================================="
echo "=== 2. TRUE HIGH-CAP RESIDENCY PROOF (PHASE 6) ==="
echo "================================================================="
echo "--- Running Cap 208 Full Slot Physical Touch & Commitment ---"
env SNAP="$MODEL_I3_CLEAN" OMP_NUM_THREADS=8 COLI_DENSE_I8=1 PILOT=1 TEMP=0 N_NEW=4 \
    COLIBRI_PROBE_TOUCH_SLOTS=1 \
    "$BIN" 208 4 /home/nayte/prompts/p1_tech.txt > "$RESULTS_DIR/cap208_probe.log" 2>&1
grep -A 2 -B 2 "PROBE" "$RESULTS_DIR/cap208_probe.log"
grep "M2" "$RESULTS_DIR/cap208_probe.log" || true

echo ""
echo "================================================================="
echo "=== 3. PERSISTENT COLIBRÌ SERVING GATE (PHASE 7) ==="
echo "================================================================="
echo "--- Initial Hardware Sensors ---"
sensors 2>/dev/null || true
echo "--- Running Sequential 4-Turn Persistent Serving (Cap 208) ---"
env SNAP="$MODEL_I3_CLEAN" OMP_NUM_THREADS=8 COLI_DENSE_I8=1 PILOT=1 COLI_TIMERS=1 TEMP=0 N_NEW=64 \
    CORPUS_FILE=/home/nayte/prompts/corpus_persistent_serving.txt \
    "$BIN" 208 4 /home/nayte/prompts/corpus_persistent_serving.txt > "$RESULTS_DIR/persistent_serving_cap208.log" 2>&1
cat "$RESULTS_DIR/persistent_serving_cap208.log"
echo "--- Post-Run Hardware Sensors ---"
sensors 2>/dev/null || true

echo ""
echo "================================================================="
echo "=== 4. CLEAN A/B PERFORMANCE MATRIX (PHASE 8 & 9) ==="
echo "================================================================="

PROMPTS=("/home/nayte/prompts/p1_tech.txt" "/home/nayte/prompts/p2_math.txt" "/home/nayte/prompts/p3_code.txt" "/home/nayte/prompts/p4_hist.txt")
PNAMES=("p1_tech" "p2_math" "p3_code" "p4_hist")

run_perf_arm() {
    local ARM_NAME=$1
    local SNAP_PATH=$2
    local CAP=$3
    local EXTRA_ENV=$4
    local OUT_FILE="$RESULTS_DIR/perf_${ARM_NAME}.log"

    echo "=== Running $ARM_NAME (snap=$SNAP_PATH, cap=$CAP) ===" > "$OUT_FILE"
    for idx in ${!PROMPTS[@]}; do
        p=${PROMPTS[$idx]}
        pname=${PNAMES[$idx]}
        echo "--- Running $ARM_NAME on $pname ---" >> "$OUT_FILE"
        env $EXTRA_ENV SNAP="$SNAP_PATH" OMP_NUM_THREADS=8 COLI_DENSE_I8=1 PILOT=1 COLI_TIMERS=1 TEMP=0 N_NEW=64 \
            "$BIN" $CAP 4 "$p" >> "$OUT_FILE" 2>&1
    done
    echo "Completed $ARM_NAME"
}

run_perf_arm "ARM_A_INT4_CAP128" "$MODEL_I4" 128 ""
run_perf_arm "ARM_B_INT3_CAP128" "$MODEL_I3_CLEAN" 128 ""
run_perf_arm "ARM_C_INT3_CAP208" "$MODEL_I3_CLEAN" 208 ""

echo ""
echo "================================================================="
echo "=== 5. QUANTITATIVE QUALITY EVALUATION GATE (PHASE 5) ==="
echo "================================================================="

EVAL_PROMPTS=(
    "/home/nayte/prompts/p1_tech.txt"
    "/home/nayte/prompts/p2_math.txt"
    "/home/nayte/prompts/p3_code.txt"
    "/home/nayte/prompts/p4_hist.txt"
    "/home/nayte/prompts/p5_reasoning.txt"
    "/home/nayte/prompts/p6_instruct.txt"
    "/home/nayte/prompts/p7_summary.txt"
    "/home/nayte/prompts/p8_knowledge.txt"
    "/home/nayte/prompts/p9_json.txt"
)
EVAL_NAMES=(
    "p1_tech"
    "p2_math"
    "p3_code"
    "p4_hist"
    "p5_reasoning"
    "p6_instruct"
    "p7_summary"
    "p8_knowledge"
    "p9_json"
)

QUALITY_LOG="$RESULTS_DIR/quality_comparison.log"
echo "=== QUALITY EVALUATION: CONTROL (INT4) vs CANDIDATE (DIRECT INT3) ===" > "$QUALITY_LOG"

for idx in ${!EVAL_PROMPTS[@]}; do
    p=${EVAL_PROMPTS[$idx]}
    pname=${EVAL_NAMES[$idx]}
    echo "" >> "$QUALITY_LOG"
    echo "=================================================================" >> "$QUALITY_LOG"
    echo "=== PROMPT $idx: $pname ($p) ===" >> "$QUALITY_LOG"
    echo "=================================================================" >> "$QUALITY_LOG"
    cat "$p" >> "$QUALITY_LOG"
    echo "" >> "$QUALITY_LOG"

    echo ">>> [CONTROL: INT4-g64] Generating on $pname..." >> "$QUALITY_LOG"
    env SNAP="$MODEL_I4" OMP_NUM_THREADS=8 COLI_DENSE_I8=1 PILOT=1 TEMP=0 N_NEW=64 \
        "$BIN" 128 4 "$p" >> "$QUALITY_LOG" 2>&1
    echo "" >> "$QUALITY_LOG"

    echo ">>> [CANDIDATE: DIRECT INT3-g64] Generating on $pname..." >> "$QUALITY_LOG"
    env SNAP="$MODEL_I3_CLEAN" OMP_NUM_THREADS=8 COLI_DENSE_I8=1 PILOT=1 TEMP=0 N_NEW=64 \
        "$BIN" 208 4 "$p" >> "$QUALITY_LOG" 2>&1
    echo "" >> "$QUALITY_LOG"
done

echo "=== ALL EVALUATION PASSES COMPLETE ==="
