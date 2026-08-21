#!/bin/bash
set -e

mkdir -p /home/nayte/bench_results
OUT=/home/nayte/bench_results/matrix_results.txt
echo "=== Qwen3.6-35B-A3B INT3 vs INT4 BOUNDED EVALUATION MATRIX ===" > $OUT
date >> $OUT

PROMPTS=("/home/nayte/prompts/p1_tech.txt" "/home/nayte/prompts/p2_math.txt" "/home/nayte/prompts/p3_code.txt" "/home/nayte/prompts/p4_hist.txt")
NAMES=("p1_tech" "p2_math" "p3_code" "p4_hist")

run_arm() {
    local ARM_NAME=$1
    local SNAP_PATH=$2
    local CAP=$3
    local EXTRA_ENV=$4

    echo "" >> $OUT
    echo "=================================================================" >> $OUT
    echo "=== $ARM_NAME (snap=$SNAP_PATH, cap=$CAP, env=$EXTRA_ENV) ===" >> $OUT
    echo "=================================================================" >> $OUT

    for idx in ${!PROMPTS[@]}; do
        p=${PROMPTS[$idx]}
        pname=${NAMES[$idx]}
        echo "--- Running $ARM_NAME on $pname ---" >> $OUT
        echo "Prompt file: $p" >> $OUT
        
        env $EXTRA_ENV SNAP=$SNAP_PATH OMP_NUM_THREADS=8 COLI_DENSE_I8=1 PILOT=1 TEMP=0 N_NEW=48             /home/nayte/ane-hot/colibri-qwen36/c/qwen36 $CAP 4 $p >> $OUT 2>&1
        echo "" >> $OUT
    done
}

# Arm A: INT4 Control (FP32 embed, INT4 experts, cap128)
run_arm "ARM_A_INT4_FP32_CAP128" "/home/nayte/models/qwen36_i4_gs64" 128 "COLI_EMBED_F32=1"

# Arm B: INT3 Representation (FP16 embed, INT3 experts, cap128)
run_arm "ARM_B_INT3_FP16_CAP128" "/home/nayte/models/qwen36_i3_gs64" 128 ""

# Arm C: INT3 Capacity (FP16 embed, INT3 experts, cap208)
run_arm "ARM_C_INT3_FP16_CAP208" "/home/nayte/models/qwen36_i3_gs64" 208 ""

echo "=== ALL MATRIX EVALUATIONS COMPLETE ===" >> $OUT
date >> $OUT
