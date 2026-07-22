#!/bin/bash

set -e

MODEL="/amulpfsdata/models/translategemma-4b-it"

TRAIN_DATA="./data/dpo.jsonl"
TEST_DATA="./data/dpo_test.jsonl"

OUTPUT_ROOT="./outputs"

# -------------------------
# Hyperparameter sweeps
# -------------------------

LEARNING_RATES=(
    5e-5
    3e-5
    2e-5
    1e-5
)

BETA=0.1
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05

EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=8

# -------------------------

mkdir -p "${OUTPUT_ROOT}"

for LR in "${LEARNING_RATES[@]}"
do

    RUN_NAME="lr_${LR}"

    echo
    echo "======================================================"
    echo "Starting experiment: ${RUN_NAME}"
    echo "======================================================"

    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --config_file accelerate_config.yaml \
        dpo_train.py \
        --model_name "${MODEL}" \
        --train_data "${TRAIN_DATA}" \
        --test_data "${TEST_DATA}" \
        --learning_rate "${LR}" \
        --beta "${BETA}" \
        --lora_r "${LORA_R}" \
        --lora_alpha "${LORA_ALPHA}" \
        --lora_dropout "${LORA_DROPOUT}" \
        --num_epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --grad_accum "${GRAD_ACCUM}" \
        --output_dir "${OUTPUT_ROOT}/${RUN_NAME}"

    echo
    echo "Finished ${RUN_NAME}"
    echo

done

echo
echo "======================================="
echo "Hyperparameter sweep complete."
echo "======================================="