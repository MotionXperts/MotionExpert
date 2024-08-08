#!/bin/bash

# Directory settings
CONFIG_FILE="/home/andrewchen/MotionExpert_v2/MotionExpert/results/finetune_test/config.yaml"
CHECKPOINTS_DIR="/home/andrewchen/MotionExpert_v2/MotionExpert/results/finetune_test/checkpoints"
OUTPUT_DIR="/home/andrewchen/MotionExpert_v2/MotionExpert/output/finetune_test"

# # Ensure the output directory exists
# mkdir -p "${OUTPUT_DIR}"

# Epochs to evaluate
EPOCHS=(35 30 25 20 15 10 05 03)

# Loop through each epoch
for EPOCH in "${EPOCHS[@]}"; do
    # Format the epoch number with leading zeros
    FORMATTED_EPOCH=$(printf "%05d" $EPOCH)

    # Construct the checkpoint filename
    CHECKPOINT_FILE="${CHECKPOINTS_DIR}/checkpoint_epoch_${FORMATTED_EPOCH}.pth"

    # Build and execute the command
    echo "Evaluating epoch ${EPOCH}"
    CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29051 evaluation.py --cfg_file ${CONFIG_FILE} --ckpt ${CHECKPOINT_FILE} > "${OUTPUT_DIR}"
done

echo "Evaluation complete."
