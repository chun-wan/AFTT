#!/bin/bash
# AFTT GLM-5 LoRA Fine-Tuning Launch Script
# Launches training inside Docker container with 8x MI325X GPUs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="/mnt/md0/GLM-5-fp8"
OUTPUT_DIR="/home/root123/AFTT/output"
IMAGE="aftt-train:latest"

mkdir -p "$OUTPUT_DIR"

echo "=== AFTT GLM-5 LoRA Training ==="
echo "Model: $MODEL_PATH"
echo "Image: $IMAGE"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if image exists
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Building Docker image..."
    docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile.train" "$SCRIPT_DIR"
fi

NUM_GPUS="${NUM_GPUS:-8}"
TRAIN_CONFIG="${TRAIN_CONFIG:-/workspace/train_config.json}"

echo "Starting training with $NUM_GPUS GPUs..."

docker run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size=256g \
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e ROCM_PATH=/opt/rocm \
    -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
    -v "$MODEL_PATH":/mnt/md0/GLM-5-fp8:ro \
    -v "$PROJECT_DIR/db/training_data":/workspace/data:ro \
    -v "$OUTPUT_DIR":/workspace/output \
    -v "$SCRIPT_DIR":/workspace/config:ro \
    --name aftt-train \
    "$IMAGE" \
    bash -c "
        cp /workspace/config/train_lora.py /workspace/train_lora.py
        cp /workspace/config/ds_config_zero2.json /workspace/ds_config_zero2.json
        cp /workspace/config/train_config.json /workspace/train_config.json

        deepspeed --num_gpus=$NUM_GPUS /workspace/train_lora.py \
            /workspace/train_config.json
    "

echo ""
echo "=== Training complete ==="
echo "Output: $OUTPUT_DIR/glm5-aftt-lora"
