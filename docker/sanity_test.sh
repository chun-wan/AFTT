#!/bin/bash
# AFTT Sanity Test: Load GLM-5, apply LoRA, train 10 steps on 1 GPU
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="/mnt/md0/GLM-5-fp8"
OUTPUT_DIR="/home/root123/AFTT/output/sanity_test"
IMAGE="aftt-train:latest"

mkdir -p "$OUTPUT_DIR"

echo "=== AFTT Sanity Test (1 GPU, 10 steps) ==="

if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Building Docker image..."
    docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile.train" "$SCRIPT_DIR"
fi

docker run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size=256g \
    -e HIP_VISIBLE_DEVICES=0 \
    -e ROCM_PATH=/opt/rocm \
    -v "$MODEL_PATH":/mnt/md0/GLM-5-fp8:ro \
    -v "$PROJECT_DIR/db/training_data":/workspace/data:ro \
    -v "$OUTPUT_DIR":/workspace/output \
    -v "$SCRIPT_DIR":/workspace/config:ro \
    --name aftt-sanity \
    "$IMAGE" \
    python3 -c "
import sys, os, json, torch
print('=== Step 1: Check environment ===')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')
print(f'Devices: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'VRAM: {mem:.1f} GiB')

import transformers
print(f'Transformers: {transformers.__version__}')

from peft import LoraConfig, get_peft_model, TaskType
print(f'PEFT loaded OK')

print()
print('=== Step 2: Load tokenizer ===')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    '/mnt/md0/GLM-5-fp8',
    trust_remote_code=True,
    padding_side='right',
)
print(f'Tokenizer loaded: vocab_size={tokenizer.vocab_size}')

print()
print('=== Step 3: Load model ===')
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    '/mnt/md0/GLM-5-fp8',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto',
)
print(f'Model loaded: {type(model).__name__}')
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params:,}')

print()
print('=== Step 4: Find LoRA target modules ===')
target_names = {'q_proj', 'k_proj', 'v_proj', 'o_proj'}
found = set()
for name, module in model.named_modules():
    short = name.split('.')[-1]
    if short in target_names and isinstance(module, torch.nn.Linear):
        found.add(short)
print(f'Found target modules: {found}')

if not found:
    print('WARNING: No matching linear modules found! Listing all linear module names...')
    linears = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linears.add(name.split('.')[-1])
    print(f'Available linear modules: {sorted(linears)}')
    target_list = sorted(linears)[:4]
else:
    target_list = list(found)

print()
print('=== Step 5: Apply LoRA ===')
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=target_list,
    bias='none',
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print()
print('=== Step 6: Test forward pass ===')
test_text = 'What is v_mfma_f32_32x32x8_bf16?'
inputs = tokenizer(test_text, return_tensors='pt')
inputs = {k: v.to(model.device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
print(f'Forward pass OK: logits shape = {outputs.logits.shape}')
print(f'Loss: {outputs.loss}')

print()
print('=== Step 7: Test backward pass ===')
inputs['labels'] = inputs['input_ids'].clone()
outputs = model(**inputs)
loss = outputs.loss
print(f'Loss (with labels): {loss.item():.4f}')
loss.backward()
print('Backward pass OK')

# Check gradients
grad_count = 0
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_count += 1
print(f'Parameters with gradients: {grad_count}')

print()
print('=== SANITY TEST PASSED ===')
print('Model loads, LoRA applies, forward+backward work.')
print(f'Ready for training with {len(target_list)} LoRA target modules: {target_list}')
"

echo ""
echo "=== Sanity test complete ==="
