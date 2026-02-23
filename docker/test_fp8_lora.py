#!/usr/bin/env python3
"""Test FP8 model loading + LoRA with dequantization fix."""
import torch
import time
import sys

print(f"GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_properties(i).total_memory/1024**3:.0f} GiB")

from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("/mnt/md0/GLM-5-fp8", trust_remote_code=True)
print(f"Tokenizer OK: vocab={tok.vocab_size}")

print("Loading model...")
sys.stdout.flush()
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "/mnt/md0/GLM-5-fp8",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print(f"Loaded in {time.time()-t0:.0f}s, type={type(model).__name__}")
sys.stdout.flush()

linears = set()
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        linears.add(n.split(".")[-1])
print(f"Linear modules: {sorted(linears)}")

targets = [m for m in linears if m in ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj")]
print(f"LoRA targets: {targets}")

# Check dtypes of target modules
fp8_count = 0
for name, module in model.named_modules():
    short = name.split(".")[-1]
    if short in targets and isinstance(module, torch.nn.Linear):
        dtype = module.weight.dtype
        if fp8_count == 0:
            print(f"  Example target dtype: {name} -> {dtype}")
        if dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz):
            fp8_count += 1
print(f"FP8 target modules: {fp8_count}")

# Dequantize FP8 target modules to BF16
if fp8_count > 0:
    print(f"Dequantizing {fp8_count} FP8 modules to BF16 with block-scaling...")
    sys.stdout.flush()
    BLOCK_SIZE = 128
    deq = 0
    for name, module in model.named_modules():
        short = name.split(".")[-1]
        if short in targets and isinstance(module, torch.nn.Linear):
            if module.weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz,
                                        torch.float8_e5m2, torch.float8_e5m2fnuz):
                w = module.weight.data.to(torch.float32)
                if hasattr(module, "weight_scale_inv"):
                    scale = module.weight_scale_inv
                    for r in range(0, w.shape[0], BLOCK_SIZE):
                        for c in range(0, w.shape[1], BLOCK_SIZE):
                            rb = min(r + BLOCK_SIZE, w.shape[0])
                            cb = min(c + BLOCK_SIZE, w.shape[1])
                            sr, sc = r // BLOCK_SIZE, c // BLOCK_SIZE
                            if sr < scale.shape[0] and sc < scale.shape[1]:
                                w[r:rb, c:cb] *= scale[sr, sc]
                module.weight = torch.nn.Parameter(w.to(torch.bfloat16), requires_grad=False)
                deq += 1
    print(f"Dequantized {deq} modules with block-scaling")
    sys.stdout.flush()

from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.0,
    target_modules=targets,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("\n=== Forward pass ===")
sys.stdout.flush()
inputs = tok("What is MFMA?", return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
with torch.no_grad():
    out = model(**inputs)
print(f"Forward OK: logits={out.logits.shape}")

print("\n=== Backward pass ===")
sys.stdout.flush()
inputs["labels"] = inputs["input_ids"].clone()
out = model(**inputs)
print(f"Loss: {out.loss.item():.4f}")
out.loss.backward()
gc = sum(1 for _, p in model.named_parameters() if p.grad is not None)
print(f"Grads: {gc}")

print("\n=== VRAM Usage ===")
for i in range(torch.cuda.device_count()):
    a = torch.cuda.memory_allocated(i) / 1024**3
    t = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"GPU{i}: {a:.1f}/{t:.0f}GiB ({a/t*100:.0f}%)")

print("\n=== ALL PASSED ===")
