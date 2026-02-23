#!/usr/bin/env python3
"""AFTT Sanity Training: 10 steps on 1 GPU to validate the full pipeline."""
import torch
import time
import sys
import json
import os
from pathlib import Path

print(f"GPUs: {torch.cuda.device_count()}")
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained("/mnt/md0/GLM-5-fp8", trust_remote_code=True, padding_side="right")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"Tokenizer OK")

print("Loading model...")
sys.stdout.flush()
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "/mnt/md0/GLM-5-fp8",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print(f"Loaded in {time.time()-t0:.0f}s")
sys.stdout.flush()

model.config.use_cache = False

targets = ["q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj"]

# Dequantize FP8 targets to BF16 with proper block-scaling
deq = 0
BLOCK_SIZE = 128
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
print(f"Dequantized {deq} FP8 modules with block-scaling")
sys.stdout.flush()

# Remove quantization config to bypass Trainer's FP8 training block.
# FP8 base weights are frozen; only BF16 LoRA adapters are trained.
for cfg in [model.config, getattr(model, "base_model", model).config if hasattr(model, "base_model") else None]:
    if cfg is not None and hasattr(cfg, "quantization_config"):
        delattr(cfg, "quantization_config")
if hasattr(model, "is_quantized"):
    model.is_quantized = False

# Patch the trainer validation to allow our setup
import transformers.trainer_utils as _tu
_tu.validate_quantization_for_training = lambda *a, **kw: None
print("Bypassed FP8 training validation (LoRA adapters are BF16)")

if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32, lora_alpha=64, lora_dropout=0.0,
    target_modules=targets, bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
print("Loading training data...")
dataset = load_dataset("json", data_files="/workspace/data/train.jsonl", split="train")
dataset = dataset.select(range(min(100, len(dataset))))
print(f"Using {len(dataset)} examples for sanity check")

def tokenize_fn(example):
    messages = example["messages"]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc = tok(text, truncation=True, max_length=1024, padding="max_length", return_tensors=None)
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized = dataset.map(tokenize_fn, remove_columns=dataset.column_names, num_proc=4, desc="Tokenizing")
print(f"Tokenized {len(tokenized)} examples")

training_args = TrainingArguments(
    output_dir="/workspace/output/sanity",
    num_train_epochs=1,
    max_steps=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=1,
    save_steps=999999,
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    processing_class=tok,
)

print("\n=== Starting sanity training (10 steps) ===")
sys.stdout.flush()
t0 = time.time()
result = trainer.train()
elapsed = time.time() - t0

print(f"\n=== Training completed in {elapsed:.1f}s ===")
print(f"Loss: {result.training_loss:.4f}")
print(f"Steps: {result.global_step}")
print(f"Samples/sec: {result.metrics.get('train_samples_per_second', 'N/A')}")

for i in range(torch.cuda.device_count()):
    a = torch.cuda.memory_allocated(i) / 1024**3
    t = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"GPU{i}: {a:.1f}/{t:.0f}GiB ({a/t*100:.0f}%)")

print("\n=== SANITY TRAINING PASSED ===")
