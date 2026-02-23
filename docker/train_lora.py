#!/usr/bin/env python3
"""AFTT GLM-5 LoRA Fine-Tuning Script.

Fine-tunes GLM-5-FP8 (744B MoE, 40B active) using LoRA on attention layers.
Designed for 8x MI325X with DeepSpeed ZeRO-2.
"""

import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from datasets import load_dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="/mnt/md0/GLM-5-fp8",
        metadata={"help": "Path to the pre-trained model"},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Trust remote code in model config"},
    )


@dataclass
class DataArguments:
    train_data: str = field(
        default="/workspace/data/train.jsonl",
        metadata={"help": "Path to training data (ChatML JSONL)"},
    )
    eval_data: Optional[str] = field(
        default=None,
        metadata={"help": "Path to eval data (ChatML JSONL)"},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"},
    )


@dataclass
class LoraArguments:
    lora_rank: int = field(default=32, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout (0.0 for FP8 base models)"})
    target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"},
    )


def find_target_modules(model, target_names):
    """Find all matching linear module names for LoRA injection."""
    target_set = set(target_names)
    found = set()
    for name, module in model.named_modules():
        short_name = name.split(".")[-1]
        if short_name in target_set and isinstance(module, (torch.nn.Linear,)):
            found.add(short_name)
    return list(found) if found else target_names


def format_chatml(example, tokenizer, max_seq_length):
    """Format a ChatML example into input_ids and labels."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    encodings = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_tensors=None,
    )

    encodings["labels"] = encodings["input_ids"].copy()
    return encodings


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, LoraArguments, TrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, lora_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, lora_args, training_args = (
            parser.parse_args_into_dataclasses()
        )

    logger.info("Loading tokenizer from %s", model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model from %s", model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    model.config.use_cache = False

    target_modules = lora_args.target_modules.split(",")
    resolved_targets = find_target_modules(model, target_modules)

    # Dequantize FP8 LoRA target modules to BF16 for training compatibility
    dequantized = 0
    for name, module in model.named_modules():
        short_name = name.split(".")[-1]
        if short_name in resolved_targets and isinstance(module, torch.nn.Linear):
            if module.weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz,
                                        torch.float8_e5m2, torch.float8_e5m2fnuz):
                module.weight.data = module.weight.data.to(torch.bfloat16)
                if hasattr(module, "weight_scale_inv"):
                    scale = module.weight_scale_inv
                    module.weight.data = module.weight.data * scale
                    delattr(module, "weight_scale_inv")
                dequantized += 1
    if dequantized:
        logger.info("Dequantized %d FP8 modules to BF16 for LoRA training", dequantized)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    logger.info("LoRA target modules resolved: %s", resolved_targets)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_args.lora_rank,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=resolved_targets,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    logger.info("Loading dataset from %s", data_args.train_data)
    dataset = load_dataset("json", data_files=data_args.train_data, split="train")

    if data_args.eval_data:
        eval_dataset = load_dataset(
            "json", data_files=data_args.eval_data, split="train"
        )
    else:
        split = dataset.train_test_split(test_size=0.02, seed=42)
        dataset = split["train"]
        eval_dataset = split["test"]

    logger.info("Train: %d examples, Eval: %d examples", len(dataset), len(eval_dataset))

    def tokenize_fn(example):
        return format_chatml(example, tokenizer, data_args.max_seq_length)

    tokenized_train = dataset.map(
        tokenize_fn,
        remove_columns=dataset.column_names,
        num_proc=min(os.cpu_count() or 4, 32),
        desc="Tokenizing train",
    )
    tokenized_eval = eval_dataset.map(
        tokenize_fn,
        remove_columns=eval_dataset.column_names,
        num_proc=min(os.cpu_count() or 4, 32),
        desc="Tokenizing eval",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    train_result = trainer.train()

    logger.info("Saving model to %s", training_args.output_dir)
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(tokenized_train)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if training_args.do_eval:
        logger.info("Running evaluation...")
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_samples"] = len(tokenized_eval)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
