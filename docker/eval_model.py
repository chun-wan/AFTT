#!/usr/bin/env python3
"""AFTT Model Evaluation Script.

Evaluates the fine-tuned GLM-5 LoRA model on ASM optimization tasks
and compares with the base model.
"""

import json
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


EVAL_PROMPTS = [
    {
        "category": "isa_knowledge",
        "prompt": "What does the AMDGPU instruction v_mfma_f32_32x32x8_bf16 do?",
        "keywords": ["mfma", "matrix", "bf16", "32x32", "accumulate"],
    },
    {
        "category": "optimization_advice",
        "prompt": "My kernel uses __shfl_xor for reductions on MI300X. How can I make it faster?",
        "keywords": ["dpp", "ds_bpermute", "wave_reduce", "row_shr", "cross-lane"],
    },
    {
        "category": "asm_analysis",
        "prompt": "I see many s_waitcnt vmcnt(0) in my kernel ASM. Is this a problem?",
        "keywords": ["stall", "pipeline", "latency", "vmcnt", "outstanding"],
    },
    {
        "category": "arch_comparison",
        "prompt": "What are the key differences between gfx90a (MI200) and gfx942 (MI300X) for kernel optimization?",
        "keywords": ["hbm", "vgpr", "fp8", "bandwidth", "memory"],
    },
    {
        "category": "pattern_detection",
        "prompt": "My GEMM kernel on gfx942 has 200 VGPRs and only achieves 2 waves occupancy. What should I do?",
        "keywords": ["occupancy", "register", "spill", "reduce", "agpr"],
    },
    {
        "category": "porting",
        "prompt": "How do I convert a CUDA __syncwarp() to HIP for AMD GPUs?",
        "keywords": ["lockstep", "wavefront", "no-op", "remove", "unnecessary"],
    },
    {
        "category": "profiling",
        "prompt": "What rocprof-compute metrics should I check to diagnose a memory-bound kernel?",
        "keywords": ["bandwidth", "fetch", "l2", "coalescing", "hit_rate"],
    },
    {
        "category": "dpp_usage",
        "prompt": "How does DPP row_shr work and when should I use it instead of LDS for reductions?",
        "keywords": ["lane", "shift", "register", "reduction", "wave"],
    },
]


def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    messages = [
        {"role": "system", "content": "You are an expert AMDGPU kernel optimization advisor."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def score_response(response, keywords):
    """Simple keyword-based scoring."""
    response_lower = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in response_lower)
    return hits / len(keywords) if keywords else 0


def main():
    base_model_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/md0/GLM-5-fp8"
    lora_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Base model: {base_model_path}")
    print(f"LoRA adapter: {lora_path or 'None (base model only)'}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left",
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if lora_path:
        print(f"Loading LoRA adapter from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    model.eval()

    results = []
    total_score = 0

    for i, eval_item in enumerate(EVAL_PROMPTS):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(EVAL_PROMPTS)}] Category: {eval_item['category']}")
        print(f"Prompt: {eval_item['prompt']}")

        response = generate_response(model, tokenizer, eval_item["prompt"])
        score = score_response(response, eval_item["keywords"])
        total_score += score

        print(f"\nResponse ({len(response)} chars):")
        print(response[:500] + ("..." if len(response) > 500 else ""))
        print(f"\nKeyword score: {score:.2%} ({int(score * len(eval_item['keywords']))}/{len(eval_item['keywords'])} keywords)")

        results.append({
            "category": eval_item["category"],
            "prompt": eval_item["prompt"],
            "response": response,
            "score": score,
            "keywords_hit": [kw for kw in eval_item["keywords"] if kw.lower() in response.lower()],
            "keywords_missed": [kw for kw in eval_item["keywords"] if kw.lower() not in response.lower()],
        })

    avg_score = total_score / len(EVAL_PROMPTS)
    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS")
    print(f"Average keyword score: {avg_score:.2%}")
    print(f"Per-category scores:")
    for r in results:
        print(f"  {r['category']}: {r['score']:.2%}")

    output_path = Path("/workspace/output/eval_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": base_model_path,
            "lora": lora_path,
            "average_score": avg_score,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
