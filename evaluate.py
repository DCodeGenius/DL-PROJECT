"""
Evaluation script for FEVER hallucination study.
Evaluates baseline (zero-shot) or fine-tuned model on a specific test set.
Confidence = softmax over SUPPORTS/REFUTES logits.

Usage:
    python evaluate.py --mode baseline  --test_set high
    python evaluate.py --mode baseline  --test_set low
    python evaluate.py --mode baseline  --test_set mixed
    python evaluate.py --mode finetuned --test_set high
    python evaluate.py --mode finetuned --test_set low
    python evaluate.py --mode finetuned --test_set mixed
"""

import argparse
import json
import os
from collections import Counter
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import config

LABEL_OPTIONS = ["SUPPORTS", "REFUTES"]


def load_model(mode):
    """Load base or fine-tuned model."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if mode == "finetuned":
        model_path = f"{config.OUTPUT_DIR}/final_model"
        print(f"Loading fine-tuned model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID,
            quantization_config=bnb_config,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        print("Loading baseline model (zero-shot)...")
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID,
            quantization_config=bnb_config,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("Model loaded!")
    return model, tokenizer


def build_prompt(claim):
    """Build claim-only prompt matching training format."""
    return (
        "### Instruction:\n"
        "Based on your knowledge, classify the following claim as "
        "SUPPORTS (true) or REFUTES (false).\n\n"
        f"### Claim:\n{claim}\n\n"
        "### Answer:\n"
    )


def get_label_token_ids(tokenizer):
    """Get first token ID for each label string."""
    return {
        label: tokenizer.encode(label, add_special_tokens=False)[0]
        for label in LABEL_OPTIONS
    }


def predict(model, tokenizer, claim, label_token_ids):
    """Get prediction and confidence via softmax over label logits."""
    prompt = build_prompt(claim)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]

    label_logits = torch.tensor([
        logits[label_token_ids["SUPPORTS"]].item(),
        logits[label_token_ids["REFUTES"]].item(),
    ])
    probs = torch.softmax(label_logits, dim=0).numpy()
    pred_idx = int(np.argmax(probs))

    return {
        "predicted_label": LABEL_OPTIONS[pred_idx],
        "predicted_idx": pred_idx,
        "confidence": float(probs[pred_idx]),
        "prob_supports": float(probs[0]),
        "prob_refutes": float(probs[1]),
    }


def run_evaluation(mode, test_set):
    """Evaluate on a specific test set."""
    meta_path = f"{config.OUTPUT_DIR}/test_{test_set}_metadata.json"
    if not os.path.exists(meta_path):
        print(f"ERROR: {meta_path} not found. Run preprocess_data.py first.")
        return

    with open(meta_path, "r") as f:
        test_metadata = json.load(f)

    print(f"Loaded {len(test_metadata)} examples from test_{test_set}")
    print(f"  Buckets: {dict(Counter(m['bucket'] for m in test_metadata))}")
    print(f"  Labels:  {dict(Counter(m['label'] for m in test_metadata))}")

    model, tokenizer = load_model(mode)
    label_token_ids = get_label_token_ids(tokenizer)

    print(f"\nLabel token IDs:")
    for label, tid in label_token_ids.items():
        print(f"  {label} -> token {tid} ('{tokenizer.decode([tid])}')")

    run_name = f"{mode}_{test_set}"
    print(f"\nEvaluating: {run_name} ({len(test_metadata)} examples)...")

    results = []
    for meta in tqdm(test_metadata, desc=f"Eval {run_name}"):
        pred = predict(model, tokenizer, meta["claim"], label_token_ids)
        results.append({
            "claim": meta["claim"],
            "true_label": meta["label"],
            "bucket": meta["bucket"],
            "wiki_page": meta["wiki_page"],
            "page_frequency": meta["page_frequency"],
            **pred,
        })

    # Print summary
    correct = sum(1 for r in results if r["predicted_idx"] == r["true_label"])
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Results: {run_name}")
    print(f"{'='*60}")
    print(f"Accuracy: {correct}/{total} = {correct/total:.4f} ({correct/total*100:.1f}%)")
    print(f"Avg confidence: {np.mean([r['confidence'] for r in results]):.4f}")

    # Save
    out_path = f"{config.OUTPUT_DIR}/eval_results_{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "finetuned"], required=True)
    parser.add_argument("--test_set", choices=["high", "low", "mixed"], required=True,
                        help="Which test set to evaluate on")
    args = parser.parse_args()
    run_evaluation(args.mode, args.test_set)

