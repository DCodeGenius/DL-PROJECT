"""
Evaluation script for FEVER hallucination study.
Runs baseline (zero-shot) or fine-tuned model on the balanced eval set.
Extracts predictions and confidence scores (softmax over SUPPORTS/REFUTES logits).

Usage:
    python evaluate.py --mode baseline
    python evaluate.py --mode finetuned --experiment high
    python evaluate.py --mode finetuned --experiment low
"""

import argparse
import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import config

LABEL_OPTIONS = ["SUPPORTS", "REFUTES"]
LABEL_TO_INT = {"SUPPORTS": 0, "REFUTES": 1}


def load_model(mode, experiment=None):
    """Load base or fine-tuned model."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if mode == "finetuned":
        model_path = f"{config.OUTPUT_DIR}/{experiment}/final_model"
        print(f"Loading fine-tuned model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        print("Loading baseline model (zero-shot)...")
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("✅ Model loaded!")
    return model, tokenizer


def build_prompt(claim):
    """Build claim-only prompt matching training format (binary)."""
    return (
        f"### Instruction:\n"
        f"Based on your knowledge, classify the following claim as "
        f"SUPPORTS (true) or REFUTES (false).\n\n"
        f"### Claim:\n{claim}\n\n"
        f"### Answer:\n"
    )


def get_label_token_ids(tokenizer):
    """Get first token ID for each label."""
    label_tokens = {}
    for label in LABEL_OPTIONS:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        label_tokens[label] = token_ids[0]
    return label_tokens


def predict(model, tokenizer, claim, label_token_ids):
    """Get prediction and confidence via softmax over label logits."""
    prompt = build_prompt(claim)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        next_logits = outputs.logits[0, -1, :]

    label_logits = torch.tensor([
        next_logits[label_token_ids["SUPPORTS"]].item(),
        next_logits[label_token_ids["REFUTES"]].item(),
    ])

    probs = torch.softmax(label_logits, dim=0).numpy()
    predicted_idx = int(np.argmax(probs))

    return {
        "predicted_label": LABEL_OPTIONS[predicted_idx],
        "predicted_idx": predicted_idx,
        "confidence": float(probs[predicted_idx]),
        "prob_supports": float(probs[0]),
        "prob_refutes": float(probs[1]),
    }


def run_evaluation(mode, experiment=None):
    """Run evaluation on the balanced eval set."""
    # Determine eval metadata path (always use the same eval set)
    meta_path = f"{config.OUTPUT_DIR}/eval_metadata.json"
    if not os.path.exists(meta_path):
        print(f"❌ {meta_path} not found. Run preprocessing first.")
        return

    with open(meta_path, "r") as f:
        eval_metadata = json.load(f)

    print(f"Loaded {len(eval_metadata)} eval examples")

    # Count by bucket
    from collections import Counter
    bucket_counts = Counter(m["bucket"] for m in eval_metadata)
    label_counts = Counter(m["label"] for m in eval_metadata)
    print(f"  Buckets: {dict(bucket_counts)}")
    print(f"  Labels: {dict(label_counts)}")

    # Load model
    model, tokenizer = load_model(mode, experiment)
    label_token_ids = get_label_token_ids(tokenizer)

    print(f"\nLabel token IDs:")
    for label, tid in label_token_ids.items():
        print(f"  {label} → token {tid} ('{tokenizer.decode([tid])}')")

    # Run predictions
    run_name = f"{mode}_{experiment}" if experiment else mode
    print(f"\n🚀 Running evaluation: {run_name} on {len(eval_metadata)} examples...")

    results = []
    for meta in tqdm(eval_metadata, desc="Evaluating"):
        pred = predict(model, tokenizer, meta["claim"], label_token_ids)
        results.append({
            "claim": meta["claim"],
            "true_label": meta["label"],
            "bucket": meta["bucket"],
            "wiki_page": meta["wiki_page"],
            "page_frequency": meta["page_frequency"],
            **pred,
        })

    # Summary
    correct = sum(1 for r in results if r["predicted_idx"] == r["true_label"])
    total = len(results)
    accuracy = correct / total

    print(f"\n{'='*60}")
    print(f"Results: {run_name}")
    print(f"{'='*60}")
    print(f"Overall accuracy: {correct}/{total} = {accuracy:.4f} ({accuracy*100:.1f}%)")

    for bucket in ["H", "L"]:
        br = [r for r in results if r["bucket"] == bucket]
        if br:
            bc = sum(1 for r in br if r["predicted_idx"] == r["true_label"])
            print(f"  {bucket}: {bc}/{len(br)} = {bc/len(br):.4f}")

    avg_conf = np.mean([r["confidence"] for r in results])
    print(f"Avg confidence: {avg_conf:.4f}")

    # Save
    output_path = f"{config.OUTPUT_DIR}/eval_results_{run_name}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "finetuned"], required=True)
    parser.add_argument("--experiment", choices=["high", "low"], default=None,
                        help="Required for finetuned mode: which model to evaluate")
    args = parser.parse_args()

    if args.mode == "finetuned" and args.experiment is None:
        parser.error("--experiment is required when mode is 'finetuned'")

    run_evaluation(args.mode, args.experiment)
