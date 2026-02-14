"""
Evaluation script for FEVER fine-tuned model.
Runs model on eval set, extracts predictions and confidence scores.
Supports both zero-shot (baseline) and fine-tuned evaluation.

Usage:
    python evaluate.py --mode baseline      # Zero-shot base model
    python evaluate.py --mode finetuned     # Fine-tuned model
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

LABEL_OPTIONS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

LABEL_TO_INT = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2,
}


def load_model(mode):
    """Load model based on evaluation mode."""
    print(f"Loading model in {mode} mode...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if mode == "finetuned":
        tokenizer = AutoTokenizer.from_pretrained(f"{config.OUTPUT_DIR}/final_model")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, f"{config.OUTPUT_DIR}/final_model")
    else:
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
    """Build claim-only prompt matching the training format."""
    return (
        f"### Instruction:\n"
        f"Based on your knowledge, classify the following claim as "
        f"SUPPORTS (true), REFUTES (false), or NOT ENOUGH INFO.\n\n"
        f"### Claim:\n{claim}\n\n"
        f"### Answer:\n"
    )


def get_label_token_ids(tokenizer):
    """Get the first token ID for each label option."""
    label_tokens = {}
    for label in LABEL_OPTIONS:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        label_tokens[label] = token_ids[0]
    return label_tokens


def predict_with_confidence(model, tokenizer, claim, label_token_ids):
    """
    Get model prediction and confidence for a single claim.
    Confidence = softmax probability over the three label tokens at the next-token position.
    """
    prompt = build_prompt(claim)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]

    # Extract logits for the three label tokens only
    label_logits = torch.tensor([
        next_token_logits[label_token_ids["SUPPORTS"]].item(),
        next_token_logits[label_token_ids["REFUTES"]].item(),
        next_token_logits[label_token_ids["NOT ENOUGH INFO"]].item(),
    ])

    # Softmax over just these three options → confidence scores
    probs = torch.softmax(label_logits, dim=0).numpy()

    predicted_idx = np.argmax(probs)
    predicted_label = LABEL_OPTIONS[predicted_idx]
    confidence = float(probs[predicted_idx])

    return {
        "predicted_label": predicted_label,
        "predicted_idx": int(predicted_idx),
        "confidence": confidence,
        "prob_supports": float(probs[0]),
        "prob_refutes": float(probs[1]),
        "prob_nei": float(probs[2]),
    }


def run_evaluation(mode):
    """Run evaluation on the full eval set."""
    # Load eval metadata
    meta_path = f"{config.OUTPUT_DIR}/eval_metadata.json"
    if not os.path.exists(meta_path):
        print(f"❌ {meta_path} not found. Run train.py first to generate eval metadata.")
        return

    with open(meta_path, "r") as f:
        eval_metadata = json.load(f)

    print(f"Loaded {len(eval_metadata)} eval examples")

    # Load model
    model, tokenizer = load_model(mode)
    label_token_ids = get_label_token_ids(tokenizer)

    print(f"\nLabel token IDs:")
    for label, tid in label_token_ids.items():
        print(f"  {label} → token {tid} ('{tokenizer.decode([tid])}')")

    # Run predictions
    print(f"\n🚀 Running {mode} evaluation on {len(eval_metadata)} examples...")
    results = []

    for i, meta in enumerate(tqdm(eval_metadata, desc="Evaluating")):
        prediction = predict_with_confidence(
            model, tokenizer, meta["claim"], label_token_ids
        )

        results.append({
            "claim": meta["claim"],
            "true_label": meta["label"],
            "wiki_page": meta["wiki_page"],
            "page_frequency": meta["page_frequency"],
            **prediction,
        })

    # Compute summary stats
    correct = sum(1 for r in results if r["predicted_idx"] == r["true_label"])
    total = len(results)
    accuracy = correct / total

    print(f"\n{'='*60}")
    print(f"Results ({mode})")
    print(f"{'='*60}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.4f} ({accuracy*100:.1f}%)")

    # Accuracy per label
    for label_name, label_int in LABEL_TO_INT.items():
        label_examples = [r for r in results if r["true_label"] == label_int]
        if label_examples:
            label_correct = sum(1 for r in label_examples if r["predicted_idx"] == label_int)
            label_acc = label_correct / len(label_examples)
            print(f"  {label_name}: {label_correct}/{len(label_examples)} = {label_acc:.4f}")

    # Average confidence
    avg_conf = np.mean([r["confidence"] for r in results])
    avg_conf_correct = np.mean([r["confidence"] for r in results if r["predicted_idx"] == r["true_label"]] or [0])
    avg_conf_wrong = np.mean([r["confidence"] for r in results if r["predicted_idx"] != r["true_label"]] or [0])

    print(f"\nAvg confidence (all):     {avg_conf:.4f}")
    print(f"Avg confidence (correct): {avg_conf_correct:.4f}")
    print(f"Avg confidence (wrong):   {avg_conf_wrong:.4f}")

    # Save results
    output_path = f"{config.OUTPUT_DIR}/eval_results_{mode}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")

    # Save summary
    summary = {
        "mode": mode,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_confidence": avg_conf,
        "avg_confidence_correct": avg_conf_correct,
        "avg_confidence_wrong": avg_conf_wrong,
    }
    summary_path = f"{config.OUTPUT_DIR}/eval_summary_{mode}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on FEVER eval set")
    parser.add_argument(
        "--mode",
        choices=["baseline", "finetuned"],
        required=True,
        help="baseline = zero-shot base model, finetuned = fine-tuned model",
    )
    args = parser.parse_args()
    run_evaluation(args.mode)
