"""
Run all 6 evaluations efficiently: load each model once, evaluate on all 3 test sets.
Then run the full analysis.

Usage:
    python run_all_evals.py
    python run_all_evals.py --skip_baseline      # only run fine-tuned
    python run_all_evals.py --skip_finetuned     # only run baseline
"""

import argparse
import gc
import json
import os
import time
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

import config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

LABEL_OPTIONS = ["SUPPORTS", "REFUTES"]
TEST_SETS = ["high", "low", "mixed"]


def patch_adapter_config(model_path):
    """Remove unknown keys from adapter_config.json to handle PEFT version mismatches."""
    cfg_path = os.path.join(model_path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    from peft import LoraConfig
    import inspect
    valid_keys = set(inspect.signature(LoraConfig.__init__).parameters.keys())
    removed = [k for k in list(cfg.keys()) if k not in valid_keys and k not in ("peft_type", "task_type", "base_model_name_or_path", "revision")]
    if removed:
        print(f"  Patching adapter_config.json: removing unsupported keys {removed}")
        for k in removed:
            del cfg[k]
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)


def load_model(mode):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if mode == "finetuned":
        model_path = f"{config.OUTPUT_DIR}/final_model"
        print(f"Loading fine-tuned model from {model_path}...")
        patch_adapter_config(model_path)
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
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
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Model loaded! GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GiB")
    return model, tokenizer


def build_prompt(claim):
    return (
        "### Instruction:\n"
        "Based on your knowledge, classify the following claim as "
        "SUPPORTS (true) or REFUTES (false).\n\n"
        f"### Claim:\n{claim}\n\n"
        "### Answer:\n"
    )


def get_label_token_ids(tokenizer):
    return {
        label: tokenizer.encode(label, add_special_tokens=False)[0]
        for label in LABEL_OPTIONS
    }


def predict(model, tokenizer, claim, label_token_ids):
    prompt = build_prompt(claim)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs)
        sup_logit = outputs.logits[0, -1, label_token_ids["SUPPORTS"]].item()
        ref_logit = outputs.logits[0, -1, label_token_ids["REFUTES"]].item()
        del outputs
    del inputs

    label_logits = torch.tensor([sup_logit, ref_logit])
    probs = torch.softmax(label_logits, dim=0).numpy()
    pred_idx = int(np.argmax(probs))

    return {
        "predicted_label": LABEL_OPTIONS[pred_idx],
        "predicted_idx": pred_idx,
        "confidence": float(probs[pred_idx]),
        "prob_supports": float(probs[0]),
        "prob_refutes": float(probs[1]),
    }


def evaluate_test_set(model, tokenizer, label_token_ids, mode, test_set):
    meta_path = f"{config.OUTPUT_DIR}/test_{test_set}_metadata.json"
    if not os.path.exists(meta_path):
        print(f"  SKIP: {meta_path} not found")
        return None

    with open(meta_path, "r") as f:
        test_metadata = json.load(f)

    run_name = f"{mode}_{test_set}"
    print(f"\n  Evaluating {run_name}: {len(test_metadata)} examples")
    print(f"    Buckets: {dict(Counter(m['bucket'] for m in test_metadata))}")
    print(f"    Labels:  {dict(Counter(m['label'] for m in test_metadata))}")

    results = []
    for i, meta in enumerate(tqdm(test_metadata, desc=f"  {run_name}")):
        pred = predict(model, tokenizer, meta["claim"], label_token_ids)
        results.append({
            "claim": meta["claim"],
            "true_label": meta["label"],
            "bucket": meta["bucket"],
            "wiki_page": meta["wiki_page"],
            "page_frequency": meta["page_frequency"],
            **pred,
        })
        if i % 50 == 0:
            torch.cuda.empty_cache()

    correct = sum(1 for r in results if r["predicted_idx"] == r["true_label"])
    total = len(results)
    print(f"  -> {run_name}: {correct}/{total} = {correct/total:.4f} ({correct/total*100:.1f}%)")

    out_path = f"{config.OUTPUT_DIR}/eval_results_{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  -> Saved {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--skip_finetuned", action="store_true")
    args = parser.parse_args()

    modes = []
    if not args.skip_baseline:
        modes.append("baseline")
    if not args.skip_finetuned:
        modes.append("finetuned")

    if not modes:
        print("ERROR: nothing to run (both modes skipped)")
        return

    start = time.time()

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  MODE: {mode.upper()} — loading model once, evaluating 3 test sets")
        print(f"{'='*60}")

        model, tokenizer = load_model(mode)
        label_token_ids = get_label_token_ids(tokenizer)

        for ts in TEST_SETS:
            evaluate_test_set(model, tokenizer, label_token_ids, mode, ts)

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n  Unloaded {mode} model, freed GPU memory.")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"All evaluations complete in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    print(f"\nRunning analysis...")
    import analyze
    analyze.main()

    print(f"\nDone! Total time: {(time.time() - start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
