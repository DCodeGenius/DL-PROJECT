"""
Quick verification — GPU, HuggingFace, model access, data loading.
"""

import sys
import torch
from transformers import AutoTokenizer
import config


def check_gpu():
    print("Checking GPU...")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({props.total_mem / 1e9:.1f} GB)")
        return True
    print("  WARNING: No GPU detected.")
    return False


def check_hf():
    print("Checking HuggingFace auth...")
    try:
        from huggingface_hub import whoami
        print(f"  Logged in as: {whoami()['name']}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def check_model():
    print(f"Checking model: {config.MODEL_ID}...")
    try:
        AutoTokenizer.from_pretrained(config.MODEL_ID)
        print("  Model accessible!")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def check_data():
    print("Checking data loading...")
    try:
        from preprocess_data import load_and_preprocess_fever
        train_ds, eval_ds = load_and_preprocess_fever()
        print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Setup Verification")
    print("=" * 60)
    ok = all([check_gpu(), check_hf(), check_model(), check_data()])
    print("\n" + "=" * 60)
    if ok:
        print("All checks passed! Ready to run.")
    else:
        print("Some checks failed.")
        sys.exit(1)
