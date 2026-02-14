"""
Quick verification script — tests GPU, HuggingFace login, model access, and data loading.
"""

import sys
import torch
from transformers import AutoTokenizer
import config


def check_gpu():
    print("Checking GPU...")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB)")
        return True
    print("⚠️  No GPU detected.")
    return False


def check_hf_login():
    print("\nChecking HuggingFace auth...")
    try:
        from huggingface_hub import whoami
        print(f"✅ Logged in as: {whoami()['name']}")
        return True
    except Exception as e:
        print(f"❌ Not logged in: {e}")
        return False


def check_model():
    print(f"\nChecking model access: {config.MODEL_ID}...")
    try:
        AutoTokenizer.from_pretrained(config.MODEL_ID)
        print("✅ Model accessible!")
        return True
    except Exception as e:
        print(f"❌ Cannot access model: {e}")
        return False


def check_data():
    print("\nChecking FEVER data loading (binary, frequency-bucketed)...")
    try:
        config.EXPERIMENT = "high"
        from preprocess_data import load_and_preprocess_fever
        train_ds, eval_ds = load_and_preprocess_fever()
        print(f"✅ Train: {len(train_ds)}, Eval: {len(eval_ds)}")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Setup Verification")
    print("=" * 60)
    checks = [check_gpu(), check_hf_login(), check_model(), check_data()]
    print("\n" + "=" * 60)
    if all(checks):
        print("✅ All checks passed! Ready to run experiment.")
    else:
        print("❌ Some checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
