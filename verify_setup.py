"""
Quick verification script to test that everything is set up correctly.
Run this before starting training to catch issues early.
"""

import sys
import torch
from transformers import AutoTokenizer
import config

def check_gpu():
    """Check GPU availability."""
    print("Checking GPU...")
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠️  No GPU detected. Training will be very slow on CPU.")
        return False

def check_huggingface_login():
    """Check if logged into HuggingFace."""
    print("\nChecking HuggingFace authentication...")
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"✅ Logged in as: {user['name']}")
        return True
    except Exception as e:
        print(f"❌ Not logged in. Run: huggingface-cli login")
        print(f"   Error: {e}")
        return False

def check_model_access():
    """Try to load tokenizer (lightweight check)."""
    print(f"\nChecking model access: {config.MODEL_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
        print("✅ Model accessible!")
        return True
    except Exception as e:
        print(f"❌ Cannot access model. Make sure:")
        print(f"   1. You've accepted the license on HuggingFace")
        print(f"   2. You're logged in: huggingface-cli login")
        print(f"   Error: {e}")
        return False

def check_data_loading():
    """Test data loading."""
    print("\nChecking FEVER data loading...")
    try:
        from preprocess_data import load_and_preprocess_fever
        train_ds, eval_ds = load_and_preprocess_fever()
        print(f"✅ Data loaded successfully!")
        print(f"   Train: {len(train_ds)} examples")
        print(f"   Eval: {len(eval_ds)} examples")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def main():
    """Run all checks."""
    print("=" * 60)
    print("Setup Verification")
    print("=" * 60)
    
    checks = [
        check_gpu(),
        check_huggingface_login(),
        check_model_access(),
        check_data_loading(),
    ]
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✅ All checks passed! You're ready to train.")
        print("\nNext step: Run 'python train.py'")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
