#!/bin/bash
# Run this script on the VM to create all project files
# Copy-paste this entire script into the VM terminal

mkdir -p ~/fever-project
cd ~/fever-project

# Create config.py
cat > config.py << 'EOF'
"""
Configuration file for FEVER fine-tuning project.
Modify these parameters to control training behavior.
"""

# Model Configuration
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Update if you see a different exact name on HF
OUTPUT_DIR = "./results"
MAX_SEQ_LEN = 512

# Training Configuration
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
LR_SCHEDULER_TYPE = "cosine"

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Data Configuration
TRAIN_SAMPLES = 20000  # Start with 20K for baseline, adjust as needed
EVAL_SAMPLES = 2000
RANDOM_SEED = 42

# Logging & Saving
LOGGING_STEPS = 25
EVAL_STEPS = 200
SAVE_STEPS = 200
REPORT_TO = "wandb"  # Set to "none" if you don't want wandb

# Quantization (4-bit QLoRA)
USE_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_USE_DOUBLE_QUANT = True
EOF

# Create preprocess_data.py
cat > preprocess_data.py << 'EOF'
"""
Data preprocessing for FEVER dataset.
Formats claims + evidence into instruction-following prompts.
"""

from datasets import load_dataset
import config

LABEL_MAP = {
    0: "SUPPORTS",
    1: "REFUTES", 
    2: "NOT ENOUGH INFO"
}

def format_prompt(example):
    """
    Format FEVER example into instruction-following prompt.
    
    Decision 1 (Option B): Model sees claim + evidence → predicts label
    """
    claim = example["claim"]
    label = LABEL_MAP[example["label"]]
    
    # Extract evidence text (FEVER provides evidence sentences)
    evidence_text = ""
    if "evidence" in example and example["evidence"]:
        # evidence is a list of lists: [[[doc_id, sent_id, text], ...], ...]
        evidence_sentences = []
        for ev_group in example["evidence"]:
            for ev in ev_group:
                if len(ev) >= 3:
                    evidence_sentences.append(ev[2])  # The text is at index 2
        
        evidence_text = " ".join(evidence_sentences[:5])  # Limit to first 5 sentences
    
    # Format as instruction-following prompt
    text = (
        f"### Instruction:\n"
        f"Given the following claim and evidence, classify whether the evidence "
        f"SUPPORTS, REFUTES, or provides NOT ENOUGH INFO for the claim.\n\n"
        f"### Claim:\n{claim}\n\n"
        f"### Evidence:\n{evidence_text}\n\n"
        f"### Answer:\n{label}"
    )
    
    return {"text": text, "label": example["label"], "claim": claim}

def load_and_preprocess_fever():
    """Load FEVER dataset and apply preprocessing."""
    print("Loading FEVER dataset...")
    dataset = load_dataset("fever", "v1.0", trust_remote_code=True)
    
    print(f"Train size: {len(dataset['train'])}")
    print(f"Labelled dev size: {len(dataset['labelled_dev'])}")
    
    # Process training set
    print("Preprocessing training data...")
    train_ds = dataset["train"].shuffle(seed=config.RANDOM_SEED)
    if config.TRAIN_SAMPLES:
        train_ds = train_ds.select(range(min(config.TRAIN_SAMPLES, len(train_ds))))
    train_ds = train_ds.map(format_prompt, remove_columns=dataset["train"].column_names)
    
    # Process eval set
    print("Preprocessing evaluation data...")
    eval_ds = dataset["labelled_dev"].shuffle(seed=config.RANDOM_SEED)
    if config.EVAL_SAMPLES:
        eval_ds = eval_ds.select(range(min(config.EVAL_SAMPLES, len(eval_ds))))
    eval_ds = eval_ds.map(format_prompt, remove_columns=dataset["labelled_dev"].column_names)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"Train examples: {len(train_ds)}")
    print(f"Eval examples: {len(eval_ds)}")
    print(f"\nSample formatted example:")
    print(train_ds[0]["text"][:500] + "...")
    
    return train_ds, eval_ds

if __name__ == "__main__":
    train_ds, eval_ds = load_and_preprocess_fever()
    print("\n✅ Data ready for training!")
EOF

# Create train.py (truncated for space - you'll need to copy the full version)
echo "Creating train.py..."
# (I'll provide the full content separately)

# Create requirements.txt
cat > requirements.txt << 'EOF'
torch>=2.0.0
transformers>=4.40.0
datasets>=2.14.0
accelerate>=0.25.0
peft>=0.8.0
bitsandbytes>=0.41.0
trl>=0.7.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
wandb>=0.16.0
tqdm>=4.65.0
EOF

echo "✅ Basic files created!"
echo "You still need to create train.py and verify_setup.py"
echo "Use the file browser to upload them, or I can provide the full script."
