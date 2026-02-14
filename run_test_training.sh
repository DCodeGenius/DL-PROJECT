#!/bin/bash
# Full pipeline: preprocess → train → evaluate (baseline + finetuned) → analyze → stop VM
# Binary classification (SUPPORTS vs REFUTES only)

echo "============================================================"
echo "FEVER Full Pipeline (Test Mode)"
echo "Binary: SUPPORTS vs REFUTES"
echo "============================================================"

# Step 1: Use test config
echo "Setting up test config (1000 train, 200 eval, 1 epoch)..."
cat > config_test.py << 'EOF'
"""
Configuration file for FEVER fine-tuning project - TEST MODE.
"""

# Model Configuration
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./results"
MAX_SEQ_LEN = 512

# Training Configuration - TEST MODE
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
LR_SCHEDULER_TYPE = "cosine"

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Data Configuration - TEST MODE
TRAIN_SAMPLES = 1000
EVAL_SAMPLES = 200
RANDOM_SEED = 42

# Logging & Saving
LOGGING_STEPS = 25
EVAL_STEPS = 50
SAVE_STEPS = 50
REPORT_TO = "none"

# Quantization (4-bit QLoRA)
USE_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_USE_DOUBLE_QUANT = True
EOF

cp config.py config_original.py
cp config_test.py config.py
echo "✅ Config updated for test run"

# Step 2: Preprocess data (generates eval_metadata.json)
echo ""
echo "============================================================"
echo "Step 1/5: Preprocessing data..."
echo "============================================================"
python preprocess_data.py
if [ $? -ne 0 ]; then
    echo "❌ Preprocessing failed!"
    cp config_original.py config.py
    exit 1
fi

# Step 3: Baseline evaluation (zero-shot, before fine-tuning)
echo ""
echo "============================================================"
echo "Step 2/5: Baseline evaluation (zero-shot)..."
echo "============================================================"
python evaluate.py --mode baseline
if [ $? -ne 0 ]; then
    echo "❌ Baseline evaluation failed!"
    cp config_original.py config.py
    exit 1
fi

# Step 4: Fine-tune
echo ""
echo "============================================================"
echo "Step 3/5: Fine-tuning..."
echo "============================================================"
python train.py
if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    cp config_original.py config.py
    exit 1
fi

# Step 5: Fine-tuned evaluation
echo ""
echo "============================================================"
echo "Step 4/5: Fine-tuned evaluation..."
echo "============================================================"
python evaluate.py --mode finetuned
if [ $? -ne 0 ]; then
    echo "❌ Fine-tuned evaluation failed!"
    cp config_original.py config.py
    exit 1
fi

# Step 6: Analysis
echo ""
echo "============================================================"
echo "Step 5/5: Analysis..."
echo "============================================================"
python analyze.py
if [ $? -ne 0 ]; then
    echo "⚠️  Analysis had issues (non-fatal)"
fi

# Restore config
cp config_original.py config.py
echo ""
echo "✅ Original config restored"

# Stop VM
echo ""
echo "============================================================"
echo "Pipeline complete! Stopping VM to save money..."
echo "============================================================"

INSTANCE_NAME=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
ZONE=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d/ -f4)

if [ -n "$INSTANCE_NAME" ] && [ -n "$ZONE" ]; then
    echo "Stopping VM: $INSTANCE_NAME in zone: $ZONE"
    gcloud compute instances stop "$INSTANCE_NAME" --zone="$ZONE" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ VM stop command sent!"
    else
        echo "⚠️  Could not stop VM automatically"
        echo "   Please stop it manually from GCP Console"
    fi
else
    echo "⚠️  Could not detect VM info"
    echo "   Please stop VM manually from GCP Console"
fi

echo ""
echo "✅ All results saved to: ./results/"
echo "   eval_results_baseline.json"
echo "   eval_results_finetuned.json"
echo "   full_analysis.json"
echo "   calibration_*.png"
echo "   frequency_analysis.png"
