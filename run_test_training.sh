#!/bin/bash
# Script to run test training and stop VM automatically
# This saves money by stopping the VM when training completes

echo "============================================================"
echo "Test Training with Auto-Shutdown"
echo "============================================================"

# Step 1: Update config for test run
echo "Updating config for test run..."
cat > config_test.py << 'EOF'
"""
Configuration file for FEVER fine-tuning project - TEST MODE.
"""

# Model Configuration
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./results"
MAX_SEQ_LEN = 512

# Training Configuration - TEST MODE (small run)
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
REPORT_TO = "none"  # Disable wandb for test

# Quantization (4-bit QLoRA)
USE_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_USE_DOUBLE_QUANT = True
EOF

# Backup original config
cp config.py config_original.py

# Use test config
cp config_test.py config.py

echo "✅ Config updated for test run (1000 samples, 1 epoch)"

# Step 2: Run training
echo ""
echo "Starting training..."
echo "⚠️  VM will stop automatically when training completes"
echo ""

python train.py

# Step 3: Restore original config
cp config_original.py config.py
echo "✅ Original config restored"

# Step 4: Stop VM
echo ""
echo "============================================================"
echo "Training complete! Stopping VM to save money..."
echo "============================================================"

# Try to stop VM using gcloud
INSTANCE_NAME=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
ZONE=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d/ -f4)

if [ -n "$INSTANCE_NAME" ] && [ -n "$ZONE" ]; then
    echo "Stopping VM: $INSTANCE_NAME in zone: $ZONE"
    gcloud compute instances stop "$INSTANCE_NAME" --zone="$ZONE" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "✅ VM stop command sent successfully!"
        echo "VM will shut down in a few moments"
    else
        echo "⚠️  Could not stop VM automatically (gcloud not configured)"
        echo "Please stop it manually from GCP Console"
    fi
else
    echo "⚠️  Could not detect VM info"
    echo "Please stop VM manually from GCP Console"
fi

echo ""
echo "✅ Results saved to: ./results/"
echo "✅ You can start the VM again anytime to continue work"
echo ""
echo "To start VM again:"
echo "  gcloud compute instances start $INSTANCE_NAME --zone=$ZONE"
echo "  Or use GCP Console → VM instances → Start"
