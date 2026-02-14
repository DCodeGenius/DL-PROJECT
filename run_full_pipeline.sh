#!/bin/bash
# Full pipeline: preprocess → train → evaluate (baseline + finetuned) → analyze → stop VM
# Binary classification (SUPPORTS vs REFUTES only)
# Uses config.py as-is (20K train, 2000 eval, 3 epochs)

echo "============================================================"
echo "FEVER Full Pipeline (FULL RUN)"
echo "Binary: SUPPORTS vs REFUTES"
echo "20K train, 2000 eval, 3 epochs"
echo "============================================================"

# Step 1: Preprocess data (generates eval_metadata.json)
echo ""
echo "============================================================"
echo "Step 1/5: Preprocessing data..."
echo "============================================================"
python preprocess_data.py
if [ $? -ne 0 ]; then
    echo "❌ Preprocessing failed!"
    exit 1
fi

# Step 2: Baseline evaluation (zero-shot, before fine-tuning)
echo ""
echo "============================================================"
echo "Step 2/5: Baseline evaluation (zero-shot)..."
echo "============================================================"
python evaluate.py --mode baseline
if [ $? -ne 0 ]; then
    echo "❌ Baseline evaluation failed!"
    exit 1
fi

# Step 3: Fine-tune
echo ""
echo "============================================================"
echo "Step 3/5: Fine-tuning (20K examples, 3 epochs)..."
echo "This will take ~2-3 hours on L4 GPU"
echo "============================================================"
python train.py
if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

# Step 4: Fine-tuned evaluation
echo ""
echo "============================================================"
echo "Step 4/5: Fine-tuned evaluation..."
echo "============================================================"
python evaluate.py --mode finetuned
if [ $? -ne 0 ]; then
    echo "❌ Fine-tuned evaluation failed!"
    exit 1
fi

# Step 5: Analysis
echo ""
echo "============================================================"
echo "Step 5/5: Analysis..."
echo "============================================================"
python analyze.py
if [ $? -ne 0 ]; then
    echo "⚠️  Analysis had issues (non-fatal)"
fi

# Stop VM
echo ""
echo "============================================================"
echo "Pipeline complete! Stopping VM to save money..."
echo "============================================================"

INSTANCE_NAME=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
ZONE=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d/ -f4)

if [ -n "$INSTANCE_NAME" ] && [ -n "$ZONE" ]; then
    echo "Stopping VM: $INSTANCE_NAME in zone: $ZONE"
    sudo shutdown -h now 2>/dev/null || gcloud compute instances stop "$INSTANCE_NAME" --zone="$ZONE" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ VM shutting down!"
    else
        echo "⚠️  Could not stop VM automatically"
        echo "   Please stop it manually from GCP Console"
    fi
else
    echo "⚠️  Could not detect VM info. Trying sudo shutdown..."
    sudo shutdown -h now 2>/dev/null
fi

echo ""
echo "✅ All results saved to: ./results/"
echo "   eval_results_baseline.json"
echo "   eval_results_finetuned.json"
echo "   full_analysis.json"
echo "   calibration_*.png"
echo "   frequency_analysis.png"
