#!/bin/bash
# FEVER Hallucination Study — Full Pipeline
#
# Steps:
#   1. Preprocess (build 1 training set + 3 test sets)
#   2. Baseline evaluation on all 3 test sets
#   3. Fine-tune single model on balanced H/M/L training data
#   4. Fine-tuned evaluation on all 3 test sets
#   5. Run analysis (2x3 matrix + charts)
#   6. Auto-stop VM
#
# Expected: ~3.5 hours on L4 GPU

set -e

echo "============================================================"
echo "FEVER Hallucination Study — Full Pipeline"
echo "============================================================"
echo "  Train: 20K (balanced H/M/L, 50/50 labels)"
echo "  Test:  3 sets x 1000 (High, Low, Mixed)"
echo "  Model: 1 fine-tuned + baseline"
echo ""

# Step 1: Preprocess
echo "============================================================"
echo "Step 1/5: Preprocessing..."
echo "============================================================"
python preprocess_data.py

# Step 2: Baseline eval (3 test sets)
echo ""
echo "============================================================"
echo "Step 2/5: Baseline evaluation (3 test sets)..."
echo "============================================================"
for ts in high low mixed; do
    echo "--- Baseline on test_${ts} ---"
    python evaluate.py --mode baseline --test_set $ts
done

# Step 3: Fine-tune
echo ""
echo "============================================================"
echo "Step 3/5: Fine-tuning (20K examples, 3 epochs, ~3 hours)..."
echo "============================================================"
python train.py

# Step 4: Fine-tuned eval (3 test sets)
echo ""
echo "============================================================"
echo "Step 4/5: Fine-tuned evaluation (3 test sets)..."
echo "============================================================"
for ts in high low mixed; do
    echo "--- Fine-tuned on test_${ts} ---"
    python evaluate.py --mode finetuned --test_set $ts
done

# Step 5: Analysis
echo ""
echo "============================================================"
echo "Step 5/5: Analysis (2x3 matrix)..."
echo "============================================================"
python analyze.py

# Done
echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE!"
echo "============================================================"
echo ""
echo "Results in ./results/:"
echo "  eval_results_baseline_high.json"
echo "  eval_results_baseline_low.json"
echo "  eval_results_baseline_mixed.json"
echo "  eval_results_finetuned_high.json"
echo "  eval_results_finetuned_low.json"
echo "  eval_results_finetuned_mixed.json"
echo "  full_analysis.json"
echo "  chart_*.png"
echo "  calibration_*.png"
echo ""
echo "Stopping VM..."
sudo shutdown -h now 2>/dev/null || {
    INSTANCE_NAME=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name 2>/dev/null)
    ZONE=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null | cut -d/ -f4)
    if [ -n "$INSTANCE_NAME" ] && [ -n "$ZONE" ]; then
        gcloud compute instances stop "$INSTANCE_NAME" --zone="$ZONE" 2>/dev/null
    fi
    echo "WARNING: Could not auto-stop VM. Stop manually from GCP Console."
}
