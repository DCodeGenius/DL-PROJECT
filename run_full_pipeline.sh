#!/bin/bash
# Full experiment pipeline:
#   1. Preprocess (build eval set + high train set)
#   2. Baseline evaluation (zero-shot)
#   3. Train on HIGH-frequency data
#   4. Evaluate high-trained model
#   5. Train on LOW-frequency data
#   6. Evaluate low-trained model
#   7. Run analysis (3×2 comparison matrix)
#   8. Auto-stop VM
#
# Expected time: ~4 hours on L4 GPU

set -e  # Exit on any error

echo "============================================================"
echo "FEVER Hallucination Study — Full Experiment"
echo "============================================================"
echo "Design: (Baseline / High-trained / Low-trained) × (Eval-H / Eval-L)"
echo "Train: 10K per model, Eval: 2K (balanced)"
echo ""

# ──── Step 1: Preprocess for HIGH experiment (generates shared eval set) ────
echo "============================================================"
echo "Step 1/7: Preprocessing (HIGH experiment → generates eval set)..."
echo "============================================================"
# We run preprocessing with EXPERIMENT=high first.
# The eval set is the same regardless of experiment.
python -c "
import config
config.EXPERIMENT = 'high'
from preprocess_data import load_and_preprocess_fever
load_and_preprocess_fever()
"

# ──── Step 2: Baseline evaluation ──────────────────────────────────────
echo ""
echo "============================================================"
echo "Step 2/7: Baseline evaluation (zero-shot)..."
echo "============================================================"
python evaluate.py --mode baseline

# ──── Step 3: Train on HIGH-frequency data ─────────────────────────────
echo ""
echo "============================================================"
echo "Step 3/7: Fine-tuning on HIGH-frequency data (~1.5 hours)..."
echo "============================================================"
python train.py --experiment high

# ──── Step 4: Evaluate high-trained model ──────────────────────────────
echo ""
echo "============================================================"
echo "Step 4/7: Evaluating HIGH-trained model..."
echo "============================================================"
python evaluate.py --mode finetuned --experiment high

# ──── Step 5: Train on LOW-frequency data ──────────────────────────────
echo ""
echo "============================================================"
echo "Step 5/7: Fine-tuning on LOW-frequency data (~1.5 hours)..."
echo "============================================================"
python train.py --experiment low

# ──── Step 6: Evaluate low-trained model ───────────────────────────────
echo ""
echo "============================================================"
echo "Step 6/7: Evaluating LOW-trained model..."
echo "============================================================"
python evaluate.py --mode finetuned --experiment low

# ──── Step 7: Analysis ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Step 7/7: Running analysis (3×2 matrix)..."
echo "============================================================"
python analyze.py

# ──── Done — stop VM ──────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "✅ EXPERIMENT COMPLETE!"
echo "============================================================"
echo ""
echo "Results saved to ./results/:"
echo "  eval_results_baseline.json"
echo "  eval_results_finetuned_high.json"
echo "  eval_results_finetuned_low.json"
echo "  full_analysis.json"
echo "  chart_accuracy.png"
echo "  chart_hce_rate.png"
echo "  chart_ece.png"
echo "  chart_overconf_gap.png"
echo "  calibration_*.png"
echo ""
echo "Stopping VM to save money..."

sudo shutdown -h now 2>/dev/null || {
    INSTANCE_NAME=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name 2>/dev/null)
    ZONE=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null | cut -d/ -f4)
    if [ -n "$INSTANCE_NAME" ] && [ -n "$ZONE" ]; then
        gcloud compute instances stop "$INSTANCE_NAME" --zone="$ZONE" 2>/dev/null
    fi
    echo "⚠️  Could not auto-stop VM. Stop manually from GCP Console."
}
