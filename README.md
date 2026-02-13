# FEVER Fine-Tuning Project

Fine-tuning Llama-3.1-8B on the FEVER dataset to study hallucination and confidence calibration in LLMs.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. HuggingFace Authentication

You need to log in to HuggingFace to access Llama-3.1:

```bash
huggingface-cli login
```

Enter your HuggingFace token when prompted (get it from https://huggingface.co/settings/tokens)

### 3. Verify Setup

Test that everything works:

```bash
python verify_setup.py
```

This will:
- Check if you can load the model
- Test data loading
- Verify GPU availability

### 4. Run Training

**Sanity check (small run):**
```bash
# First, edit config.py to set:
# TRAIN_SAMPLES = 1000
# NUM_EPOCHS = 1
python train.py
```

**Full baseline training:**
```bash
# Edit config.py:
# TRAIN_SAMPLES = 20000
# NUM_EPOCHS = 3
python train.py
```

## Project Structure

```
DL-PROJECT/
├── config.py              # All configuration parameters
├── preprocess_data.py      # FEVER data loading & formatting
├── train.py               # Main training script
├── verify_setup.py        # Setup verification script
├── requirements.txt       # Python dependencies
└── results/               # Training outputs (created automatically)
```

## Configuration

All training parameters are in `config.py`. Key settings:

- **MODEL_ID**: HuggingFace model identifier
- **TRAIN_SAMPLES**: Number of training examples (start small for testing)
- **BATCH_SIZE**: Per-device batch size
- **NUM_EPOCHS**: Training epochs
- **OUTPUT_DIR**: Where to save checkpoints

## Task Format

The model is trained to classify claims given evidence:
- **Input**: Claim + Evidence text
- **Output**: SUPPORTS / REFUTES / NOT ENOUGH INFO

## GPU Requirements

- **Minimum**: 16GB VRAM (with 4-bit quantization)
- **Recommended**: 24GB+ (L4, A100, etc.)
- **Cloud**: GCP L4 spot instances (~$0.80/hr)

## Next Steps (Week 2-3)

1. Create high-exposure vs. low-exposure data splits
2. Train separate models on each split
3. Evaluate confidence calibration (ECE, reliability diagrams)
4. Analyze high-confidence errors

## Troubleshooting

**"Model not found" error:**
- Make sure you've accepted the license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- Verify you're logged in: `huggingface-cli whoami`

**CUDA out of memory:**
- Reduce `BATCH_SIZE` in config.py
- Reduce `MAX_SEQ_LEN` in config.py
- Make sure `USE_4BIT = True` in config.py

**Import errors:**
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt` again
