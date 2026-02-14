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

REPORT_TO = "none"  # Set to "wandb" if you want experiment tracking

# Quantization (4-bit QLoRA)
USE_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_USE_DOUBLE_QUANT = True
