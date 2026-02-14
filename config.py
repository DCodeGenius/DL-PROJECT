"""
Configuration for FEVER hallucination study.
Controlled frequency experiment: High vs Low training exposure.
"""

# Model
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./results"
MAX_SEQ_LEN = 512

# Experiment: "high" or "low" (which frequency bucket to train on)
# Changed by the pipeline script for each run
EXPERIMENT = "high"

# Training
TRAIN_SAMPLES = 10000   # per experiment (5K SUPPORTS + 5K REFUTES)
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # effective batch = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
LR_SCHEDULER_TYPE = "cosine"

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Eval
EVAL_SAMPLES = 2000  # 500 per cell (H+SUP, H+REF, L+SUP, L+REF)
RANDOM_SEED = 42

# Logging & Saving
LOGGING_STEPS = 25
EVAL_STEPS = 200
SAVE_STEPS = 200
REPORT_TO = "none"

# Quantization (4-bit QLoRA)
USE_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_USE_DOUBLE_QUANT = True
