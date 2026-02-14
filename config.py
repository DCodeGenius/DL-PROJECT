"""
Configuration for FEVER hallucination study.
Single model, frequency-stratified training and evaluation.
"""

# Model
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./results"
MAX_SEQ_LEN = 512

# Frequency bucketing: percentile-based (adapts to data)
# Bottom 33% of examples = Low, middle 33% = Medium, top 33% = High
FREQ_LOW_PERCENTILE = 33
FREQ_HIGH_PERCENTILE = 67

# Training: 20K total, ~6666 per bucket, 50/50 labels within each
TRAIN_SAMPLES = 20000
TRAIN_PER_BUCKET = 6666

# Test sets: 1000 each, label-balanced
TEST_SAMPLES_PER_SET = 1000

# Training hyperparameters
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

# Misc
RANDOM_SEED = 42
LOGGING_STEPS = 25
EVAL_STEPS = 200
SAVE_STEPS = 200
REPORT_TO = "none"

# Quantization (4-bit QLoRA)
USE_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_USE_DOUBLE_QUANT = True
