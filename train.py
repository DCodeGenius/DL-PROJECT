"""
Main training script for fine-tuning Llama-3.1-8B on FEVER dataset.
Uses QLoRA (4-bit quantization + LoRA) for efficient fine-tuning.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import config
from preprocess_data import load_and_preprocess_fever

def setup_model_and_tokenizer():
    """Load model and tokenizer with 4-bit quantization."""
    print(f"Loading model: {config.MODEL_ID}")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.USE_4BIT,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=getattr(torch, config.BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    print("✅ Model and tokenizer loaded!")
    return model, tokenizer

def setup_lora(model):
    """Configure LoRA adapters."""
    # Target modules for Llama-3.1 architecture
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("FEVER Fine-Tuning Pipeline")
    print("=" * 60)
    
    # Load and preprocess data
    train_ds, eval_ds = load_and_preprocess_fever()
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer()
    model = setup_lora(model)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        bf16=True,
        logging_steps=config.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        warmup_ratio=config.WARMUP_RATIO,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        report_to=config.REPORT_TO,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=config.MAX_SEQ_LEN,
        dataset_text_field="text",
    )
    
    # Train!
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Save final model
    final_model_path = f"{config.OUTPUT_DIR}/final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\n✅ Training complete! Model saved to: {final_model_path}")

if __name__ == "__main__":
    main()
