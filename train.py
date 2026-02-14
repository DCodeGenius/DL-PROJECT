"""
Fine-tuning script for FEVER hallucination study.
Trains a single model on balanced H/M/L frequency data using QLoRA.

Usage:
    python train.py
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import config
from preprocess_data import load_and_preprocess_fever


def setup_model_and_tokenizer():
    """Load model with 4-bit quantization."""
    print(f"Loading model: {config.MODEL_ID}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.USE_4BIT,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=getattr(torch, config.BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model)
    print("Model and tokenizer loaded!")
    return model, tokenizer


def setup_lora(model):
    """Configure LoRA adapters."""
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def tokenize_dataset(dataset, tokenizer, max_length):
    """Tokenize for causal LM."""
    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)


def main():
    print("=" * 60)
    print("FEVER Fine-Tuning (Single Model, Balanced H/M/L)")
    print("=" * 60)

    train_ds, eval_ds = load_and_preprocess_fever()

    model, tokenizer = setup_model_and_tokenizer()
    model = setup_lora(model)

    print("\nTokenizing datasets...")
    train_tok = tokenize_dataset(train_ds, tokenizer, config.MAX_SEQ_LEN)
    eval_tok = tokenize_dataset(eval_ds, tokenizer, config.MAX_SEQ_LEN)
    print("Tokenization complete!")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
        warmup_steps=10,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        report_to=config.REPORT_TO,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    final_path = f"{config.OUTPUT_DIR}/final_model"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nTraining complete! Model saved to: {final_path}")


if __name__ == "__main__":
    main()
