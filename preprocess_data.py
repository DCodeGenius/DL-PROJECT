"""
Data preprocessing for FEVER dataset.
Formats claims + evidence into instruction-following prompts.
"""

from datasets import load_dataset
import config

LABEL_MAP = {
    0: "SUPPORTS",
    1: "REFUTES", 
    2: "NOT ENOUGH INFO"
}

def format_prompt(example):
    """
    Format FEVER example into instruction-following prompt.
    
    Decision 1 (Option B): Model sees claim + evidence → predicts label
    """
    claim = example["claim"]
    label = LABEL_MAP[example["label"]]
    
    # Extract evidence text (FEVER provides evidence sentences)
    evidence_text = ""
    if "evidence" in example and example["evidence"]:
        # evidence is a list of lists: [[[doc_id, sent_id, text], ...], ...]
        evidence_sentences = []
        for ev_group in example["evidence"]:
            for ev in ev_group:
                if len(ev) >= 3:
                    evidence_sentences.append(ev[2])  # The text is at index 2
        
        evidence_text = " ".join(evidence_sentences[:5])  # Limit to first 5 sentences
    
    # Format as instruction-following prompt
    text = (
        f"### Instruction:\n"
        f"Given the following claim and evidence, classify whether the evidence "
        f"SUPPORTS, REFUTES, or provides NOT ENOUGH INFO for the claim.\n\n"
        f"### Claim:\n{claim}\n\n"
        f"### Evidence:\n{evidence_text}\n\n"
        f"### Answer:\n{label}"
    )
    
    return {"text": text, "label": example["label"], "claim": claim}

def load_and_preprocess_fever():
    """Load FEVER dataset and apply preprocessing."""
    print("Loading FEVER dataset...")
    dataset = load_dataset("fever", "v1.0", trust_remote_code=True)
    
    print(f"Train size: {len(dataset['train'])}")
    print(f"Labelled dev size: {len(dataset['labelled_dev'])}")
    
    # Process training set
    print("Preprocessing training data...")
    train_ds = dataset["train"].shuffle(seed=config.RANDOM_SEED)
    if config.TRAIN_SAMPLES:
        train_ds = train_ds.select(range(min(config.TRAIN_SAMPLES, len(train_ds))))
    train_ds = train_ds.map(format_prompt, remove_columns=dataset["train"].column_names)
    
    # Process eval set
    print("Preprocessing evaluation data...")
    eval_ds = dataset["labelled_dev"].shuffle(seed=config.RANDOM_SEED)
    if config.EVAL_SAMPLES:
        eval_ds = eval_ds.select(range(min(config.EVAL_SAMPLES, len(eval_ds))))
    eval_ds = eval_ds.map(format_prompt, remove_columns=dataset["labelled_dev"].column_names)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"Train examples: {len(train_ds)}")
    print(f"Eval examples: {len(eval_ds)}")
    print(f"\nSample formatted example:")
    print(train_ds[0]["text"][:500] + "...")
    
    return train_ds, eval_ds

if __name__ == "__main__":
    train_ds, eval_ds = load_and_preprocess_fever()
    print("\n✅ Data ready for training!")
