"""
Data preprocessing for FEVER dataset.
Claim-only prompts (no evidence) for hallucination analysis.
Extracts Wikipedia page metadata for frequency analysis.
"""

import json
import os
from collections import Counter
from datasets import load_dataset
import config

LABEL_MAP = {
    0: "SUPPORTS",
    1: "REFUTES",
    2: "NOT ENOUGH INFO"
}

LABEL_MAP_REVERSE = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2
}


def extract_wiki_pages(example):
    """Extract Wikipedia page titles from evidence metadata."""
    pages = []
    if "evidence" in example and example["evidence"]:
        for ev_group in example["evidence"]:
            for ev in ev_group:
                if len(ev) >= 3 and ev[2] is not None:
                    page = str(ev[2]).replace("_", " ")
                    if page and page not in pages:
                        pages.append(page)
    return pages


def compute_page_frequencies(dataset):
    """Count how many claims reference each Wikipedia page across the full dataset."""
    page_counter = Counter()
    for example in dataset:
        pages = extract_wiki_pages(example)
        for page in pages:
            page_counter[page] += 1
    return page_counter


def create_format_fn(page_frequencies):
    """Create formatting function with frequency data baked in."""
    def format_prompt(example):
        claim = example["claim"]

        label_raw = example["label"]
        if isinstance(label_raw, str):
            label = label_raw
            label_int = LABEL_MAP_REVERSE.get(label_raw, 0)
        else:
            label = LABEL_MAP[label_raw]
            label_int = label_raw

        wiki_pages = extract_wiki_pages(example)
        wiki_page = wiki_pages[0] if wiki_pages else "UNKNOWN"
        freq = page_frequencies.get(wiki_page, 0) if wiki_page != "UNKNOWN" else 0

        # Claim-only prompt — no evidence provided
        text = (
            f"### Instruction:\n"
            f"Based on your knowledge, classify the following claim as "
            f"SUPPORTS (true), REFUTES (false), or NOT ENOUGH INFO.\n\n"
            f"### Claim:\n{claim}\n\n"
            f"### Answer:\n{label}"
        )

        return {
            "text": text,
            "label": label_int,
            "claim": claim,
            "wiki_page": wiki_page,
            "page_frequency": freq,
        }
    return format_prompt


def load_and_preprocess_fever():
    """Load FEVER dataset and apply preprocessing."""
    print("Loading FEVER dataset...")
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        dataset = load_dataset("fever", "v1.0")
    except Exception as e1:
        print(f"Standard load failed: {e1}")
        try:
            print("Trying with trust_remote_code...")
            dataset = load_dataset("fever", "v1.0", trust_remote_code=True)
        except Exception as e2:
            print(f"Loading with trust_remote_code failed: {e2}")
            print("Attempting alternative loading method...")
            try:
                from datasets import load_dataset_builder
                builder = load_dataset_builder("fever", "v1.0")
                builder.download_and_prepare()
                dataset = builder.as_dataset()
            except Exception as e3:
                print(f"Builder method failed: {e3}")
                print("Trying to load from cache or different method...")
                dataset = load_dataset("fever", "v1.0", verification_mode="no_checks")

    print(f"Train size: {len(dataset['train'])}")
    print(f"Labelled dev size: {len(dataset['labelled_dev'])}")

    # Compute page frequencies on the FULL training set (before sampling)
    print("Computing Wikipedia page frequencies on full dataset...")
    page_frequencies = compute_page_frequencies(dataset["train"])
    print(f"Found {len(page_frequencies)} unique Wikipedia pages")

    # Save frequency data for later analysis
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    freq_path = f"{config.OUTPUT_DIR}/page_frequencies.json"
    with open(freq_path, "w") as f:
        json.dump(dict(page_frequencies.most_common()), f, indent=2)
    print(f"Saved page frequencies to {freq_path}")

    # Create formatting function with frequency data
    format_fn = create_format_fn(page_frequencies)

    # Process training set
    print("Preprocessing training data...")
    train_ds = dataset["train"].shuffle(seed=config.RANDOM_SEED)
    if config.TRAIN_SAMPLES:
        train_ds = train_ds.select(range(min(config.TRAIN_SAMPLES, len(train_ds))))
    train_ds = train_ds.map(format_fn, remove_columns=dataset["train"].column_names)

    # Process eval set
    print("Preprocessing evaluation data...")
    eval_ds = dataset["labelled_dev"].shuffle(seed=config.RANDOM_SEED)
    if config.EVAL_SAMPLES:
        eval_ds = eval_ds.select(range(min(config.EVAL_SAMPLES, len(eval_ds))))
    eval_ds = eval_ds.map(format_fn, remove_columns=dataset["labelled_dev"].column_names)

    # Save eval metadata for later analysis (before tokenization drops these columns)
    eval_metadata = []
    for i in range(len(eval_ds)):
        eval_metadata.append({
            "claim": eval_ds[i]["claim"],
            "label": eval_ds[i]["label"],
            "wiki_page": eval_ds[i]["wiki_page"],
            "page_frequency": eval_ds[i]["page_frequency"],
        })
    meta_path = f"{config.OUTPUT_DIR}/eval_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(eval_metadata, f, indent=2)
    print(f"Saved eval metadata to {meta_path}")

    print(f"\n✅ Preprocessing complete!")
    print(f"Train examples: {len(train_ds)}")
    print(f"Eval examples: {len(eval_ds)}")
    print(f"\nSample formatted example:")
    print(train_ds[0]["text"][:500] + "...")
    print(f"Wiki page: {train_ds[0]['wiki_page']}")
    print(f"Page frequency: {train_ds[0]['page_frequency']}")

    return train_ds, eval_ds


if __name__ == "__main__":
    train_ds, eval_ds = load_and_preprocess_fever()
    print("\n✅ Data ready for training!")
