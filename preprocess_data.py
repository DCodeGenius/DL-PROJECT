"""
Data preprocessing for FEVER dataset.
Binary classification (SUPPORTS vs REFUTES only — NEI excluded).
Claim-only prompts (no evidence) for hallucination analysis.
Extracts Wikipedia page metadata and computes per-example frequency
using max(freq) across all referenced pages.
"""

import json
import os
from collections import Counter
from datasets import load_dataset
import config

LABEL_MAP = {
    0: "SUPPORTS",
    1: "REFUTES",
}

LABEL_MAP_REVERSE = {
    "SUPPORTS": 0,
    "REFUTES": 1,
}


def extract_wiki_pages(example):
    """
    Extract all Wikipedia page titles referenced by this claim.
    Returns a list of cleaned page names.
    """
    pages = []
    wiki_url = example.get("evidence_wiki_url", "")
    if wiki_url:
        raw = str(wiki_url)
        for part in raw.split(","):
            page = part.strip().replace("_", " ")
            page = page.replace("-LRB-", "(").replace("-RRB-", ")")
            if page and page not in pages:
                pages.append(page)
    return pages


def compute_page_frequencies(dataset):
    """Count how many claims reference each Wikipedia page (train split only)."""
    page_counter = Counter()
    for example in dataset:
        for page in extract_wiki_pages(example):
            page_counter[page] += 1
    return page_counter


def compute_example_freq(pages, page_frequencies):
    """
    Compute a single frequency value for a claim.
    Uses max(freq) across all referenced pages.
    """
    if not pages:
        return 0
    freqs = [page_frequencies.get(p, 0) for p in pages]
    return max(freqs)


def create_format_fn(page_frequencies):
    """Create formatting function with frequency data baked in."""
    def format_prompt(example):
        claim = example["claim"]

        label_raw = example["label"]
        if isinstance(label_raw, str):
            label = label_raw
            label_int = LABEL_MAP_REVERSE.get(label_raw, -1)
        else:
            label = LABEL_MAP.get(label_raw, "")
            label_int = label_raw

        wiki_pages = extract_wiki_pages(example)
        primary_page = wiki_pages[0] if wiki_pages else "UNKNOWN"
        freq = compute_example_freq(wiki_pages, page_frequencies)

        # Claim-only prompt — no evidence provided
        text = (
            f"### Instruction:\n"
            f"Based on your knowledge, classify the following claim as "
            f"SUPPORTS (true) or REFUTES (false).\n\n"
            f"### Claim:\n{claim}\n\n"
            f"### Answer:\n{label}"
        )

        return {
            "text": text,
            "label": label_int,
            "claim": claim,
            "wiki_page": primary_page,
            "page_frequency": freq,
        }
    return format_prompt


def filter_binary(dataset):
    """Filter dataset to only SUPPORTS (0) and REFUTES (1), removing NEI."""
    label_col = dataset[0]["label"] if len(dataset) > 0 else None

    if isinstance(label_col, str):
        return dataset.filter(lambda x: x["label"] in ("SUPPORTS", "REFUTES"))
    else:
        return dataset.filter(lambda x: x["label"] in (0, 1))


def load_and_preprocess_fever():
    """Load FEVER dataset and apply preprocessing (binary, claim-only)."""
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
            try:
                from datasets import load_dataset_builder
                builder = load_dataset_builder("fever", "v1.0")
                builder.download_and_prepare()
                dataset = builder.as_dataset()
            except Exception as e3:
                print(f"Builder method failed: {e3}")
                dataset = load_dataset("fever", "v1.0", verification_mode="no_checks")

    print(f"Full train size: {len(dataset['train'])}")
    print(f"Full labelled dev size: {len(dataset['labelled_dev'])}")

    # Filter to binary (SUPPORTS + REFUTES only)
    print("\nFiltering to binary classification (dropping NEI)...")
    train_full = filter_binary(dataset["train"])
    dev_full = filter_binary(dataset["labelled_dev"])
    print(f"Binary train size: {len(train_full)}")
    print(f"Binary dev size: {len(dev_full)}")

    # Compute page frequencies on the FULL binary training set
    print("\nComputing Wikipedia page frequencies (train split only)...")
    page_frequencies = compute_page_frequencies(train_full)
    print(f"Found {len(page_frequencies)} unique Wikipedia pages")

    # Save frequency data
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    freq_path = f"{config.OUTPUT_DIR}/page_frequencies.json"
    with open(freq_path, "w") as f:
        json.dump(dict(page_frequencies.most_common()), f, indent=2)
    print(f"Saved page frequencies to {freq_path}")

    # Print frequency distribution stats
    freq_values = list(page_frequencies.values())
    print(f"\nFrequency stats across {len(freq_values)} pages:")
    print(f"  Min: {min(freq_values)}, Max: {max(freq_values)}, "
          f"Mean: {sum(freq_values)/len(freq_values):.1f}, "
          f"Median: {sorted(freq_values)[len(freq_values)//2]}")

    # Create formatting function
    format_fn = create_format_fn(page_frequencies)

    # Process training set
    print("\nPreprocessing training data...")
    train_ds = train_full.shuffle(seed=config.RANDOM_SEED)
    if config.TRAIN_SAMPLES:
        train_ds = train_ds.select(range(min(config.TRAIN_SAMPLES, len(train_ds))))
    train_ds = train_ds.map(format_fn, remove_columns=train_full.column_names)

    # Process eval set
    print("Preprocessing evaluation data...")
    eval_ds = dev_full.shuffle(seed=config.RANDOM_SEED)
    if config.EVAL_SAMPLES:
        eval_ds = eval_ds.select(range(min(config.EVAL_SAMPLES, len(eval_ds))))
    eval_ds = eval_ds.map(format_fn, remove_columns=dev_full.column_names)

    # Save eval metadata for analysis
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
