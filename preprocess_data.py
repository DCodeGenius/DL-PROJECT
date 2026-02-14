"""
Data preprocessing for FEVER hallucination study.
Binary classification (SUPPORTS vs REFUTES only).
Controlled frequency manipulation: High-only vs Low-only training.

Experimental design:
  - Frequency = #train claims referencing a Wikipedia page (train-derived only)
  - H = top 30% by example_freq, L = bottom 30%, middle 40% dropped
  - Eval: balanced 50% H / 50% L, within each 50% SUP / 50% REF (4 cells)
  - Train A (high-only): 100% H, 50/50 labels
  - Train B (low-only):  100% L, 50/50 labels
"""

import json
import os
from collections import Counter
from datasets import load_dataset, Dataset
import numpy as np
import config

LABEL_MAP = {0: "SUPPORTS", 1: "REFUTES"}
LABEL_MAP_REVERSE = {"SUPPORTS": 0, "REFUTES": 1}


def extract_wiki_pages(example):
    """Extract all Wikipedia page titles referenced by this claim."""
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
    """Count how many claims reference each Wikipedia page."""
    counter = Counter()
    for example in dataset:
        for page in extract_wiki_pages(example):
            counter[page] += 1
    return counter


def example_freq(pages, page_frequencies):
    """max(freq) across all referenced pages."""
    if not pages:
        return 0
    return max(page_frequencies.get(p, 0) for p in pages)


def filter_binary(dataset):
    """Keep only SUPPORTS and REFUTES."""
    sample = dataset[0]["label"]
    if isinstance(sample, str):
        return dataset.filter(lambda x: x["label"] in ("SUPPORTS", "REFUTES"))
    else:
        return dataset.filter(lambda x: x["label"] in (0, 1))


def get_label_int(example):
    """Normalize label to int."""
    lab = example["label"]
    if isinstance(lab, str):
        return LABEL_MAP_REVERSE.get(lab, -1)
    return lab


def build_prompt(claim, label_str=None):
    """Build claim-only prompt. If label_str given, append it (for training)."""
    prompt = (
        f"### Instruction:\n"
        f"Based on your knowledge, classify the following claim as "
        f"SUPPORTS (true) or REFUTES (false).\n\n"
        f"### Claim:\n{claim}\n\n"
        f"### Answer:\n"
    )
    if label_str is not None:
        prompt += label_str
    return prompt


def annotate_with_frequency(dataset, page_frequencies):
    """Add wiki_page, page_frequency, label_int, freq_bucket to each example."""
    annotated = []
    for example in dataset:
        pages = extract_wiki_pages(example)
        primary_page = pages[0] if pages else "UNKNOWN"
        freq = example_freq(pages, page_frequencies)
        label_int = get_label_int(example)
        label_str = LABEL_MAP.get(label_int, str(example["label"]))
        annotated.append({
            "claim": example["claim"],
            "label_int": label_int,
            "label_str": label_str,
            "wiki_page": primary_page,
            "page_frequency": freq,
        })
    return annotated


def compute_frequency_thresholds(annotated_train):
    """Compute H/L thresholds: top 30% = H, bottom 30% = L."""
    freqs = [ex["page_frequency"] for ex in annotated_train if ex["wiki_page"] != "UNKNOWN"]
    freqs_arr = np.array(freqs)

    p30 = np.percentile(freqs_arr, 30)
    p70 = np.percentile(freqs_arr, 70)

    print(f"\nFrequency thresholds (train-derived):")
    print(f"  Total examples: {len(freqs)}")
    print(f"  Min: {freqs_arr.min()}, Max: {freqs_arr.max()}, "
          f"Mean: {freqs_arr.mean():.1f}, Median: {np.median(freqs_arr):.0f}")
    print(f"  30th percentile (L ceiling): {p30}")
    print(f"  70th percentile (H floor):   {p70}")

    return p30, p70


def assign_bucket(freq, p30, p70):
    """Assign H, L, or MID bucket."""
    if freq <= p30:
        return "L"
    elif freq >= p70:
        return "H"
    else:
        return "MID"


def stratified_sample(annotated, bucket, label_int, n, seed=42):
    """Sample n examples matching a given bucket and label."""
    candidates = [ex for ex in annotated if ex["bucket"] == bucket and ex["label_int"] == label_int]
    rng = np.random.RandomState(seed)
    if len(candidates) < n:
        print(f"  ⚠️  Only {len(candidates)} candidates for bucket={bucket}, label={label_int}, need {n}")
        return candidates
    indices = rng.choice(len(candidates), size=n, replace=False)
    return [candidates[i] for i in indices]


def build_hf_dataset(examples):
    """Convert list of dicts to HuggingFace Dataset with 'text' column for training."""
    records = []
    for ex in examples:
        text = build_prompt(ex["claim"], label_str=ex["label_str"])
        records.append({
            "text": text,
            "claim": ex["claim"],
            "label": ex["label_int"],
            "wiki_page": ex["wiki_page"],
            "page_frequency": ex["page_frequency"],
            "bucket": ex["bucket"],
        })
    return Dataset.from_list(records)


def deduplicate_eval_from_train(eval_examples, train_examples):
    """Remove any eval claims that appear in the training set."""
    train_claims = set(ex["claim"] for ex in train_examples)
    before = len(eval_examples)
    filtered = [ex for ex in eval_examples if ex["claim"] not in train_claims]
    removed = before - len(filtered)
    if removed > 0:
        print(f"  Removed {removed} eval examples that overlap with training")
    return filtered


def load_and_preprocess_fever():
    """
    Main preprocessing pipeline.
    Returns train_ds, eval_ds for the current experiment (set by config.EXPERIMENT).
    Also saves all datasets and metadata.
    """
    print("Loading FEVER dataset...")
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        dataset = load_dataset("fever", "v1.0")
    except Exception:
        try:
            dataset = load_dataset("fever", "v1.0", trust_remote_code=True)
        except Exception:
            dataset = load_dataset("fever", "v1.0", verification_mode="no_checks")

    print(f"Full train size: {len(dataset['train'])}")
    print(f"Full dev size: {len(dataset['labelled_dev'])}")

    # Binary filter
    print("\nFiltering to binary (dropping NEI)...")
    train_full = filter_binary(dataset["train"])
    dev_full = filter_binary(dataset["labelled_dev"])
    print(f"Binary train: {len(train_full)}, Binary dev: {len(dev_full)}")

    # Compute page frequencies on full binary training set
    print("\nComputing page frequencies (train only)...")
    page_frequencies = compute_page_frequencies(train_full)
    print(f"Found {len(page_frequencies)} unique Wikipedia pages")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    freq_path = f"{config.OUTPUT_DIR}/page_frequencies.json"
    with open(freq_path, "w") as f:
        json.dump(dict(page_frequencies.most_common()), f, indent=2)

    # Annotate all examples with frequency
    print("\nAnnotating train examples with frequency...")
    annotated_train = annotate_with_frequency(train_full, page_frequencies)
    print("Annotating dev examples with frequency...")
    annotated_dev = annotate_with_frequency(dev_full, page_frequencies)

    # Compute H/L thresholds
    p30, p70 = compute_frequency_thresholds(annotated_train)

    # Assign buckets
    for ex in annotated_train:
        ex["bucket"] = assign_bucket(ex["page_frequency"], p30, p70)
    for ex in annotated_dev:
        ex["bucket"] = assign_bucket(ex["page_frequency"], p30, p70)

    # Print bucket distribution
    for split_name, data in [("Train", annotated_train), ("Dev", annotated_dev)]:
        counts = Counter(ex["bucket"] for ex in data)
        print(f"\n{split_name} bucket distribution:")
        for b in ["L", "MID", "H"]:
            print(f"  {b}: {counts.get(b, 0)}")

    # === BUILD EVAL SET (balanced-balanced) ===
    eval_per_cell = config.EVAL_SAMPLES // 4
    print(f"\nBuilding eval set: {eval_per_cell} per cell × 4 = {eval_per_cell * 4} total")

    eval_examples = []
    for bucket in ["H", "L"]:
        for label_int in [0, 1]:
            label_name = LABEL_MAP[label_int]
            cell = stratified_sample(annotated_dev, bucket, label_int, eval_per_cell, seed=config.RANDOM_SEED)
            print(f"  Eval {bucket}+{label_name}: {len(cell)} examples")
            eval_examples.extend(cell)

    # === BUILD TRAINING SETS ===
    train_per_label = config.TRAIN_SAMPLES // 2
    experiment = getattr(config, "EXPERIMENT", "high")

    print(f"\nBuilding training set for experiment: {experiment}")
    print(f"  {train_per_label} per label × 2 = {config.TRAIN_SAMPLES} total")

    if experiment == "high":
        train_bucket = "H"
    elif experiment == "low":
        train_bucket = "L"
    else:
        raise ValueError(f"Unknown experiment: {experiment}. Use 'high' or 'low'.")

    train_examples = []
    for label_int in [0, 1]:
        label_name = LABEL_MAP[label_int]
        cell = stratified_sample(annotated_train, train_bucket, label_int, train_per_label, seed=config.RANDOM_SEED)
        print(f"  Train {train_bucket}+{label_name}: {len(cell)} examples")
        train_examples.extend(cell)

    # Deduplicate
    eval_examples = deduplicate_eval_from_train(eval_examples, train_examples)

    # Shuffle
    rng = np.random.RandomState(config.RANDOM_SEED)
    rng.shuffle(train_examples)
    rng.shuffle(eval_examples)

    # Convert to HF datasets
    train_ds = build_hf_dataset(train_examples)
    eval_ds = build_hf_dataset(eval_examples)

    # Save eval metadata
    eval_metadata = []
    for ex in eval_examples:
        eval_metadata.append({
            "claim": ex["claim"],
            "label": ex["label_int"],
            "wiki_page": ex["wiki_page"],
            "page_frequency": ex["page_frequency"],
            "bucket": ex["bucket"],
        })
    meta_path = f"{config.OUTPUT_DIR}/eval_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(eval_metadata, f, indent=2)

    # Save experiment info
    exp_info = {
        "experiment": experiment,
        "train_bucket": train_bucket if experiment != "balanced" else "H+L",
        "train_size": len(train_examples),
        "eval_size": len(eval_examples),
        "p30_threshold": float(p30),
        "p70_threshold": float(p70),
        "train_per_label": train_per_label,
        "eval_per_cell": eval_per_cell,
        "seed": config.RANDOM_SEED,
    }
    with open(f"{config.OUTPUT_DIR}/experiment_info.json", "w") as f:
        json.dump(exp_info, f, indent=2)

    print(f"\n✅ Preprocessing complete!")
    print(f"  Experiment: {experiment}")
    print(f"  Train: {len(train_ds)} examples (bucket={train_bucket})")
    print(f"  Eval:  {len(eval_ds)} examples (balanced H/L × SUP/REF)")
    print(f"\nSample:")
    print(train_ds[0]["text"][:400] + "...")
    print(f"  Wiki page: {train_ds[0]['wiki_page']}, freq: {train_ds[0]['page_frequency']}, bucket: {train_ds[0]['bucket']}")

    return train_ds, eval_ds


if __name__ == "__main__":
    train_ds, eval_ds = load_and_preprocess_fever()
