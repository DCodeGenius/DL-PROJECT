"""
Data preprocessing for FEVER hallucination study.
Binary classification (SUPPORTS vs REFUTES).

Builds:
  - 1 training set: 20K total, ~6666 H + ~6666 M + ~6666 L, 50/50 labels per bucket
  - 3 test sets from FEVER dev (label-balanced, no claim overlap with train):
      test_high:  1000 examples from High-frequency bucket
      test_low:   1000 examples from Low-frequency bucket
      test_mixed: 1000 examples (~333 H + ~333 M + ~333 L)

Frequency bucketing: percentile-based on UNIQUE PAGE frequencies.
  We compute page_freq for every Wikipedia page in train, then use the 33rd
  and 66th percentiles of that page-level distribution to define buckets:
    Low:    example_freq <= p33_of_pages
    Medium: p33 < example_freq <= p67
    High:   example_freq > p67
  Dev examples are bucketed using the SAME thresholds derived from train.
"""

import json
import os
from collections import Counter
from datasets import load_dataset, Dataset
import numpy as np
import config

LABEL_MAP = {0: "SUPPORTS", 1: "REFUTES"}
LABEL_MAP_REVERSE = {"SUPPORTS": 0, "REFUTES": 1}


# --- Helpers ---------------------------------------------------------------

def extract_wiki_pages(example):
    """Extract all Wikipedia page titles referenced by a claim."""
    pages = []
    wiki_url = example.get("evidence_wiki_url", "")
    if wiki_url:
        for part in str(wiki_url).split(","):
            page = part.strip().replace("_", " ")
            page = page.replace("-LRB-", "(").replace("-RRB-", ")")
            if page and page not in pages:
                pages.append(page)
    return pages


def compute_page_frequencies(dataset):
    """Count how many claims reference each Wikipedia page (train only)."""
    counter = Counter()
    for example in dataset:
        for page in extract_wiki_pages(example):
            counter[page] += 1
    return counter


def get_example_freq(pages, page_frequencies):
    """max(freq) across all referenced pages for a single claim."""
    if not pages:
        return 0
    return max(page_frequencies.get(p, 0) for p in pages)


def get_label_int(example):
    """Normalize label to int regardless of dataset format."""
    lab = example["label"]
    if isinstance(lab, str):
        return LABEL_MAP_REVERSE.get(lab, -1)
    return lab


def compute_freq_thresholds(page_frequencies):
    """
    Compute percentile-based thresholds on the UNIQUE PAGE frequency distribution.
    This avoids skew from popular pages that generate many examples.
    Returns (low_cutoff, high_cutoff).
    """
    unique_freqs = np.array(list(page_frequencies.values()))
    low_cutoff = np.percentile(unique_freqs, config.FREQ_LOW_PERCENTILE)
    high_cutoff = np.percentile(unique_freqs, config.FREQ_HIGH_PERCENTILE)
    print(f"  Page frequency stats: min={unique_freqs.min()}, median={np.median(unique_freqs):.0f}, "
          f"mean={unique_freqs.mean():.1f}, max={unique_freqs.max()}")
    return float(low_cutoff), float(high_cutoff)


def assign_bucket(freq, low_cutoff, high_cutoff):
    """Assign frequency bucket using data-derived percentile thresholds."""
    if freq > high_cutoff:
        return "high"
    elif freq > low_cutoff:
        return "medium"
    else:
        return "low"


def filter_binary(dataset):
    """Keep only SUPPORTS and REFUTES."""
    sample = dataset[0]["label"]
    if isinstance(sample, str):
        return dataset.filter(lambda x: x["label"] in ("SUPPORTS", "REFUTES"))
    return dataset.filter(lambda x: x["label"] in (0, 1))


def annotate_examples_raw(dataset, page_frequencies):
    """Annotate each example with frequency and label (no bucket yet)."""
    annotated = []
    for example in dataset:
        pages = extract_wiki_pages(example)
        freq = get_example_freq(pages, page_frequencies)
        label_int = get_label_int(example)

        annotated.append({
            "claim": example["claim"],
            "label_int": label_int,
            "label_str": LABEL_MAP.get(label_int, "UNKNOWN"),
            "wiki_page": pages[0] if pages else "UNKNOWN",
            "page_frequency": freq,
        })
    return annotated


def apply_buckets(annotated, low_cutoff, high_cutoff):
    """Assign bucket to each annotated example using the given thresholds."""
    for e in annotated:
        e["bucket"] = assign_bucket(e["page_frequency"], low_cutoff, high_cutoff)
    return annotated


def balanced_sample(examples, bucket, n_per_label, seed):
    """Sample n_per_label SUPPORTS + n_per_label REFUTES from a given bucket."""
    rng = np.random.RandomState(seed)
    sampled = []
    for label_int in [0, 1]:
        candidates = [e for e in examples if e["bucket"] == bucket and e["label_int"] == label_int]
        n = min(n_per_label, len(candidates))
        if n < n_per_label:
            print(f"  WARNING: only {len(candidates)} candidates for bucket={bucket}, "
                  f"label={LABEL_MAP[label_int]}, need {n_per_label}")
        indices = rng.choice(len(candidates), size=n, replace=False)
        sampled.extend(candidates[i] for i in indices)
    return sampled


def build_prompt(claim, label_str=None):
    """Build claim-only prompt. Append label for training examples."""
    prompt = (
        "### Instruction:\n"
        "Based on your knowledge, classify the following claim as "
        "SUPPORTS (true) or REFUTES (false).\n\n"
        f"### Claim:\n{claim}\n\n"
        "### Answer:\n"
    )
    if label_str is not None:
        prompt += label_str
    return prompt


def to_hf_dataset(examples):
    """Convert annotated examples to HuggingFace Dataset with text column."""
    records = []
    for ex in examples:
        records.append({
            "text": build_prompt(ex["claim"], label_str=ex["label_str"]),
            "claim": ex["claim"],
            "label": ex["label_int"],
            "wiki_page": ex["wiki_page"],
            "page_frequency": ex["page_frequency"],
            "bucket": ex["bucket"],
        })
    return Dataset.from_list(records)


# --- Main pipeline ---------------------------------------------------------

def load_and_preprocess_fever():
    """
    Main preprocessing. Returns (train_ds, eval_ds_for_training_loss).
    Also saves 3 test set metadata files and all experiment info.
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

    print(f"Full train: {len(dataset['train'])}")
    print(f"Full dev:   {len(dataset['labelled_dev'])}")

    # Binary filter
    print("\nFiltering to binary (dropping NEI)...")
    train_full = filter_binary(dataset["train"])
    dev_full = filter_binary(dataset["labelled_dev"])
    print(f"Binary train: {len(train_full)}")
    print(f"Binary dev:   {len(dev_full)}")

    # Compute page frequencies on FULL binary training set
    print("\nComputing page frequencies (train only)...")
    page_frequencies = compute_page_frequencies(train_full)
    print(f"Unique Wikipedia pages: {len(page_frequencies)}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(f"{config.OUTPUT_DIR}/page_frequencies.json", "w") as f:
        json.dump(dict(page_frequencies.most_common()), f, indent=2)

    # Compute percentile-based thresholds on UNIQUE PAGE frequencies
    print("\nComputing percentile thresholds on unique page frequencies...")
    low_cutoff, high_cutoff = compute_freq_thresholds(page_frequencies)
    print(f"  p{config.FREQ_LOW_PERCENTILE} = {low_cutoff:.0f}  (Low: freq <= {low_cutoff:.0f})")
    print(f"  p{config.FREQ_HIGH_PERCENTILE} = {high_cutoff:.0f}  (High: freq > {high_cutoff:.0f})")
    print(f"  Medium: {low_cutoff:.0f} < freq <= {high_cutoff:.0f}")

    # Annotate all examples (frequency + label, no bucket yet)
    print("\nAnnotating examples with frequency...")
    ann_train = annotate_examples_raw(train_full, page_frequencies)
    ann_dev = annotate_examples_raw(dev_full, page_frequencies)

    # Apply buckets using train-derived thresholds to BOTH sets
    ann_train = apply_buckets(ann_train, low_cutoff, high_cutoff)
    ann_dev = apply_buckets(ann_dev, low_cutoff, high_cutoff)

    # Print bucket distribution
    for name, data in [("Train", ann_train), ("Dev", ann_dev)]:
        counts = Counter(e["bucket"] for e in data)
        print(f"\n{name} bucket distribution:")
        for b in ["high", "medium", "low"]:
            print(f"  {b}: {counts.get(b, 0)}")

    # ---- BUILD TRAINING SET ------------------------------------------------
    n_per_label_per_bucket = config.TRAIN_PER_BUCKET // 2
    print(f"\nBuilding training set: {config.TRAIN_PER_BUCKET} per bucket "
          f"({n_per_label_per_bucket} per label), 3 buckets")

    train_examples = []
    for bucket in ["high", "medium", "low"]:
        sampled = balanced_sample(ann_train, bucket, n_per_label_per_bucket, config.RANDOM_SEED)
        print(f"  Train {bucket}: {len(sampled)} examples")
        train_examples.extend(sampled)

    train_claims = set(e["claim"] for e in train_examples)
    print(f"Total training examples: {len(train_examples)}")

    # ---- BUILD TEST SETS (from dev, no claim overlap) ----------------------
    ann_dev_clean = [e for e in ann_dev if e["claim"] not in train_claims]
    removed = len(ann_dev) - len(ann_dev_clean)
    if removed > 0:
        print(f"\nRemoved {removed} dev examples that overlap with training claims")

    test_per_label = config.TEST_SAMPLES_PER_SET // 2

    # Test-High
    print(f"\nBuilding test sets: {config.TEST_SAMPLES_PER_SET} per set ({test_per_label} per label)")
    test_high = balanced_sample(ann_dev_clean, "high", test_per_label, config.RANDOM_SEED + 1)
    print(f"  test_high:  {len(test_high)} examples")

    # Test-Low
    test_low = balanced_sample(ann_dev_clean, "low", test_per_label, config.RANDOM_SEED + 2)
    print(f"  test_low:   {len(test_low)} examples")

    # Test-Mixed: ~333 per bucket, label-balanced within each
    mixed_per_bucket = config.TEST_SAMPLES_PER_SET // 3
    mixed_per_label_per_bucket = mixed_per_bucket // 2

    used_claims = set(e["claim"] for e in test_high + test_low)
    ann_dev_remaining = [e for e in ann_dev_clean if e["claim"] not in used_claims]

    test_mixed = []
    for bucket in ["high", "medium", "low"]:
        sampled = balanced_sample(ann_dev_remaining, bucket, mixed_per_label_per_bucket, config.RANDOM_SEED + 3)
        print(f"  test_mixed {bucket}: {len(sampled)} examples")
        test_mixed.extend(sampled)
    print(f"  test_mixed total: {len(test_mixed)} examples")

    # Shuffle all sets
    rng = np.random.RandomState(config.RANDOM_SEED)
    rng.shuffle(train_examples)
    rng.shuffle(test_high)
    rng.shuffle(test_low)
    rng.shuffle(test_mixed)

    # ---- SAVE TEST SET METADATA --------------------------------------------
    for name, data in [("test_high", test_high), ("test_low", test_low), ("test_mixed", test_mixed)]:
        meta = [{
            "claim": e["claim"],
            "label": e["label_int"],
            "wiki_page": e["wiki_page"],
            "page_frequency": e["page_frequency"],
            "bucket": e["bucket"],
        } for e in data]
        path = f"{config.OUTPUT_DIR}/{name}_metadata.json"
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved {path} ({len(meta)} examples)")

    # Save experiment info
    exp_info = {
        "bucketing_method": "percentile_on_unique_pages",
        "low_percentile": config.FREQ_LOW_PERCENTILE,
        "high_percentile": config.FREQ_HIGH_PERCENTILE,
        "low_cutoff": low_cutoff,
        "high_cutoff": high_cutoff,
        "bucket_rule": f"Low: freq <= {low_cutoff:.0f}, "
                       f"Medium: {low_cutoff:.0f} < freq <= {high_cutoff:.0f}, "
                       f"High: freq > {high_cutoff:.0f}",
        "train_size": len(train_examples),
        "train_per_bucket": config.TRAIN_PER_BUCKET,
        "test_high_size": len(test_high),
        "test_low_size": len(test_low),
        "test_mixed_size": len(test_mixed),
        "seed": config.RANDOM_SEED,
        "claims_removed_overlap": removed,
    }
    with open(f"{config.OUTPUT_DIR}/experiment_info.json", "w") as f:
        json.dump(exp_info, f, indent=2)

    # ---- CONVERT TO HF DATASETS -------------------------------------------
    train_ds = to_hf_dataset(train_examples)
    eval_ds = to_hf_dataset(test_mixed)

    print(f"\n{'='*60}")
    print(f"Preprocessing complete!")
    print(f"  Bucket thresholds: Low <= {low_cutoff:.0f} | "
          f"Medium {low_cutoff:.0f}-{high_cutoff:.0f} | High > {high_cutoff:.0f}")
    print(f"  Train: {len(train_ds)} examples (balanced H/M/L + labels)")
    print(f"  Test-High: {len(test_high)} | Test-Low: {len(test_low)} | Test-Mixed: {len(test_mixed)}")
    print(f"{'='*60}")
    print(f"\nSample training example:")
    print(train_ds[0]["text"][:400] + "...")
    print(f"  page: {train_ds[0]['wiki_page']}, freq: {train_ds[0]['page_frequency']}, bucket: {train_ds[0]['bucket']}")

    return train_ds, eval_ds


if __name__ == "__main__":
    load_and_preprocess_fever()
