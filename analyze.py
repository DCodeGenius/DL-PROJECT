"""
Analysis script for FEVER hallucination and confidence calibration study.
Binary classification (SUPPORTS vs REFUTES).
Compares baseline vs fine-tuned model across frequency buckets.
Produces charts and summary statistics including ECE.

Usage:
    python analyze.py                          # Analyze whatever results exist
    python analyze.py --baseline_only          # Only analyze baseline results
"""

import argparse
import json
import os
import numpy as np
import config

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not installed. Will save data but skip charts.")
    print("   Install with: pip install matplotlib")


def load_eval_metadata():
    """Load eval metadata with correct wiki page info."""
    meta_path = f"{config.OUTPUT_DIR}/eval_metadata.json"
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r") as f:
        return json.load(f)


def load_results(mode):
    """Load evaluation results and merge with current eval metadata."""
    path = f"{config.OUTPUT_DIR}/eval_results_{mode}.json"
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        results = json.load(f)

    # Merge wiki page data from eval_metadata
    metadata = load_eval_metadata()
    if metadata and len(metadata) == len(results):
        meta_by_claim = {m["claim"]: m for m in metadata}
        updated = 0
        for r in results:
            meta = meta_by_claim.get(r["claim"])
            if meta:
                r["wiki_page"] = meta["wiki_page"]
                r["page_frequency"] = meta["page_frequency"]
                updated += 1
        print(f"  Merged wiki page data for {updated}/{len(results)} examples")

    return results


def assign_frequency_buckets(results):
    """
    Assign frequency buckets based on page_frequency.
    Uses percentile-based bucketing on known pages.
    """
    known_results = [r for r in results if r.get("wiki_page", "UNKNOWN") != "UNKNOWN"]
    freqs = [r["page_frequency"] for r in known_results]

    if not freqs:
        print("⚠️  No results with known wiki pages")
        return results

    freqs_arr = np.array(freqs)
    q33 = np.percentile(freqs_arr, 33)
    q66 = np.percentile(freqs_arr, 66)

    # If percentiles collapse (very skewed), use fixed thresholds
    if q33 == q66:
        unique_freqs = sorted(set(freqs))
        if len(unique_freqs) >= 3:
            q33 = unique_freqs[len(unique_freqs) // 3]
            q66 = unique_freqs[2 * len(unique_freqs) // 3]
        elif len(unique_freqs) == 2:
            q33 = unique_freqs[0]
            q66 = unique_freqs[1]

    print(f"\nFrequency distribution ({len(known_results)} examples with known pages):")
    print(f"  Min: {freqs_arr.min()}, Max: {freqs_arr.max()}, Mean: {freqs_arr.mean():.1f}")
    print(f"  Low/Medium boundary: {q33:.0f}")
    print(f"  Medium/High boundary: {q66:.0f}")

    for r in results:
        if r.get("wiki_page", "UNKNOWN") == "UNKNOWN":
            r["freq_bucket"] = "UNKNOWN"
        elif r["page_frequency"] <= q33:
            r["freq_bucket"] = "low"
        elif r["page_frequency"] <= q66:
            r["freq_bucket"] = "medium"
        else:
            r["freq_bucket"] = "high"

    # Print bucket sizes
    for bucket in ["low", "medium", "high", "UNKNOWN"]:
        count = sum(1 for r in results if r.get("freq_bucket") == bucket)
        if count > 0:
            print(f"  {bucket}: {count} examples")

    return results


def compute_ece(results, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    ECE = sum over bins of (bin_size/total) * |accuracy(bin) - confidence(bin)|
    """
    if not results:
        return 0.0

    confidences = np.array([r["confidence"] for r in results])
    correct = np.array([1 if r["predicted_idx"] == r["true_label"] else 0 for r in results])
    total = len(results)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        bin_size = mask.sum()
        if bin_size > 0:
            bin_acc = correct[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (bin_size / total) * abs(bin_acc - bin_conf)

    return ece


def compute_metrics(results, label=""):
    """Compute accuracy, confidence, hallucination, and ECE metrics."""
    if not results:
        return {}

    total = len(results)
    correct = sum(1 for r in results if r["predicted_idx"] == r["true_label"])
    accuracy = correct / total if total > 0 else 0

    confidences = [r["confidence"] for r in results]
    avg_confidence = np.mean(confidences)

    correct_confs = [r["confidence"] for r in results if r["predicted_idx"] == r["true_label"]]
    wrong_confs = [r["confidence"] for r in results if r["predicted_idx"] != r["true_label"]]

    avg_conf_correct = np.mean(correct_confs) if correct_confs else 0
    avg_conf_wrong = np.mean(wrong_confs) if wrong_confs else 0

    # High-confidence error rate (our "hallucination" metric)
    high_conf_errors = [r for r in results if r["predicted_idx"] != r["true_label"] and r["confidence"] > 0.7]
    high_conf_error_rate = len(high_conf_errors) / total if total > 0 else 0

    # ECE
    ece = compute_ece(results)

    # Overconfidence gap
    overconfidence_gap = avg_confidence - accuracy

    return {
        "label": label,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "avg_conf_correct": avg_conf_correct,
        "avg_conf_wrong": avg_conf_wrong,
        "high_conf_error_count": len(high_conf_errors),
        "high_conf_error_rate": high_conf_error_rate,
        "ece": ece,
        "overconfidence_gap": overconfidence_gap,
    }


def print_metrics(metrics):
    """Pretty-print a metrics dictionary."""
    if not metrics:
        return
    print(f"\n  {'Metric':<30} {'Value':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Total examples':<30} {metrics['total']:>10}")
    print(f"  {'Correct':<30} {metrics['correct']:>10}")
    print(f"  {'Accuracy':<30} {metrics['accuracy']:>10.4f}")
    print(f"  {'Avg confidence (all)':<30} {metrics['avg_confidence']:>10.4f}")
    print(f"  {'Avg confidence (correct)':<30} {metrics['avg_conf_correct']:>10.4f}")
    print(f"  {'Avg confidence (wrong)':<30} {metrics['avg_conf_wrong']:>10.4f}")
    print(f"  {'High-conf error count':<30} {metrics['high_conf_error_count']:>10}")
    print(f"  {'High-conf error rate':<30} {metrics['high_conf_error_rate']:>10.4f}")
    print(f"  {'ECE':<30} {metrics['ece']:>10.4f}")
    print(f"  {'Overconfidence gap':<30} {metrics['overconfidence_gap']:>10.4f}")


def analyze_by_frequency(results, mode_name):
    """Break down metrics by frequency bucket."""
    print(f"\n{'='*60}")
    print(f"Frequency Analysis: {mode_name}")
    print(f"{'='*60}")

    results = assign_frequency_buckets(results)

    all_metrics = {}
    for bucket in ["low", "medium", "high"]:
        bucket_results = [r for r in results if r.get("freq_bucket") == bucket]
        if bucket_results:
            metrics = compute_metrics(bucket_results, label=f"{mode_name} - {bucket} freq")
            all_metrics[bucket] = metrics
            print(f"\n--- {bucket.upper()} frequency ({len(bucket_results)} examples) ---")
            print_metrics(metrics)

    return all_metrics


def analyze_by_label(results, mode_name):
    """Break down metrics by true label."""
    print(f"\n{'='*60}")
    print(f"Per-Label Analysis: {mode_name}")
    print(f"{'='*60}")

    label_names = {0: "SUPPORTS", 1: "REFUTES"}
    all_metrics = {}

    for label_int, label_name in label_names.items():
        label_results = [r for r in results if r["true_label"] == label_int]
        if label_results:
            metrics = compute_metrics(label_results, label=f"{mode_name} - {label_name}")
            all_metrics[label_name] = metrics
            print(f"\n--- {label_name} ({len(label_results)} examples) ---")
            print_metrics(metrics)

    return all_metrics


def plot_frequency_comparison(baseline_freq_metrics, finetuned_freq_metrics):
    """Create bar charts comparing baseline vs finetuned across frequency buckets."""
    if not HAS_MATPLOTLIB:
        return

    buckets = ["low", "medium", "high"]
    x = np.arange(len(buckets))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Chart 1: Accuracy
    ax = axes[0, 0]
    b_vals = [baseline_freq_metrics.get(b, {}).get("accuracy", 0) for b in buckets]
    f_vals = [finetuned_freq_metrics.get(b, {}).get("accuracy", 0) for b in buckets]
    ax.bar(x - width/2, b_vals, width, label="Baseline (zero-shot)", color="#4ECDC4")
    ax.bar(x + width/2, f_vals, width, label="Fine-tuned", color="#FF6B6B")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Entity Frequency")
    ax.set_xticks(x)
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.legend()
    ax.set_ylim(0, 1)

    # Chart 2: High-confidence error rate
    ax = axes[0, 1]
    b_vals = [baseline_freq_metrics.get(b, {}).get("high_conf_error_rate", 0) for b in buckets]
    f_vals = [finetuned_freq_metrics.get(b, {}).get("high_conf_error_rate", 0) for b in buckets]
    ax.bar(x - width/2, b_vals, width, label="Baseline (zero-shot)", color="#4ECDC4")
    ax.bar(x + width/2, f_vals, width, label="Fine-tuned", color="#FF6B6B")
    ax.set_ylabel("High-Confidence Error Rate")
    ax.set_title("High-Confidence Errors by Entity Frequency")
    ax.set_xticks(x)
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.legend()
    ax.set_ylim(0, max(max(b_vals or [0.1]), max(f_vals or [0.1])) * 1.3)

    # Chart 3: ECE
    ax = axes[1, 0]
    b_vals = [baseline_freq_metrics.get(b, {}).get("ece", 0) for b in buckets]
    f_vals = [finetuned_freq_metrics.get(b, {}).get("ece", 0) for b in buckets]
    ax.bar(x - width/2, b_vals, width, label="Baseline (zero-shot)", color="#4ECDC4")
    ax.bar(x + width/2, f_vals, width, label="Fine-tuned", color="#FF6B6B")
    ax.set_ylabel("ECE")
    ax.set_title("Expected Calibration Error by Entity Frequency")
    ax.set_xticks(x)
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.legend()

    # Chart 4: Avg confidence (correct vs wrong)
    ax = axes[1, 1]
    b_correct = [baseline_freq_metrics.get(b, {}).get("avg_conf_correct", 0) for b in buckets]
    b_wrong = [baseline_freq_metrics.get(b, {}).get("avg_conf_wrong", 0) for b in buckets]
    f_correct = [finetuned_freq_metrics.get(b, {}).get("avg_conf_correct", 0) for b in buckets]
    f_wrong = [finetuned_freq_metrics.get(b, {}).get("avg_conf_wrong", 0) for b in buckets]
    w = 0.2
    ax.bar(x - 1.5*w, b_correct, w, label="Baseline correct", color="#4ECDC4")
    ax.bar(x - 0.5*w, b_wrong, w, label="Baseline wrong", color="#4ECDC4", alpha=0.5)
    ax.bar(x + 0.5*w, f_correct, w, label="Fine-tuned correct", color="#FF6B6B")
    ax.bar(x + 1.5*w, f_wrong, w, label="Fine-tuned wrong", color="#FF6B6B", alpha=0.5)
    ax.set_ylabel("Avg Confidence")
    ax.set_title("Confidence: Correct vs Wrong Predictions")
    ax.set_xticks(x)
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    chart_path = f"{config.OUTPUT_DIR}/frequency_analysis.png"
    plt.savefig(chart_path, dpi=150)
    print(f"\n📊 Frequency analysis chart saved to {chart_path}")
    plt.close()


def plot_calibration_curve(results, mode_name):
    """Plot reliability diagram (calibration curve)."""
    if not HAS_MATPLOTLIB:
        return

    confidences = np.array([r["confidence"] for r in results])
    correct = np.array([1 if r["predicted_idx"] == r["true_label"] else 0 for r in results])

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(int(mask.sum()))
        else:
            bin_accs.append(0)
            bin_confs.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_counts.append(0)

    ece = compute_ece(results)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, color="#FF6B6B", label=f"{mode_name} (ECE={ece:.4f})")
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Actual Accuracy")
    ax.set_title(f"Calibration Curve — {mode_name}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    chart_path = f"{config.OUTPUT_DIR}/calibration_{mode_name.lower().replace(' ', '_')}.png"
    plt.savefig(chart_path, dpi=150)
    print(f"📊 Calibration chart saved to {chart_path}")
    plt.close()


def main(baseline_only=False):
    """Run full analysis."""
    print("=" * 60)
    print("FEVER Hallucination & Calibration Analysis")
    print("Binary Classification: SUPPORTS vs REFUTES")
    print("=" * 60)

    baseline_results = load_results("baseline")
    finetuned_results = load_results("finetuned")

    if not baseline_results and not finetuned_results:
        print("❌ No results found. Run evaluate.py first.")
        print("   python evaluate.py --mode baseline")
        print("   python evaluate.py --mode finetuned")
        return

    all_analysis = {}

    # Analyze baseline
    if baseline_results:
        print(f"\n{'#'*60}")
        print(f"# BASELINE (Zero-Shot)")
        print(f"{'#'*60}")
        overall = compute_metrics(baseline_results, "Baseline - Overall")
        print("\n--- OVERALL ---")
        print_metrics(overall)
        all_analysis["baseline_overall"] = overall

        freq_metrics = analyze_by_frequency(baseline_results, "Baseline")
        all_analysis["baseline_frequency"] = freq_metrics

        label_metrics = analyze_by_label(baseline_results, "Baseline")
        all_analysis["baseline_labels"] = label_metrics

        if HAS_MATPLOTLIB:
            plot_calibration_curve(baseline_results, "Baseline")

    # Analyze fine-tuned
    if finetuned_results and not baseline_only:
        print(f"\n{'#'*60}")
        print(f"# FINE-TUNED")
        print(f"{'#'*60}")
        overall = compute_metrics(finetuned_results, "Fine-tuned - Overall")
        print("\n--- OVERALL ---")
        print_metrics(overall)
        all_analysis["finetuned_overall"] = overall

        freq_metrics = analyze_by_frequency(finetuned_results, "Fine-tuned")
        all_analysis["finetuned_frequency"] = freq_metrics

        label_metrics = analyze_by_label(finetuned_results, "Fine-tuned")
        all_analysis["finetuned_labels"] = label_metrics

        if HAS_MATPLOTLIB:
            plot_calibration_curve(finetuned_results, "Fine-tuned")

    # Comparison
    if baseline_results and finetuned_results and not baseline_only:
        print(f"\n{'#'*60}")
        print(f"# COMPARISON: Baseline vs Fine-tuned")
        print(f"{'#'*60}")

        b_overall = all_analysis.get("baseline_overall", {})
        f_overall = all_analysis.get("finetuned_overall", {})

        print(f"\n  {'Metric':<30} {'Baseline':>10} {'Fine-tuned':>10} {'Delta':>10}")
        print(f"  {'-'*62}")
        for key in ["accuracy", "avg_confidence", "high_conf_error_rate", "ece", "overconfidence_gap"]:
            b_val = b_overall.get(key, 0)
            f_val = f_overall.get(key, 0)
            delta = f_val - b_val
            sign = "+" if delta > 0 else ""
            print(f"  {key:<30} {b_val:>10.4f} {f_val:>10.4f} {sign}{delta:>9.4f}")

        if HAS_MATPLOTLIB:
            b_freq = all_analysis.get("baseline_frequency", {})
            f_freq = all_analysis.get("finetuned_frequency", {})
            plot_frequency_comparison(b_freq, f_freq)

    # Save full analysis
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(all_analysis, default=convert))
    analysis_path = f"{config.OUTPUT_DIR}/full_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n✅ Full analysis saved to {analysis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument("--baseline_only", action="store_true", help="Only analyze baseline results")
    args = parser.parse_args()
    main(baseline_only=args.baseline_only)
