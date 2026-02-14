"""
Analysis script for FEVER hallucination study.
Produces a 3×2 comparison matrix (Baseline / High-trained / Low-trained) × (Eval-H / Eval-L).
Computes accuracy, ECE, high-confidence error rate, and calibration plots.

Usage:
    python analyze.py
"""

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
    print("⚠️  matplotlib not installed. Skipping charts.")


# ─── Metric computation ───────────────────────────────────────────────

def compute_ece(results, n_bins=10):
    """Expected Calibration Error."""
    if not results:
        return 0.0
    confs = np.array([r["confidence"] for r in results])
    correct = np.array([1 if r["predicted_idx"] == r["true_label"] else 0 for r in results])
    total = len(results)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs >= bin_edges[i]) & (confs < bin_edges[i + 1])
        n = mask.sum()
        if n > 0:
            ece += (n / total) * abs(correct[mask].mean() - confs[mask].mean())
    return ece


def compute_metrics(results):
    """Compute all metrics for a set of results."""
    if not results:
        return {}
    total = len(results)
    correct = sum(1 for r in results if r["predicted_idx"] == r["true_label"])
    accuracy = correct / total

    confs = [r["confidence"] for r in results]
    avg_conf = np.mean(confs)

    correct_confs = [r["confidence"] for r in results if r["predicted_idx"] == r["true_label"]]
    wrong_confs = [r["confidence"] for r in results if r["predicted_idx"] != r["true_label"]]

    # High-confidence error: wrong AND confidence > 0.7
    hce_count = sum(1 for r in results if r["predicted_idx"] != r["true_label"] and r["confidence"] > 0.7)
    hce_rate = hce_count / total

    ece = compute_ece(results)
    overconf_gap = avg_conf - accuracy

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_confidence": avg_conf,
        "avg_conf_correct": np.mean(correct_confs) if correct_confs else 0,
        "avg_conf_wrong": np.mean(wrong_confs) if wrong_confs else 0,
        "hce_count": hce_count,
        "hce_rate": hce_rate,
        "ece": ece,
        "overconf_gap": overconf_gap,
    }


def print_metrics(title, m):
    """Pretty-print metrics."""
    if not m:
        return
    print(f"\n  --- {title} ---")
    print(f"  {'N':<28} {m['total']:>8}")
    print(f"  {'Accuracy':<28} {m['accuracy']:>8.4f}")
    print(f"  {'Avg confidence':<28} {m['avg_confidence']:>8.4f}")
    print(f"  {'Avg conf (correct)':<28} {m['avg_conf_correct']:>8.4f}")
    print(f"  {'Avg conf (wrong)':<28} {m['avg_conf_wrong']:>8.4f}")
    print(f"  {'High-conf error rate':<28} {m['hce_rate']:>8.4f}")
    print(f"  {'ECE':<28} {m['ece']:>8.4f}")
    print(f"  {'Overconfidence gap':<28} {m['overconf_gap']:>8.4f}")


# ─── Loading ──────────────────────────────────────────────────────────

def load_results(name):
    """Load eval_results_{name}.json if it exists."""
    path = f"{config.OUTPUT_DIR}/eval_results_{name}.json"
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# ─── Charts ───────────────────────────────────────────────────────────

def plot_3x2_bars(matrix, metric_key, metric_label, filename, ylim=None):
    """Bar chart for a metric across the 3×2 matrix."""
    if not HAS_MATPLOTLIB:
        return

    models = ["Baseline", "Trained-on-High", "Trained-on-Low"]
    eval_buckets = ["Eval-H", "Eval-L"]
    colors = {"Eval-H": "#FF6B6B", "Eval-L": "#4ECDC4"}

    x = np.arange(len(models))
    width = 0.3

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, eb in enumerate(eval_buckets):
        vals = [matrix.get(m, {}).get(eb, {}).get(metric_key, 0) for m in models]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=eb, color=colors[eb])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} — by Training Exposure × Eval Bucket")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = f"{config.OUTPUT_DIR}/{filename}"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Saved {path}")


def plot_calibration(results, title, filename):
    """Reliability diagram."""
    if not HAS_MATPLOTLIB:
        return
    confs = np.array([r["confidence"] for r in results])
    correct = np.array([1 if r["predicted_idx"] == r["true_label"] else 0 for r in results])

    n_bins = 10
    edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(n_bins):
        mask = (confs >= edges[i]) & (confs < edges[i + 1])
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confs[mask].mean())
            bin_counts.append(int(mask.sum()))
        else:
            bin_accs.append(0)
            bin_confs.append((edges[i] + edges[i + 1]) / 2)
            bin_counts.append(0)

    ece = compute_ece(results)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, color="#FF6B6B",
           label=f"{title} (ECE={ece:.4f})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Calibration — {title}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    path = f"{config.OUTPUT_DIR}/{filename}"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Saved {path}")


def plot_combined_calibration(all_results_dict, filename):
    """Overlay calibration curves for all models."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1)

    colors = {"Baseline": "#888888", "Trained-on-High": "#FF6B6B", "Trained-on-Low": "#4ECDC4"}
    n_bins = 10
    edges = np.linspace(0, 1, n_bins + 1)

    for name, results in all_results_dict.items():
        if not results:
            continue
        confs = np.array([r["confidence"] for r in results])
        correct = np.array([1 if r["predicted_idx"] == r["true_label"] else 0 for r in results])
        ece = compute_ece(results)

        bin_accs, bin_confs_list = [], []
        for i in range(n_bins):
            mask = (confs >= edges[i]) & (confs < edges[i + 1])
            if mask.sum() > 0:
                bin_accs.append(correct[mask].mean())
                bin_confs_list.append(confs[mask].mean())
            else:
                bin_accs.append(None)
                bin_confs_list.append(None)

        xs = [c for c, a in zip(bin_confs_list, bin_accs) if a is not None]
        ys = [a for a in bin_accs if a is not None]
        ax.plot(xs, ys, "o-", label=f"{name} (ECE={ece:.4f})", color=colors.get(name, "#000000"))

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Calibration Curves — All Models")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    path = f"{config.OUTPUT_DIR}/{filename}"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Saved {path}")


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FEVER Hallucination Study — Analysis")
    print("3×2 Matrix: (Baseline / High-trained / Low-trained) × (Eval-H / Eval-L)")
    print("=" * 60)

    # Load all result files
    baseline = load_results("baseline")
    finetuned_high = load_results("finetuned_high")
    finetuned_low = load_results("finetuned_low")

    models = {}
    if baseline:
        models["Baseline"] = baseline
        print(f"✅ Loaded baseline results: {len(baseline)} examples")
    if finetuned_high:
        models["Trained-on-High"] = finetuned_high
        print(f"✅ Loaded high-trained results: {len(finetuned_high)} examples")
    if finetuned_low:
        models["Trained-on-Low"] = finetuned_low
        print(f"✅ Loaded low-trained results: {len(finetuned_low)} examples")

    if not models:
        print("❌ No results found. Run evaluate.py first.")
        return

    # ─── Build 3×2 matrix ─────────────────────────────────────────
    matrix = {}
    all_analysis = {}

    for model_name, results in models.items():
        print(f"\n{'#'*60}")
        print(f"# {model_name}")
        print(f"{'#'*60}")

        # Overall
        overall = compute_metrics(results)
        print_metrics("OVERALL", overall)
        all_analysis[f"{model_name}_overall"] = overall

        # By eval bucket
        matrix[model_name] = {}
        for bucket, label in [("H", "Eval-H"), ("L", "Eval-L")]:
            bucket_results = [r for r in results if r["bucket"] == bucket]
            if bucket_results:
                m = compute_metrics(bucket_results)
                print_metrics(f"{label} ({len(bucket_results)} examples)", m)
                matrix[model_name][label] = m
                all_analysis[f"{model_name}_{label}"] = m

        # By true label
        for label_int, label_name in [(0, "SUPPORTS"), (1, "REFUTES")]:
            lr = [r for r in results if r["true_label"] == label_int]
            if lr:
                m = compute_metrics(lr)
                print_metrics(f"{label_name} ({len(lr)} examples)", m)
                all_analysis[f"{model_name}_{label_name}"] = m

    # ─── Comparison table ─────────────────────────────────────────
    if len(models) >= 2:
        print(f"\n{'#'*60}")
        print(f"# 3×2 COMPARISON MATRIX")
        print(f"{'#'*60}")

        for metric_key, metric_label in [
            ("accuracy", "Accuracy"),
            ("hce_rate", "High-Conf Error Rate"),
            ("ece", "ECE"),
            ("overconf_gap", "Overconfidence Gap"),
            ("avg_confidence", "Avg Confidence"),
        ]:
            print(f"\n  {metric_label}:")
            print(f"  {'Model':<22} {'Eval-H':>10} {'Eval-L':>10} {'Delta':>10}")
            print(f"  {'-'*54}")
            for model_name in models:
                h_val = matrix.get(model_name, {}).get("Eval-H", {}).get(metric_key, 0)
                l_val = matrix.get(model_name, {}).get("Eval-L", {}).get(metric_key, 0)
                delta = h_val - l_val
                sign = "+" if delta > 0 else ""
                print(f"  {model_name:<22} {h_val:>10.4f} {l_val:>10.4f} {sign}{delta:>9.4f}")

    # ─── Charts ───────────────────────────────────────────────────
    if HAS_MATPLOTLIB and len(models) >= 2:
        plot_3x2_bars(matrix, "accuracy", "Accuracy", "chart_accuracy.png", ylim=(0, 1))
        plot_3x2_bars(matrix, "hce_rate", "High-Confidence Error Rate", "chart_hce_rate.png")
        plot_3x2_bars(matrix, "ece", "ECE", "chart_ece.png")
        plot_3x2_bars(matrix, "overconf_gap", "Overconfidence Gap", "chart_overconf_gap.png")

        # Calibration curves
        for model_name, results in models.items():
            safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
            plot_calibration(results, model_name, f"calibration_{safe_name}.png")

        plot_combined_calibration(models, "calibration_all_models.png")

    # ─── Save ─────────────────────────────────────────────────────
    all_analysis["matrix"] = {}
    for mk in matrix:
        all_analysis["matrix"][mk] = {}
        for eb in matrix[mk]:
            all_analysis["matrix"][mk][eb] = {
                k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
                for k, v in matrix[mk][eb].items()
            }

    out_path = f"{config.OUTPUT_DIR}/full_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_analysis, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x)
    print(f"\n✅ Full analysis saved to {out_path}")


if __name__ == "__main__":
    main()
