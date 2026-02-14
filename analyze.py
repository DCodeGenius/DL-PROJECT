"""
Analysis script for FEVER hallucination study.
Produces a 2x3 comparison matrix:
    (Baseline / Fine-tuned) x (Test-High / Test-Low / Test-Mixed)
Computes accuracy, ECE, high-confidence error rate, calibration plots.

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
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Skipping charts.")


# ─── Metrics ──────────────────────────────────────────────────────────

def compute_ece(results, n_bins=10):
    """Expected Calibration Error."""
    if not results:
        return 0.0
    confs = np.array([r["confidence"] for r in results])
    correct = np.array([1 if r["predicted_idx"] == r["true_label"] else 0 for r in results])
    total = len(results)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs >= edges[i]) & (confs < edges[i + 1])
        n = mask.sum()
        if n > 0:
            ece += (n / total) * abs(correct[mask].mean() - confs[mask].mean())
    return ece


def compute_metrics(results):
    """Compute all metrics for a result set."""
    if not results:
        return {}
    total = len(results)
    correct = sum(1 for r in results if r["predicted_idx"] == r["true_label"])
    accuracy = correct / total
    confs = [r["confidence"] for r in results]
    avg_conf = np.mean(confs)
    conf_correct = [r["confidence"] for r in results if r["predicted_idx"] == r["true_label"]]
    conf_wrong = [r["confidence"] for r in results if r["predicted_idx"] != r["true_label"]]
    hce = sum(1 for r in results if r["predicted_idx"] != r["true_label"] and r["confidence"] > 0.7)

    return {
        "n": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_confidence": avg_conf,
        "avg_conf_correct": np.mean(conf_correct) if conf_correct else 0,
        "avg_conf_wrong": np.mean(conf_wrong) if conf_wrong else 0,
        "hce_count": hce,
        "hce_rate": hce / total,
        "ece": compute_ece(results),
        "overconf_gap": avg_conf - accuracy,
    }


def print_metrics(title, m):
    if not m:
        return
    print(f"\n  --- {title} (n={m['n']}) ---")
    print(f"  {'Accuracy':<26} {m['accuracy']:>8.4f}")
    print(f"  {'Avg confidence':<26} {m['avg_confidence']:>8.4f}")
    print(f"  {'Avg conf (correct)':<26} {m['avg_conf_correct']:>8.4f}")
    print(f"  {'Avg conf (wrong)':<26} {m['avg_conf_wrong']:>8.4f}")
    print(f"  {'High-conf error rate':<26} {m['hce_rate']:>8.4f}  ({m['hce_count']} errors)")
    print(f"  {'ECE':<26} {m['ece']:>8.4f}")
    print(f"  {'Overconfidence gap':<26} {m['overconf_gap']:>8.4f}")


# ─── Loading ──────────────────────────────────────────────────────────

def load_results(mode, test_set):
    """Load eval_results_{mode}_{test_set}.json."""
    path = f"{config.OUTPUT_DIR}/eval_results_{mode}_{test_set}.json"
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# ─── Charts ───────────────────────────────────────────────────────────

def plot_2x3_bars(matrix, metric_key, metric_label, filename, ylim=None):
    """Grouped bar chart: 2 models x 3 test sets."""
    if not HAS_MPL:
        return

    models = ["Baseline", "Fine-tuned"]
    test_sets = ["Test-High", "Test-Low", "Test-Mixed"]
    colors = {"Baseline": "#4ECDC4", "Fine-tuned": "#FF6B6B"}

    x = np.arange(len(test_sets))
    width = 0.3

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, model in enumerate(models):
        vals = [matrix.get(model, {}).get(ts, {}).get(metric_key, 0) for ts in test_sets]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model, color=colors[model])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} by Test Set")
    ax.set_xticks(x)
    ax.set_xticklabels(test_sets)
    ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = f"{config.OUTPUT_DIR}/{filename}"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Chart saved: {path}")


def plot_calibration(results, title, filename):
    """Reliability diagram for one model-testset combination."""
    if not HAS_MPL:
        return
    confs = np.array([r["confidence"] for r in results])
    correct = np.array([1 if r["predicted_idx"] == r["true_label"] else 0 for r in results])
    ece = compute_ece(results)
    n_bins = 10
    edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs = [], []
    for i in range(n_bins):
        mask = (confs >= edges[i]) & (confs < edges[i + 1])
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confs[mask].mean())
        else:
            bin_accs.append(0)
            bin_confs.append((edges[i] + edges[i + 1]) / 2)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, color="#FF6B6B",
           label=f"ECE = {ece:.4f}")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Calibration — {title}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    path = f"{config.OUTPUT_DIR}/{filename}"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Chart saved: {path}")


def plot_combined_calibration(all_results, filename):
    """Overlay calibration curves for all model+testset combos."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")

    palette = {
        "Baseline": {"Test-High": "#2ecc71", "Test-Low": "#27ae60", "Test-Mixed": "#1abc9c"},
        "Fine-tuned": {"Test-High": "#e74c3c", "Test-Low": "#c0392b", "Test-Mixed": "#e67e22"},
    }
    markers = {"Test-High": "o", "Test-Low": "s", "Test-Mixed": "D"}

    n_bins = 10
    edges = np.linspace(0, 1, n_bins + 1)

    for (model, ts), results in all_results.items():
        if not results:
            continue
        confs = np.array([r["confidence"] for r in results])
        correct = np.array([1 if r["predicted_idx"] == r["true_label"] else 0 for r in results])
        ece = compute_ece(results)

        xs, ys = [], []
        for i in range(n_bins):
            mask = (confs >= edges[i]) & (confs < edges[i + 1])
            if mask.sum() > 0:
                xs.append(confs[mask].mean())
                ys.append(correct[mask].mean())

        color = palette.get(model, {}).get(ts, "#888")
        marker = markers.get(ts, "o")
        ax.plot(xs, ys, f"{marker}-", color=color, label=f"{model} / {ts} (ECE={ece:.3f})")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Calibration Curves — All Conditions")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = f"{config.OUTPUT_DIR}/{filename}"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Chart saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FEVER Hallucination Study — Analysis")
    print("2x3: (Baseline / Fine-tuned) x (Test-High / Test-Low / Test-Mixed)")
    print("=" * 60)

    modes = ["baseline", "finetuned"]
    mode_labels = {"baseline": "Baseline", "finetuned": "Fine-tuned"}
    test_sets = ["high", "low", "mixed"]
    ts_labels = {"high": "Test-High", "low": "Test-Low", "mixed": "Test-Mixed"}

    # Load all results
    all_data = {}
    for mode in modes:
        for ts in test_sets:
            results = load_results(mode, ts)
            if results:
                key = (mode_labels[mode], ts_labels[ts])
                all_data[key] = results
                print(f"  Loaded {mode}_{ts}: {len(results)} examples")

    if not all_data:
        print("ERROR: No results found. Run evaluate.py first.")
        return

    # Build 2x3 matrix
    matrix = {}
    all_analysis = {}

    for mode in modes:
        ml = mode_labels[mode]
        matrix[ml] = {}
        print(f"\n{'#'*60}")
        print(f"# {ml}")
        print(f"{'#'*60}")

        # Overall across all test sets for this mode
        all_mode_results = []
        for ts in test_sets:
            key = (ml, ts_labels[ts])
            results = all_data.get(key)
            if results:
                all_mode_results.extend(results)
                m = compute_metrics(results)
                matrix[ml][ts_labels[ts]] = m
                print_metrics(ts_labels[ts], m)
                all_analysis[f"{ml}_{ts_labels[ts]}"] = m

        if all_mode_results:
            overall = compute_metrics(all_mode_results)
            print_metrics(f"{ml} OVERALL (all test sets)", overall)
            all_analysis[f"{ml}_overall"] = overall

        # Per-label breakdown
        for label_int, label_name in [(0, "SUPPORTS"), (1, "REFUTES")]:
            lr = [r for r in all_mode_results if r["true_label"] == label_int]
            if lr:
                m = compute_metrics(lr)
                print_metrics(f"{ml} {label_name}", m)
                all_analysis[f"{ml}_{label_name}"] = m

    # ─── Comparison table ─────────────────────────────────────────
    if len([m for m in modes if any((mode_labels[m], ts_labels[ts]) in all_data for ts in test_sets)]) >= 2:
        print(f"\n{'#'*60}")
        print(f"# 2x3 COMPARISON MATRIX")
        print(f"{'#'*60}")

        for metric_key, metric_label in [
            ("accuracy", "Accuracy"),
            ("hce_rate", "High-Conf Error Rate"),
            ("ece", "ECE"),
            ("overconf_gap", "Overconfidence Gap"),
            ("avg_confidence", "Avg Confidence"),
        ]:
            print(f"\n  {metric_label}:")
            header = f"  {'Model':<16}"
            for ts in test_sets:
                header += f" {ts_labels[ts]:>12}"
            print(header)
            print(f"  {'-'*54}")
            for mode in modes:
                ml = mode_labels[mode]
                row = f"  {ml:<16}"
                for ts in test_sets:
                    val = matrix.get(ml, {}).get(ts_labels[ts], {}).get(metric_key, 0)
                    row += f" {val:>12.4f}"
                print(row)

    # ─── Charts ───────────────────────────────────────────────────
    if HAS_MPL:
        print(f"\nGenerating charts...")
        plot_2x3_bars(matrix, "accuracy", "Accuracy", "chart_accuracy.png", ylim=(0, 1))
        plot_2x3_bars(matrix, "hce_rate", "High-Confidence Error Rate", "chart_hce_rate.png")
        plot_2x3_bars(matrix, "ece", "ECE", "chart_ece.png")
        plot_2x3_bars(matrix, "overconf_gap", "Overconfidence Gap", "chart_overconf_gap.png")
        plot_2x3_bars(matrix, "avg_confidence", "Average Confidence", "chart_avg_confidence.png")

        # Individual calibration curves
        for (model, ts), results in all_data.items():
            safe = f"{model}_{ts}".lower().replace(" ", "_").replace("-", "_")
            plot_calibration(results, f"{model} / {ts}", f"calibration_{safe}.png")

        # Combined overlay
        plot_combined_calibration(all_data, "calibration_all.png")

    # ─── Save ─────────────────────────────────────────────────────
    def convert(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    # Flatten matrix for JSON
    all_analysis["matrix"] = {}
    for ml in matrix:
        all_analysis["matrix"][ml] = {}
        for ts in matrix[ml]:
            all_analysis["matrix"][ml][ts] = {
                k: convert(v) for k, v in matrix[ml][ts].items()
            }

    out_path = f"{config.OUTPUT_DIR}/full_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_analysis, f, indent=2, default=convert)
    print(f"\nFull analysis saved to {out_path}")


if __name__ == "__main__":
    main()
