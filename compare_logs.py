"""
Compare two train.log files (withoutMIL vs withMIL) per fold and overall averages.
Outputs a CSV and PNG into a timestamped comparison folder.

Usage:
    python compare_logs.py
"""

import re
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────────────────────
LOG1_PATH = (
    "/shared/anastasio-s3/jyue/Large-Scale-Medical/Downstream/"
    "dual_branch_3d_cnn/outputs/"
    "155_dual_branch_recurrence_20260224_164611/train.log"
)
LOG2_PATH = (
    "/shared/anastasio-s3/jyue/Large-Scale-Medical/Downstream/"
    "dual_branch_3d_cnn/outputs/"
    "MIL_120_ual_branch_recurrence_20260223_213324/train.log"
)
LABEL1 = "withoutMIL"
LABEL2 = "withMIL"

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"comparison_{LABEL1}_vs_{LABEL2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)

# Metrics shown in the PNG
PLOT_METRICS = ["auc", "sensitivity", "specificity", "ppv", "npv", "accuracy"]
# All metrics written to CSV
ALL_METRICS  = ["auc", "threshold", "sensitivity", "specificity",
                "ppv", "npv", "accuracy", "tp", "fn", "tn", "fp"]
# ──────────────────────────────────────────────────────────────────────────────


def parse_log(path: str) -> dict:
    """
    Extract the last recorded Val metrics block per fold from a train.log.
    Returns {fold_number: {metric_name: value}}.
    """
    with open(path, "r") as f:
        content = f.read()

    block_re = re.compile(
        r"\[Fold\s+(\d+)\]\s+Val metrics[^\n]*\n"
        r"((?:[ \t]+\w[\w ]*:[ \t]*[\d.eE+\-]+\n?)+)",
        re.MULTILINE,
    )

    fold_metrics = {}
    for m in block_re.finditer(content):
        fold = int(m.group(1))
        metrics = {}
        for line in m.group(2).splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip().lower().replace(" ", "_")
            try:
                metrics[key] = float(val.strip())
            except ValueError:
                pass
        fold_metrics[fold] = metrics  # last occurrence wins

    if not fold_metrics:
        print(f"  WARNING: no Val metrics blocks found in {path}")
    return fold_metrics


def build_dataframe(metrics1, metrics2):
    """
    Build a tidy long-format DataFrame:
      fold | metric | withoutMIL | withMIL | delta
    followed by mean and std summary rows.
    """
    folds = sorted(set(metrics1) | set(metrics2))
    rows = []
    for fold in folds:
        m1 = metrics1.get(fold, {})
        m2 = metrics2.get(fold, {})
        for metric in ALL_METRICS:
            v1 = m1.get(metric, float("nan"))
            v2 = m2.get(metric, float("nan"))
            rows.append({
                "fold":       fold,
                "metric":     metric,
                LABEL1:       v1,
                LABEL2:       v2,
                "delta":      round(v2 - v1, 8),
            })

    df = pd.DataFrame(rows)

    # ── Summary rows ───────────────────────────────────────────────────────────
    summary_rows = []
    for metric in ALL_METRICS:
        sub = df[df["metric"] == metric]
        v1_arr = sub[LABEL1].dropna().values
        v2_arr = sub[LABEL2].dropna().values
        n = min(len(v1_arr), len(v2_arr))
        summary_rows.append({
            "fold":   "mean",
            "metric": metric,
            LABEL1:   np.mean(v1_arr) if len(v1_arr) else float("nan"),
            LABEL2:   np.mean(v2_arr) if len(v2_arr) else float("nan"),
            "delta":  np.mean(v2_arr[:n] - v1_arr[:n]) if n else float("nan"),
        })
        summary_rows.append({
            "fold":   "std",
            "metric": metric,
            LABEL1:   np.std(v1_arr, ddof=1) if len(v1_arr) > 1 else float("nan"),
            LABEL2:   np.std(v2_arr, ddof=1) if len(v2_arr) > 1 else float("nan"),
            "delta":  float("nan"),
        })

    return pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)


def plot_comparison(metrics1, metrics2, save_path):
    folds = sorted(set(metrics1) | set(metrics2))
    n = len(PLOT_METRICS)

    colors = {LABEL1: "#4C72B0", LABEL2: "#DD8452"}

    fig = plt.figure(figsize=(4.5 * n, 9))
    gs  = gridspec.GridSpec(2, n, figure=fig, hspace=0.50, wspace=0.38)

    for col, metric in enumerate(PLOT_METRICS):

        # ── Top: grouped bar (mean ± std) ──────────────────────────────────
        ax_bar = fig.add_subplot(gs[0, col])
        means, stds, bar_labels = [], [], [LABEL1, LABEL2]
        for mdict_map in (metrics1, metrics2):
            vals = [mdict_map.get(f, {}).get(metric, float("nan")) for f in folds]
            vals = [v for v in vals if not np.isnan(v)]
            means.append(np.mean(vals) if vals else float("nan"))
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

        x = np.arange(2)
        bar_objs = ax_bar.bar(
            x, means, yerr=stds, capsize=5,
            color=[colors[l] for l in bar_labels],
            error_kw={"elinewidth": 1.5, "ecolor": "#333333"},
            width=0.5,
        )
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(bar_labels, fontsize=8.5)
        ax_bar.set_title(metric, fontsize=11, fontweight="bold", pad=6)
        ax_bar.set_ylabel("mean ± std", fontsize=8)
        ax_bar.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax_bar.set_axisbelow(True)

        y_top = max(m + s for m, s in zip(means, stds) if not np.isnan(m))
        for bar, mean_val in zip(bar_objs, means):
            if not np.isnan(mean_val):
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean_val + max(stds) * 0.08 + 0.005,
                    f"{mean_val:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

        # ── Bottom: per-fold line ───────────────────────────────────────────
        ax_line = fig.add_subplot(gs[1, col])
        for lbl, mdict_map in [(LABEL1, metrics1), (LABEL2, metrics2)]:
            vals = [mdict_map.get(f, {}).get(metric, float("nan")) for f in folds]
            ax_line.plot(
                folds, vals,
                marker="o", label=lbl,
                color=colors[lbl], linewidth=1.8, markersize=5,
            )
        ax_line.set_xticks(folds)
        ax_line.set_xlabel("Fold", fontsize=8)
        ax_line.set_ylabel(metric, fontsize=8)
        ax_line.legend(fontsize=7, loc="best")
        ax_line.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax_line.set_axisbelow(True)

    fig.suptitle(
        f"Validation Metrics: {LABEL1}  vs  {LABEL2}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PNG  → {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output dir: {OUTPUT_DIR}\n")

    print(f"[1] Parsing {LABEL1}:\n    {LOG1_PATH}")
    m1 = parse_log(LOG1_PATH)
    print(f"    Folds found: {sorted(m1)}")

    print(f"[2] Parsing {LABEL2}:\n    {LOG2_PATH}")
    m2 = parse_log(LOG2_PATH)
    print(f"    Folds found: {sorted(m2)}")

    # ── CSV ────────────────────────────────────────────────────────────────────
    df = build_dataframe(m1, m2)
    csv_path = os.path.join(OUTPUT_DIR, "comparison.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n  CSV  → {csv_path}")

    # Print mean table to terminal
    print("\n─── Mean values ─────────────────────────────────────────────────────")
    mean_df = (
        df[df["fold"] == "mean"][["metric", LABEL1, LABEL2, "delta"]]
        .set_index("metric")
    )
    print(mean_df.to_string(float_format=lambda x: f"{x:+.4f}" if not np.isnan(x) else "  nan"))

    # ── PNG ────────────────────────────────────────────────────────────────────
    png_path = os.path.join(OUTPUT_DIR, "comparison.png")
    plot_comparison(m1, m2, png_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
