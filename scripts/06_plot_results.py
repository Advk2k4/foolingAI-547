"""
06_plot_results.py - foolingAI-547
Read experiment_log.csv and produce publication-ready figures:
  Fig 1 — Misclassification rate vs magnitude (random vs targeted, 1-pt & 2-pt)
  Fig 2 — SNR vs magnitude (imperceptibility confirmation)
  Fig 3 — Heatmap: attack_type × magnitude × n_points

Run from project root: python scripts/06_plot_results.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "experiment_log.csv"
FIGURES_DIR = RESULTS_DIR / "figures"

PERTURBATION_PCTS = [0.5, 1.0, 2.0, 3.0]   # x-axis knob (% of peak-to-peak)
SNR_THRESHOLD_DB = 20.0

COLORS = {
    ("random",   1): "#4C72B0",
    ("random",   2): "#9EC8F2",
    ("targeted", 1): "#DD4444",
    ("targeted", 2): "#F4A0A0",
}
MARKERS = {
    ("random",   1): "o",
    ("random",   2): "s",
    ("targeted", 1): "^",
    ("targeted", 2): "D",
}
LINE_STYLES = {
    ("random",   1): "-",
    ("random",   2): "--",
    ("targeted", 1): "-",
    ("targeted", 2): "--",
}


def load_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run 05_run_experiments.py first.")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded {len(df):,} rows from {CSV_PATH}")
    return df


def fig1_misclassification_rate(df: pd.DataFrame):
    """Line plot: misclassification rate vs perturbation percentage."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for attack in ("random", "targeted"):
        for npts in (1, 2):
            subset = df[(df["attack_type"] == attack) & (df["n_points"] == npts)]
            rates = (
                subset.groupby("perturbation_pct")["misclassified"]
                .mean()
                .reindex(PERTURBATION_PCTS) * 100
            )
            key = (attack, npts)
            label = f"{attack.capitalize()} ({npts}-pt)"
            ax.plot(
                PERTURBATION_PCTS,
                rates.values,
                color=COLORS[key],
                linestyle=LINE_STYLES[key],
                marker=MARKERS[key],
                markersize=7,
                linewidth=2,
                label=label,
            )

    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Perturbation (% of signal peak-to-peak)", fontsize=12)
    ax.set_ylabel("Misclassification Rate (%)", fontsize=12)
    ax.set_title("Adversarial Attack Effectiveness\nvs Perturbation Magnitude", fontsize=13)
    ax.set_xticks(PERTURBATION_PCTS)
    ax.set_xticklabels([f"{p}%" for p in PERTURBATION_PCTS])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / "fig1_misclassification_rate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


def fig2_snr(df: pd.DataFrame):
    """Line plot: average SNR vs perturbation % — confirms imperceptibility."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for attack in ("random", "targeted"):
        subset = df[df["attack_type"] == attack]
        snrs = (
            subset.groupby("perturbation_pct")["snr_db"]
            .mean()
            .reindex(PERTURBATION_PCTS)
        )
        key = (attack, 1)
        ax.plot(
            PERTURBATION_PCTS,
            snrs.values,
            color=COLORS[key],
            linestyle=LINE_STYLES[key],
            marker=MARKERS[key],
            markersize=7,
            linewidth=2,
            label=attack.capitalize(),
        )

    ax.axhline(
        y=SNR_THRESHOLD_DB,
        color="green",
        linewidth=1.5,
        linestyle="--",
        label=f"Imperceptibility threshold ({SNR_THRESHOLD_DB} dB)",
    )
    ax.set_xlabel("Perturbation (% of signal peak-to-peak)", fontsize=12)
    ax.set_ylabel("Average SNR (dB)", fontsize=12)
    ax.set_title("Signal-to-Noise Ratio vs Perturbation Magnitude\n(above threshold = imperceptible)", fontsize=13)
    ax.set_xticks(PERTURBATION_PCTS)
    ax.set_xticklabels([f"{p}%" for p in PERTURBATION_PCTS])
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / "fig2_snr.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


def fig3_heatmap(df: pd.DataFrame):
    """Heatmap: rows = (attack, n_points), cols = perturbation %."""
    groups = [
        ("random",   1, "Random 1-pt"),
        ("random",   2, "Random 2-pt"),
        ("targeted", 1, "Targeted 1-pt"),
        ("targeted", 2, "Targeted 2-pt"),
    ]

    matrix = np.zeros((len(groups), len(PERTURBATION_PCTS)))
    row_labels = []

    for i, (attack, npts, label) in enumerate(groups):
        subset = df[(df["attack_type"] == attack) & (df["n_points"] == npts)]
        rates = (
            subset.groupby("perturbation_pct")["misclassified"]
            .mean()
            .reindex(PERTURBATION_PCTS)
            .fillna(0) * 100
        )
        matrix[i] = rates.values
        row_labels.append(label)

    # Scale color range to data max so low values still show contrast
    vmax = max(matrix.max(), 5.0)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)

    ax.set_xticks(range(len(PERTURBATION_PCTS)))
    ax.set_xticklabels([f"{p}%" for p in PERTURBATION_PCTS], fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xlabel("Perturbation (% of peak-to-peak)", fontsize=12)
    ax.set_title("Misclassification Rate Heatmap (%)", fontsize=13)

    for i in range(len(groups)):
        for j in range(len(PERTURBATION_PCTS)):
            val = matrix[i, j]
            text_color = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=10, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Misclassification Rate (%)")
    fig.tight_layout()

    out = FIGURES_DIR / "fig3_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


def print_summary_table(df: pd.DataFrame):
    """Print console summary matching professor's 'knob' view."""
    print("\n" + "=" * 65)
    print("SUMMARY: Misclassification Rate (%) — the professor's knob")
    print("=" * 65)

    for npts in (1, 2):
        print(f"\n  n_points = {npts}")
        print(f"  {'Pert %':<12} {'Random':<12} {'Targeted':<12} {'Δ (targeted−random)'}")
        print("  " + "─" * 52)
        for pct in PERTURBATION_PCTS:
            rand_rate = df[
                (df["attack_type"] == "random") &
                (df["n_points"] == npts) &
                (df["perturbation_pct"] == pct)
            ]["misclassified"].mean() * 100

            tgt_rate = df[
                (df["attack_type"] == "targeted") &
                (df["n_points"] == npts) &
                (df["perturbation_pct"] == pct)
            ]["misclassified"].mean() * 100

            delta = tgt_rate - rand_rate
            sign = "+" if delta >= 0 else ""
            print(f"  {pct}%{'':<9} {rand_rate:<12.1f} {tgt_rate:<12.1f} {sign}{delta:.1f}")

    print("=" * 65)


def run():
    print("=" * 60)
    print("foolingAI-547  |  Script 06: Plot Results")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading experiment log...")
    df = load_data()

    print("\nGenerating figures...")
    fig1_misclassification_rate(df)
    fig2_snr(df)
    fig3_heatmap(df)

    print_summary_table(df)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("Script 06 complete.")


if __name__ == "__main__":
    run()
