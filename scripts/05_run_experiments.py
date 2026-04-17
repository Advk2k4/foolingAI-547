"""
05_run_experiments.py - foolingAI-547
Full experiment sweep: all test trials × magnitudes × attack types × n_points.
Writes results to results/experiment_log.csv incrementally.

Run from project root: python scripts/05_run_experiments.py
This script performs a comprehensive sweep of adversarial perturbation experiments
on the test set using both random and targeted attacks, varying the number of   points perturbed and the magnitude of perturbation. Results are saved incrementally
"""

import sys
import csv
import pickle
import importlib.util
import time
import numpy as np
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
RESULTS_DIR = Path("results")
OUTPUT_CSV = RESULTS_DIR / "experiment_log.csv"

MAGNITUDES_UV = [0.5, 1.0, 2.0, 5.0, 10.0]
ATTACK_TYPES = ["random", "targeted"]
N_POINTS_LIST = [1, 2]

CSV_COLUMNS = [
    "trial_id",
    "attack_type",
    "n_points",
    "magnitude_uv",
    "orig_pred",
    "pert_pred",
    "misclassified",
    "snr_db",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_perturbation_module():
    """Dynamically import PerturbationEngine and TargetedPerturbationEngine
    from 03_perturbation.py (same method script 04 uses)."""
    script_dir = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "perturbation_module", script_dir / "03_perturbation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_artefacts():
    """Load the trained model and test data saved by script 02."""
    model_path = RESULTS_DIR / "model.pkl"
    x_path = RESULTS_DIR / "X_test.npy"
    y_path = RESULTS_DIR / "y_test.npy"

    for p in (model_path, x_path, y_path):
        if not p.exists():
            print(f"ERROR: {p} not found. Run scripts 01 and 02 first.")
            sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_test = np.load(x_path)
    y_test = np.load(y_path)
    return model, X_test, y_test


def run_single_experiment(
    trial_id: int,
    trial: np.ndarray,
    model,
    attack_type: str,
    n_points: int,
    magnitude_uv: float,
    random_engine,
    targeted_engine,
) -> dict:
    """Run one experiment and return a result row.

    Args:
        trial_id:        Index of the trial in X_test.
        trial:           EEG array (n_channels, n_times).
        model:           Fitted sklearn pipeline.
        attack_type:     "random" or "targeted".
        n_points:        Number of sample points to perturb (1 or 2).
        magnitude_uv:    Perturbation magnitude in µV.
        random_engine:   PerturbationEngine instance.
        targeted_engine: TargetedPerturbationEngine instance.

    Returns:
        dict matching CSV_COLUMNS.
    """
    # Original prediction
    orig_pred = int(model.predict(trial[np.newaxis])[0])

    # Apply perturbation
    if attack_type == "random":
        indices = random_engine.select_random_samples(trial, num_points=n_points)
        perturbed = random_engine.apply_perturbation(trial, indices, magnitude_uv)
        metrics = random_engine.verify_imperceptibility(trial, perturbed)
    elif attack_type == "targeted":
        perturbed = targeted_engine.apply_targeted_perturbation(
            trial, model, magnitude_uv, num_points=n_points
        )
        metrics = targeted_engine.verify_imperceptibility(trial, perturbed)
    else:
        raise ValueError(f"Unknown attack_type: {attack_type}")

    # New prediction
    pert_pred = int(model.predict(perturbed[np.newaxis])[0])

    return {
        "trial_id": trial_id,
        "attack_type": attack_type,
        "n_points": n_points,
        "magnitude_uv": magnitude_uv,
        "orig_pred": orig_pred,
        "pert_pred": pert_pred,
        "misclassified": int(orig_pred != pert_pred),
        "snr_db": round(metrics["snr_db"], 2),
    }


# ── Main sweep ───────────────────────────────────────────────────────────────

def run():
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("foolingAI-547  |  Script 05: Full Experiment Sweep")
    print("=" * 60)

    # Load everything
    print("\nLoading model and test data...")
    model, X_test, y_test = load_artefacts()
    n_trials = len(X_test)
    print(f"  {n_trials} test trials loaded.")

    print("Loading perturbation engines...")
    pm = load_perturbation_module()
    random_engine = pm.PerturbationEngine(random_seed=RANDOM_SEED)
    targeted_engine = pm.TargetedPerturbationEngine(random_seed=RANDOM_SEED)

    # Calculate total experiments
    total = n_trials * len(ATTACK_TYPES) * len(N_POINTS_LIST) * len(MAGNITUDES_UV)
    print(f"\n  Experiment grid:")
    print(f"    Trials:       {n_trials}")
    print(f"    Attack types: {ATTACK_TYPES}")
    print(f"    N points:     {N_POINTS_LIST}")
    print(f"    Magnitudes:   {MAGNITUDES_UV} µV")
    print(f"    Total:        {total} experiments")

    # Open CSV and write incrementally (won't lose data on crash)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csvfile = open(OUTPUT_CSV, "w", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    completed = 0
    start_time = time.time()

    print(f"\n{'─' * 60}")
    print("Starting experiment sweep...")
    print(f"{'─' * 60}\n")

    for trial_id in range(n_trials):
        trial = X_test[trial_id]

        for attack_type in ATTACK_TYPES:
            for n_points in N_POINTS_LIST:
                for magnitude_uv in MAGNITUDES_UV:

                    row = run_single_experiment(
                        trial_id=trial_id,
                        trial=trial,
                        model=model,
                        attack_type=attack_type,
                        n_points=n_points,
                        magnitude_uv=magnitude_uv,
                        random_engine=random_engine,
                        targeted_engine=targeted_engine,
                    )

                    writer.writerow(row)
                    completed += 1

                    # Progress update every 100 experiments
                    if completed % 100 == 0 or completed == total:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (total - completed) / rate if rate > 0 else 0
                        print(
                            f"  [{completed:>{len(str(total))}}/{total}] "
                            f"{completed / total * 100:5.1f}% | "
                            f"{elapsed:.0f}s elapsed | "
                            f"ETA {eta:.0f}s"
                        )

        # Flush after each trial so data is saved even if interrupted
        csvfile.flush()

    csvfile.close()
    elapsed_total = time.time() - start_time

    # ── Summary statistics ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("EXPERIMENT SWEEP COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total experiments: {total}")
    print(f"  Time elapsed:      {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"  Results saved to:  {OUTPUT_CSV}")

    # Quick summary from the CSV we just wrote
    import pandas as pd
    df = pd.read_csv(OUTPUT_CSV)

    print(f"\n{'─' * 60}")
    print("SUMMARY: Misclassification rates by attack type & magnitude")
    print(f"{'─' * 60}")

    summary = (
        df.groupby(["attack_type", "n_points", "magnitude_uv"])["misclassified"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "flipped", "count": "total", "mean": "rate"})
    )
    summary["rate_pct"] = (summary["rate"] * 100).round(1)

    for (attack, npts), group in summary.groupby(level=[0, 1]):
        print(f"\n  {attack.upper()} attack, {npts} point(s):")
        print(f"    {'Magnitude (µV)':<18} {'Flipped':<10} {'Rate'}")
        print(f"    {'─' * 40}")
        for mag in MAGNITUDES_UV:
            if mag in group.index.get_level_values("magnitude_uv"):
                r = group.loc[(attack, npts, mag)]
                print(f"    {mag:<18} {int(r['flipped']):<10} {r['rate_pct']}%")

    avg_snr = df.groupby(["attack_type", "magnitude_uv"])["snr_db"].mean()
    print(f"\n  Average SNR (dB) by attack & magnitude:")
    print(f"    {'Magnitude':<12} {'Random':<12} {'Targeted'}")
    print(f"    {'─' * 36}")
    for mag in MAGNITUDES_UV:
        r_snr = avg_snr.get(("random", mag), float("nan"))
        t_snr = avg_snr.get(("targeted", mag), float("nan"))
        print(f"    {mag:<12} {r_snr:<12.1f} {t_snr:.1f}")

    print(f"\n{'=' * 60}")
    print(f"Done. CSV ready for Elizabeth's analysis: {OUTPUT_CSV}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()