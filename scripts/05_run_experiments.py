"""
05_run_experiments.py - foolingAI-547
Full experiment sweep: load model.pkl + X_test.npy, run all 16 conditions.

Key design decisions:
  - Perturbation as percentage of signal range — the "knob"
  - Both random AND targeted attacks at each level
  - Only perturb trials the model classifies correctly at baseline
  - Uses the trained FBCSP+SVM model from script 02

Run from project root: python scripts/05_run_experiments.py
"""

import sys
import csv
import pickle
import importlib.util
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from fbcsp import FilterBankCSP  # noqa: F401 — required for pickle to resolve model.pkl

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
RESULTS_DIR = Path("results")
OUTPUT_CSV = RESULTS_DIR / "experiment_log.csv"

# The "knob" — perturbation as % of each trial's peak-to-peak amplitude
PERTURBATION_PCTS = [5.0, 10.0, 15.0, 20.0]
ATTACK_TYPES = ["random", "targeted"]
N_POINTS_LIST = [1, 2]

CSV_COLUMNS = [
    "trial_id",
    "attack_type",
    "n_points",
    "perturbation_pct",
    "magnitude_uv",
    "orig_pred",
    "pert_pred",
    "misclassified",
    "snr_db",
    "trial_range_uv",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_perturbation_module():
    script_dir = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "perturbation_module", script_dir / "03_perturbation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_artefacts():
    model_path = RESULTS_DIR / "model.pkl"
    x_path = RESULTS_DIR / "X_test.npy"
    y_path = RESULTS_DIR / "y_test.npy"

    for p in (model_path, x_path, y_path):
        if not p.exists():
            print(f"ERROR: {p} not found. Run 02_load_pretrained.py first.")
            sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_test = np.load(x_path)
    y_test = np.load(y_path)
    return model, X_test, y_test


def get_correctly_classified_trials(model, X_test, y_test):
    """Return only trials the model already predicts correctly."""
    preds = model.predict(X_test)
    correct_mask = preds == y_test
    indices = np.where(correct_mask)[0]
    return X_test[correct_mask], y_test[correct_mask], indices


def run_single_experiment(
    trial, model, attack_type, n_points, magnitude_uv,
    random_engine, targeted_engine,
):
    orig_pred = int(model.predict(trial[np.newaxis])[0])

    if attack_type == "random":
        indices = random_engine.select_random_samples(trial, num_points=n_points)
        perturbed = random_engine.apply_perturbation(trial, indices, magnitude_uv)
        metrics = random_engine.verify_imperceptibility(trial, perturbed)
    else:
        perturbed = targeted_engine.apply_targeted_perturbation(
            trial, model, magnitude_uv, num_points=n_points
        )
        metrics = targeted_engine.verify_imperceptibility(trial, perturbed)

    pert_pred = int(model.predict(perturbed[np.newaxis])[0])

    return {
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

    print("\nLoading model and test data...")
    model, X_test, y_test = load_artefacts()

    baseline_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  Baseline accuracy: {baseline_acc*100:.1f}%")

    X_correct, y_correct, correct_idx = get_correctly_classified_trials(
        model, X_test, y_test
    )
    n_correct = len(X_correct)
    print(f"  Test trials:       {len(X_test)}")
    print(f"  Correctly classified: {n_correct}")

    print("\nLoading perturbation engines...")
    pm = load_perturbation_module()
    random_engine = pm.PerturbationEngine(random_seed=RANDOM_SEED)
    targeted_engine = pm.TargetedPerturbationEngine(random_seed=RANDOM_SEED)

    conditions_per_trial = len(ATTACK_TYPES) * len(N_POINTS_LIST) * len(PERTURBATION_PCTS)
    total = n_correct * conditions_per_trial
    print(f"\n  Experiment grid:")
    print(f"    Perturbation %%: {PERTURBATION_PCTS}")
    print(f"    Attack types:    {ATTACK_TYPES}")
    print(f"    N points:        {N_POINTS_LIST}")
    print(f"    Correct trials:  {n_correct}")
    print(f"    Total:           {total} experiments")
    print(f"\n  NOTE: Only correctly-classified trials are used.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csvfile = open(OUTPUT_CSV, "w", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    completed = 0
    start_time = time.time()

    print(f"\n{'━' * 60}")

    for local_id, global_id in enumerate(correct_idx):
        trial = X_correct[local_id]
        trial_range_uv = (trial.max() - trial.min()) / 1e-6

        for attack_type in ATTACK_TYPES:
            for n_points in N_POINTS_LIST:
                for pct in PERTURBATION_PCTS:
                    mag_uv = trial_range_uv * (pct / 100.0)

                    result = run_single_experiment(
                        trial, model, attack_type, n_points, mag_uv,
                        random_engine, targeted_engine,
                    )

                    writer.writerow({
                        "trial_id":        int(global_id),
                        "attack_type":     attack_type,
                        "n_points":        n_points,
                        "perturbation_pct": pct,
                        "magnitude_uv":    round(mag_uv, 2),
                        "orig_pred":       result["orig_pred"],
                        "pert_pred":       result["pert_pred"],
                        "misclassified":   result["misclassified"],
                        "snr_db":          result["snr_db"],
                        "trial_range_uv":  round(trial_range_uv, 2),
                    })
                    completed += 1

        if completed % 50 == 0 or completed == total:
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0
            print(f"  [{completed:>{len(str(total))}}/{total}] "
                  f"{completed/total*100:5.1f}% | {elapsed:.0f}s elapsed | ETA {eta:.0f}s")

        csvfile.flush()

    csvfile.close()
    elapsed_total = time.time() - start_time

    # ── Summary ──────────────────────────────────────────────────────────────
    import pandas as pd
    df = pd.read_csv(OUTPUT_CSV)

    print(f"\n{'=' * 60}")
    print("EXPERIMENT SWEEP COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total experiments: {completed}")
    print(f"  Time elapsed:      {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"  Results saved to:  {OUTPUT_CSV}")

    print(f"\n{'─' * 60}")
    print("MISCLASSIFICATION RATES BY ATTACK TYPE & PERTURBATION %")
    print(f"{'─' * 60}")

    for (attack, npts), group in df.groupby(["attack_type", "n_points"]):
        rates = group.groupby("perturbation_pct")["misclassified"].mean() * 100
        print(f"\n  {attack.upper()} attack, {npts} point(s):")
        print(f"    {'Pert %':<12} {'Flipped':<10} {'Rate'}")
        print(f"    {'─' * 32}")
        for pct in PERTURBATION_PCTS:
            flipped = int(group[group["perturbation_pct"] == pct]["misclassified"].sum())
            rate = rates.get(pct, 0.0)
            print(f"    {pct}%{'':<9} {flipped:<10} {rate:.1f}%")

    print(f"\n{'=' * 60}")
    print(f"Done. Run script 06 to generate figures.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
