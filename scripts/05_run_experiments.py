"""
05_run_experiments.py - foolingAI-547
Full experiment sweep across multiple subjects.

Key design decisions (from TA/professor meeting):
  - Perturbation as percentage of signal range (1%, 2%, 3%) — the "knob"
  - Both random AND targeted attacks at each level
  - Also includes sub-1% targeted (0.5%) to show threshold effects
  - Only perturb trials the model classifies correctly at baseline
  - Per-subject model training (within-subject CSP+LDA)

Run from project root: python scripts/05_run_experiments.py

Author: Havi (experiment execution)
"""

import sys
import csv
import importlib.util
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from fbcsp import FilterBankCSP  # noqa: F401 — required for pickle to resolve model.pkl

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
RESULTS_DIR = Path("results")
DATA_DIR = Path("data/raw")
OUTPUT_CSV = RESULTS_DIR / "experiment_log.csv"

SUBJECTS = [1]                        # Single subject — FINAL_dataset_547
TEST_SIZE = 0.25
SFREQ = 160.0                        # PhysioNet sampling rate

# The "knob" — perturbation as % of each trial's peak-to-peak amplitude
PERTURBATION_PCTS = [0.5, 1.0, 2.0, 3.0]   # 0.5% is the sub-1% condition
ATTACK_TYPES = ["random", "targeted"]
N_POINTS_LIST = [1, 2]

CSV_COLUMNS = [
    "subject_id",
    "trial_id",
    "attack_type",
    "n_points",
    "perturbation_pct",
    "magnitude_uv",
    "orig_pred",
    "pert_pred",
    "misclassified",
    "snr_db",
    "baseline_acc",
    "trial_range_uv",
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


LABEL_MAP = {"feet": 0, "left_hand": 1, "rest": 2}  # right_hand excluded


def load_all_raw_data():
    """Load FINAL_dataset_547 files and return (X, y_int, subject_ids)."""
    x_path = DATA_DIR / "FINAL_dataset_547_data_300.npy"
    y_path = DATA_DIR / "NEW_dataset_547_labels_300.npy"

    for p in (x_path, y_path):
        if not p.exists():
            print(f"ERROR: {p} not found. Check data/raw/.")
            sys.exit(1)

    X = np.load(x_path) * 1e-6          # µV → V
    y_raw = np.load(y_path)

    mask = np.array([lbl in LABEL_MAP for lbl in y_raw])
    X = X[mask]
    y = np.array([LABEL_MAP[lbl] for lbl in y_raw[mask]], dtype=int)
    subject_ids = np.ones(len(X), dtype=int)   # all trials → subject 1

    print(f"  Loaded: {X.shape[0]} trials, {X.shape[1]} channels, "
          f"{X.shape[2]} time points")
    return X, y, subject_ids


def build_csp_lda_pipeline():
    """Build FBCSP + StandardScaler + RBF-SVM pipeline (matches script 02)."""
    from fbcsp import FilterBankCSP, FREQ_BANDS, CSP_COMPONENTS_PER_BAND

    return Pipeline([
        ("fbcsp",  FilterBankCSP(freq_bands=FREQ_BANDS,
                                 n_components=CSP_COMPONENTS_PER_BAND,
                                 sfreq=SFREQ)),
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=100, gamma="scale", probability=True)),
    ])


def preprocess_and_split(X, y):
    """Stratified 75/25 train/test split. FBCSP handles band filtering internally."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def get_correctly_classified_trials(model, X_test, y_test):
    """Filter test set to only trials the model predicts correctly.

    This ensures perturbation results are not confounded by trials
    the model was already getting wrong (per TA guidance).

    Returns:
        X_correct, y_correct, correct_indices (original indices into X_test)
    """
    preds = model.predict(X_test)
    correct_mask = preds == y_test
    indices = np.where(correct_mask)[0]
    return X_test[correct_mask], y_test[correct_mask], indices


def compute_trial_magnitude_uv(trial, pct):
    """Compute perturbation magnitude in µV as a percentage of the trial's
    peak-to-peak amplitude.

    Args:
        trial: EEG array (n_channels, n_times) in volts.
        pct:   Perturbation percentage (e.g. 1.0 means 1%).

    Returns:
        float: magnitude in µV
    """
    peak_to_peak_v = trial.max() - trial.min()
    peak_to_peak_uv = peak_to_peak_v / 1e-6   # convert V → µV
    return peak_to_peak_uv * (pct / 100.0)


def run_single_experiment(
    trial, model, attack_type, n_points, magnitude_uv,
    random_engine, targeted_engine,
):
    """Run one perturbation experiment. Returns result dict."""
    orig_pred = int(model.predict(trial[np.newaxis])[0])

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

    # ── Load raw data ────────────────────────────────────────────────────
    print("\nLoading raw data...")
    X_all, y_all, subject_ids = load_all_raw_data()
    print(f"  Total: {X_all.shape[0]} trials across "
          f"{len(np.unique(subject_ids))} subjects")

    # ── Load perturbation engines ────────────────────────────────────────
    print("Loading perturbation engines...")
    pm = load_perturbation_module()
    random_engine = pm.PerturbationEngine(random_seed=RANDOM_SEED)
    targeted_engine = pm.TargetedPerturbationEngine(random_seed=RANDOM_SEED)

    # ── Print experiment grid ────────────────────────────────────────────
    conditions_per_trial = (
        len(ATTACK_TYPES) * len(N_POINTS_LIST) * len(PERTURBATION_PCTS)
    )
    print(f"\n  Experiment grid per trial:")
    print(f"    Perturbation %%: {PERTURBATION_PCTS}")
    print(f"    Attack types:    {ATTACK_TYPES}")
    print(f"    N points:        {N_POINTS_LIST}")
    print(f"    Conditions:      {conditions_per_trial} per trial")
    print(f"    Subjects:        {SUBJECTS}")
    print(f"\n  NOTE: Only correctly-classified trials are used.")

    # ── Open CSV ─────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csvfile = open(OUTPUT_CSV, "w", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    completed = 0
    total_correct_trials = 0
    start_time = time.time()
    subject_summaries = []

    print(f"\n{'━' * 60}")

    for subj in SUBJECTS:
        # ── Filter to this subject ───────────────────────────────────────
        mask = subject_ids == subj
        X_subj, y_subj = X_all[mask], y_all[mask]

        if len(X_subj) < 8:
            print(f"\n  Subject {subj}: only {len(X_subj)} trials — "
                  f"skipping (need >= 8)")
            continue

        if len(np.unique(y_subj)) < 2:
            print(f"\n  Subject {subj}: only one class — skipping")
            continue

        # ── Preprocess and split ─────────────────────────────────────────
        try:
            X_train, X_test, y_train, y_test = preprocess_and_split(
                X_subj, y_subj
            )
        except ValueError as e:
            print(f"\n  Subject {subj}: split failed ({e}) — skipping")
            continue

        # ── Train per-subject model ──────────────────────────────────────
        try:
            model = build_csp_lda_pipeline()
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"\n  Subject {subj}: training failed ({e}) — skipping")
            continue

        # ── Baseline accuracy ────────────────────────────────────────────
        baseline_acc = accuracy_score(y_test, model.predict(X_test))

        # ── Filter to correctly classified trials only ───────────────────
        X_correct, y_correct, correct_idx = get_correctly_classified_trials(
            model, X_test, y_test
        )
        n_correct = len(X_correct)
        total_correct_trials += n_correct
        subj_experiments = n_correct * conditions_per_trial

        print(f"\n  Subject {subj:>2}: {len(X_subj)} total → "
              f"{len(X_test)} test → {n_correct} correct | "
              f"acc: {baseline_acc*100:.1f}% | "
              f"{subj_experiments} experiments")

        if n_correct == 0:
            print(f"    No correctly classified trials — skipping")
            subject_summaries.append({
                "subject": subj,
                "n_test": len(X_test),
                "n_correct": 0,
                "baseline_acc": baseline_acc,
                "overall_flip_rate": 0.0,
            })
            continue

        # ── Run experiments ──────────────────────────────────────────────
        subj_misclassified = 0
        subj_total = 0

        for local_id, global_id in enumerate(correct_idx):
            trial = X_correct[local_id]

            # Compute trial's peak-to-peak range for percentage conversion
            trial_range_uv = (trial.max() - trial.min()) / 1e-6

            for attack_type in ATTACK_TYPES:
                for n_points in N_POINTS_LIST:
                    for pct in PERTURBATION_PCTS:
                        # Convert percentage to µV for this trial
                        mag_uv = trial_range_uv * (pct / 100.0)

                        result = run_single_experiment(
                            trial, model, attack_type, n_points, mag_uv,
                            random_engine, targeted_engine,
                        )

                        row = {
                            "subject_id": subj,
                            "trial_id": int(global_id),
                            "attack_type": attack_type,
                            "n_points": n_points,
                            "perturbation_pct": pct,
                            "magnitude_uv": round(mag_uv, 2),
                            "orig_pred": result["orig_pred"],
                            "pert_pred": result["pert_pred"],
                            "misclassified": result["misclassified"],
                            "snr_db": result["snr_db"],
                            "baseline_acc": round(baseline_acc, 4),
                            "trial_range_uv": round(trial_range_uv, 2),
                        }
                        writer.writerow(row)
                        completed += 1

                        subj_misclassified += result["misclassified"]
                        subj_total += 1

            # Progress update every 100 experiments
            if completed % 100 == 0:
                elapsed = time.time() - start_time
                print(f"    [{completed} done] {elapsed:.0f}s elapsed")

        # Flush after each subject
        csvfile.flush()

        subj_rate = (
            subj_misclassified / subj_total * 100 if subj_total > 0 else 0
        )
        subject_summaries.append({
            "subject": subj,
            "n_test": len(X_test),
            "n_correct": n_correct,
            "baseline_acc": baseline_acc,
            "overall_flip_rate": subj_rate,
        })

    csvfile.close()
    elapsed_total = time.time() - start_time

    # ══════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 60}")
    print(f"{'=' * 60}")
    print("EXPERIMENT SWEEP COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total experiments:       {completed}")
    print(f"  Correct trials tested:   {total_correct_trials}")
    print(f"  Subjects processed:      {len(subject_summaries)}")
    print(f"  Time elapsed:            {elapsed_total:.1f}s "
          f"({elapsed_total/60:.1f} min)")
    print(f"  Results saved to:        {OUTPUT_CSV}")

    # ── Per-subject summary ──────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("PER-SUBJECT SUMMARY")
    print(f"{'─' * 60}")
    print(f"  {'Subj':<6} {'Test':<7} {'Correct':<9} "
          f"{'Acc':<9} {'Flip Rate'}")
    print(f"  {'─' * 45}")
    for s in subject_summaries:
        print(f"  {s['subject']:<6} {s['n_test']:<7} {s['n_correct']:<9} "
              f"{s['baseline_acc']*100:<8.1f}% "
              f"{s['overall_flip_rate']:.1f}%")

    # ── Detailed breakdown from CSV ──────────────────────────────────────
    import pandas as pd
    df = pd.read_csv(OUTPUT_CSV)

    if len(df) == 0:
        print("\n  No experiments recorded. Check subject data.")
        return

    print(f"\n{'─' * 60}")
    print("MISCLASSIFICATION RATES BY ATTACK TYPE & PERTURBATION %")
    print(f"{'─' * 60}")

    summary = (
        df.groupby(["attack_type", "n_points", "perturbation_pct"])
        ["misclassified"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "flipped", "count": "total", "mean": "rate"})
    )
    summary["rate_pct"] = (summary["rate"] * 100).round(1)

    for (attack, npts), group in summary.groupby(level=[0, 1]):
        print(f"\n  {attack.upper()} attack, {npts} point(s):")
        print(f"    {'Perturbation %':<18} {'Flipped':<10} "
              f"{'Total':<10} {'Misclass Rate'}")
        print(f"    {'─' * 50}")
        for pct in PERTURBATION_PCTS:
            if pct in group.index.get_level_values("perturbation_pct"):
                r = group.loc[(attack, npts, pct)]
                print(f"    {pct:<18} {int(r['flipped']):<10} "
                      f"{int(r['total']):<10} {r['rate_pct']}%")

    # ── SNR summary ──────────────────────────────────────────────────────
    avg_snr = df.groupby(
        ["attack_type", "perturbation_pct"]
    )["snr_db"].mean()

    print(f"\n  Average SNR (dB) by attack & perturbation %:")
    print(f"    {'Pert %':<12} {'Random':<12} {'Targeted'}")
    print(f"    {'─' * 36}")
    for pct in PERTURBATION_PCTS:
        r_snr = avg_snr.get(("random", pct), float("nan"))
        t_snr = avg_snr.get(("targeted", pct), float("nan"))
        print(f"    {pct:<12} {r_snr:<12.1f} {t_snr:.1f}")

    # ── Cross-subject aggregate ──────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("AGGREGATE: Targeted attack misclassification across all subjects")
    print(f"{'─' * 60}")
    targeted_df = df[df["attack_type"] == "targeted"]
    if len(targeted_df) > 0:
        agg = (
            targeted_df.groupby(["perturbation_pct", "n_points"])
            ["misclassified"]
            .agg(["sum", "count", "mean"])
        )
        agg["rate_pct"] = (agg["mean"] * 100).round(1)
        print(f"    {'Pert %':<10} {'N pts':<8} {'Flipped':<10} "
              f"{'Total':<10} {'Rate'}")
        print(f"    {'─' * 48}")
        for (pct, npts), r in agg.iterrows():
            print(f"    {pct:<10} {npts:<8} {int(r['sum']):<10} "
                  f"{int(r['count']):<10} {r['rate_pct']}%")

    print(f"\n{'=' * 60}")
    print(f"Done. CSV ready for analysis: {OUTPUT_CSV}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()