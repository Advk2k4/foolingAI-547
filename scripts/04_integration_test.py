"""
04_integration_test.py - foolingAI-547
End-to-end integration test: load pre-trained model + test data, apply
1-point perturbation (1µV), compare original vs. perturbed predictions.
Run from project root: python scripts/04_integration_test.py
"""

import sys
import pickle
import importlib.util
import numpy as np
from pathlib import Path

RANDOM_SEED = 42
RESULTS_DIR = Path("results")
CLASS_NAMES = {0: "left hand", 1: "right hand"}
PERTURBATION_MAGNITUDE_UV = 1.0
N_POINTS = 1


def _load_perturbation_module():
    """Dynamically import perturbation classes from 03_perturbation.py."""
    script_dir = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "perturbation_module", script_dir / "03_perturbation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_artefacts():
    """Load model and test data written by 02_load_pretrained.py."""
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

    print(f"  Model loaded:   {type(model).__name__} pipeline")
    print(f"  X_test shape:   {X_test.shape}")
    print(f"  y_test shape:   {y_test.shape}")
    return model, X_test, y_test


def run_integration_test(model, X_test, y_test, engine):
    """Apply perturbation to each test trial and compare predictions.

    Args:
        model:  Fitted sklearn pipeline (CSP + LDA).
        X_test: Test trials (n_trials, n_channels, n_times).
        y_test: True labels.
        engine: PerturbationEngine instance.

    Returns:
        dict with per-trial results and aggregate statistics.
    """
    results = []

    for i, (trial, true_label) in enumerate(zip(X_test, y_test)):
        # Original prediction (model expects 3-D batch input for CSP)
        orig_pred = model.predict(trial[np.newaxis, :])[0]

        # Perturb selected sample points
        idx = engine.select_random_samples(trial, num_points=N_POINTS)
        perturbed = engine.apply_perturbation(
            trial, idx, magnitude_uv=PERTURBATION_MAGNITUDE_UV
        )
        pert_pred = model.predict(perturbed[np.newaxis, :])[0]

        metrics = engine.verify_imperceptibility(trial, perturbed)
        misclassified = int(orig_pred) != int(pert_pred)

        results.append({
            "trial_id": i,
            "true_label": int(true_label),
            "orig_pred": int(orig_pred),
            "pert_pred": int(pert_pred),
            "misclassified": misclassified,
            "snr_db": metrics["snr_db"],
        })

    n_total = len(results)
    n_misclassified = sum(r["misclassified"] for r in results)
    misclassification_rate = n_misclassified / n_total if n_total else 0.0
    avg_snr = float(np.mean([r["snr_db"] for r in results]))

    return {
        "results": results,
        "n_total": n_total,
        "n_misclassified": n_misclassified,
        "misclassification_rate": misclassification_rate,
        "avg_snr_db": avg_snr,
    }


def run_magnitude_sweep(model, X_test, random_engine, targeted_engine, magnitudes):
    """Sweep perturbation magnitudes and report random vs targeted misclassification rates."""
    print("\n" + "=" * 55)
    print("MAGNITUDE SWEEP — Random vs Targeted Attack")
    print("=" * 55)
    print(f"  {'Magnitude (µV)':<16} {'Random':<12} {'Targeted':<12} {'SNR (dB)'}")
    print("  " + "-" * 52)

    for mag in magnitudes:
        rand_mc = targeted_mc = 0
        snrs = []

        for trial in X_test:
            orig = model.predict(trial[np.newaxis])[0]

            idx = random_engine.select_random_samples(trial, num_points=N_POINTS)
            rand_pert = random_engine.apply_perturbation(trial, idx, magnitude_uv=mag)
            if model.predict(rand_pert[np.newaxis])[0] != orig:
                rand_mc += 1

            tgt_pert = targeted_engine.apply_targeted_perturbation(
                trial, model, magnitude_uv=mag, num_points=N_POINTS
            )
            if model.predict(tgt_pert[np.newaxis])[0] != orig:
                targeted_mc += 1

            m = random_engine.verify_imperceptibility(trial, tgt_pert)
            snrs.append(m["snr_db"])

        n = len(X_test)
        avg_snr = float(np.mean(snrs))
        print(
            f"  {mag:<16} {rand_mc/n*100:>5.1f}%       {targeted_mc/n*100:>5.1f}%"
            f"       {avg_snr:.1f}"
        )

    print("=" * 55)


def print_summary(report: dict):
    """Print a human-readable integration test report."""
    print("\n" + "=" * 55)
    print("INTEGRATION TEST RESULTS  (random attack @ 1µV)")
    print("=" * 55)
    print(f"  Trials tested:          {report['n_total']}")
    print(f"  Misclassified:          {report['n_misclassified']}")
    print(
        f"  Misclassification rate: "
        f"{report['misclassification_rate'] * 100:.1f}%"
    )
    print(f"  Average SNR:            {report['avg_snr_db']:.1f} dB")
    print(f"  Perturbation:           {N_POINTS} point(s) @ {PERTURBATION_MAGNITUDE_UV}µV")
    print("=" * 55)


def run():
    np.random.seed(RANDOM_SEED)

    print("=" * 55)
    print("foolingAI-547  |  Script 04: Integration Test")
    print("=" * 55)

    print("\nLoading model and test data...")
    model, X_test, y_test = load_artefacts()

    pm = _load_perturbation_module()
    random_engine = pm.PerturbationEngine(random_seed=RANDOM_SEED)
    targeted_engine = pm.TargetedPerturbationEngine(random_seed=RANDOM_SEED)

    print(
        f"\nRunning random attack: "
        f"{N_POINTS}-point perturbation @ {PERTURBATION_MAGNITUDE_UV}µV..."
    )
    report = run_integration_test(model, X_test, y_test, random_engine)
    print_summary(report)

    print("\nRunning magnitude sweep (random vs targeted)...")
    magnitudes = pm.MAGNITUDE_SWEEP_UV
    run_magnitude_sweep(model, X_test, random_engine, targeted_engine, magnitudes)

    print("\nIntegration test complete ✓")
    return report


if __name__ == "__main__":
    run()
