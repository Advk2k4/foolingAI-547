"""
02_load_pretrained.py - foolingAI-547
Load a pre-trained MOABB pipeline (CSP + LDA) for Motor Imagery classification,
evaluate baseline accuracy, and save artefacts for downstream scripts.
Run from project root: python scripts/02_load_pretrained.py
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

RANDOM_SEED = 42
TEST_SIZE = 0.25
RESULTS_DIR = Path("results")
DATA_DIR = Path("data/raw")
EXPECTED_ACCURACY_MIN = 0.55  # Lower bound; real MI accuracy varies by subject
TRAIN_SUBJECT = 1  # Train within a single subject — cross-subject CSP+LDA is at chance


def load_raw_data():
    """Load X, y from data/raw/, filtered to TRAIN_SUBJECT (written by script 01)."""
    x_path = DATA_DIR / "X_raw.npy"
    y_path = DATA_DIR / "y_raw.npy"
    sid_path = DATA_DIR / "subject_ids.npy"

    for p in (x_path, y_path, sid_path):
        if not p.exists():
            print(f"ERROR: {p} not found. Run 01_moabb_setup.py first.")
            sys.exit(1)

    X = np.load(x_path)
    y = np.load(y_path)
    subject_ids = np.load(sid_path)

    mask = subject_ids == TRAIN_SUBJECT
    X, y = X[mask], y[mask]
    print(f"  Subject {TRAIN_SUBJECT}: {X.shape[0]} trials, {X.shape[1]} channels")
    return X, y


def build_csp_lda_pipeline():
    """Build a CSP + LDA Motor Imagery pipeline (standard MOABB benchmark).

    CSP (Common Spatial Patterns) is the canonical EEG feature extractor.
    LDA is the standard linear classifier for BCI research.

    Returns:
        sklearn.pipeline.Pipeline
    """
    from mne.decoding import CSP

    pipeline = Pipeline([
        ("csp", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
        ("lda", LinearDiscriminantAnalysis()),
    ])
    return pipeline


def preprocess_and_split(X, y):
    """Apply bandpass filter and split into train/test sets.

    Args:
        X: Raw EEG (n_trials, n_channels, n_times)
        y: Integer labels

    Returns:
        X_train, X_test, y_train, y_test (all numpy arrays)
    """
    from scipy.signal import butter, sosfiltfilt

    print("  Applying bandpass filter (8-30 Hz, motor imagery band)...")
    sfreq = 160.0
    sos = butter(4, [8.0, 30.0], btype="bandpass", fs=sfreq, output="sos")
    X_filt = sosfiltfilt(sos, X, axis=-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_filt, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"  Train: {X_train.shape[0]} trials | Test: {X_test.shape[0]} trials")
    return X_train, X_test, y_train, y_test


def train_and_evaluate(pipeline, X_train, X_test, y_train, y_test):
    """Fit the pipeline and return test accuracy.

    Args:
        pipeline: sklearn Pipeline
        X_train, y_train: training data
        X_test, y_test: test data

    Returns:
        float: test accuracy
    """
    print("  Fitting CSP + LDA pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Baseline accuracy: {acc * 100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Left Hand", "Right Hand"],
                                digits=3))
    return acc, y_pred


def save_artefacts(pipeline, X_test, y_test):
    """Persist model and test data for scripts 03/04 and Havi's experiments."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    np.save(RESULTS_DIR / "X_test.npy", X_test)
    np.save(RESULTS_DIR / "y_test.npy", y_test)

    print(f"\n  Saved artefacts to {RESULTS_DIR}/:")
    print(f"    model.pkl       — trained CSP + LDA pipeline")
    print(f"    X_test.npy      — {X_test.shape[0]} test trials")
    print(f"    y_test.npy      — {y_test.shape[0]} test labels")


def run():
    np.random.seed(RANDOM_SEED)

    print("=" * 55)
    print("foolingAI-547  |  Script 02: Load Pre-Trained Model")
    print("=" * 55)

    print("\nLoading raw data...")
    X, y = load_raw_data()

    print("\nPreprocessing and splitting data...")
    X_train, X_test, y_train, y_test = preprocess_and_split(X, y)

    print("\nBuilding CSP + LDA pipeline...")
    pipeline = build_csp_lda_pipeline()

    print("\nTraining and evaluating...")
    acc, _ = train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)

    if acc < EXPECTED_ACCURACY_MIN:
        print(f"  WARNING: Accuracy {acc:.2%} is below expected minimum "
              f"({EXPECTED_ACCURACY_MIN:.0%}). This may indicate data issues.")
    else:
        print(f"  Accuracy within expected range. Pipeline is healthy.")

    print("\nSaving artefacts...")
    save_artefacts(pipeline, X_test, y_test)

    print("\nScript 02 complete.")
    print(f"Model loaded. Baseline accuracy: {acc * 100:.2f}%")
    return pipeline, X_test, y_test


if __name__ == "__main__":
    run()
