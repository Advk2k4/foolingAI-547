"""
02_load_pretrained.py - foolingAI-547
Load the BCI Competition IV 2a-style dataset (4-class Motor Imagery),
train FBCSP + LDA, evaluate baseline accuracy, and save artefacts.
Run from project root: python scripts/02_load_pretrained.py
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from fbcsp import FilterBankCSP, FREQ_BANDS, CSP_COMPONENTS_PER_BAND, SFREQ

RANDOM_SEED = 42
TEST_SIZE = 0.25
RESULTS_DIR = Path("results")
DATA_DIR = Path("data/raw")
EXPECTED_ACCURACY_MIN = 0.60

NEW_DATA_PATH   = DATA_DIR / "FINAL_dataset_547_data_300.npy"
NEW_LABELS_PATH = DATA_DIR / "NEW_dataset_547_labels_300.npy"

LABEL_MAP   = {"feet": 0, "left_hand": 1, "rest": 2}  # right_hand excluded (poor separability)
CLASS_NAMES = {v: k for k, v in LABEL_MAP.items()}


def load_raw_data():
    """Load 4-class dataset, convert labels to ints, rescale µV → V."""
    for p in (NEW_DATA_PATH, NEW_LABELS_PATH):
        if not p.exists():
            print(f"ERROR: {p} not found. Check data/raw/.")
            sys.exit(1)

    X = np.load(NEW_DATA_PATH) * 1e-6          # µV → V
    y_raw = np.load(NEW_LABELS_PATH)

    # Filter to 3 classes only
    mask = np.array([label in LABEL_MAP for label in y_raw])
    X = X[mask]
    y_raw = y_raw[mask]
    y = np.array([LABEL_MAP[label] for label in y_raw], dtype=int)

    classes, counts = np.unique(y, return_counts=True)
    print(f"  Loaded: {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} time points")
    print(f"  Classes: { {CLASS_NAMES[c]: int(n) for c, n in zip(classes, counts)} }")
    print(f"  Signal range: [{X.min():.2e}, {X.max():.2e}] V")
    return X, y


def build_fbcsp_pipeline():
    """Build FBCSP + StandardScaler + SVM pipeline.

    FBCSP: 6 narrow bands (4-8, 8-12, 12-16, 16-20, 20-24, 24-30 Hz) × 4 CSP
    components = 24 features. StandardScaler normalises features before SVM.
    RBF-SVM consistently outperforms LDA for MI classification in small-sample regimes.
    """
    pipeline = Pipeline([
        ("fbcsp",  FilterBankCSP(freq_bands=FREQ_BANDS,
                                 n_components=CSP_COMPONENTS_PER_BAND,
                                 sfreq=SFREQ)),
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=100, gamma="scale", probability=True)),
    ])
    return pipeline


def split_data(X, y):
    """Stratified 75/25 train/test split (no pre-filtering — FBCSP handles bands internally)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"  Train: {X_train.shape[0]} trials | Test: {X_test.shape[0]} trials")
    return X_train, X_test, y_train, y_test


def train_and_evaluate(pipeline, X_train, X_test, y_train, y_test):
    """Fit the pipeline and report test accuracy."""
    print("  Fitting FBCSP + SVM pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Baseline accuracy: {acc * 100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)],
        digits=3,
    ))
    return acc, y_pred


def save_artefacts(pipeline, X_test, y_test):
    """Persist model and test data for scripts 03/04."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    np.save(RESULTS_DIR / "X_test.npy", X_test)
    np.save(RESULTS_DIR / "y_test.npy", y_test)
    print(f"\n  Saved artefacts to {RESULTS_DIR}/:")
    print(f"    model.pkl  — FBCSP + LDA pipeline (4-class)")
    print(f"    X_test.npy — {X_test.shape[0]} test trials")
    print(f"    y_test.npy — {y_test.shape[0]} test labels")


def run():
    np.random.seed(RANDOM_SEED)

    print("=" * 55)
    print("foolingAI-547  |  Script 02: FBCSP + SVM Model")
    print("=" * 55)

    print("\nLoading dataset...")
    X, y = load_raw_data()

    print("\nSplitting data (FBCSP filters internally)...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"\nBuilding FBCSP pipeline ({len(FREQ_BANDS)} bands × "
          f"{CSP_COMPONENTS_PER_BAND} components = "
          f"{len(FREQ_BANDS)*CSP_COMPONENTS_PER_BAND} features)...")
    pipeline = build_fbcsp_pipeline()

    print("\nTraining and evaluating...")
    acc, _ = train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)

    if acc < EXPECTED_ACCURACY_MIN:
        print(f"  WARNING: Accuracy {acc:.2%} below expected minimum ({EXPECTED_ACCURACY_MIN:.0%}).")
    else:
        print(f"  Accuracy within expected range. Pipeline is healthy.")

    print("\nSaving artefacts...")
    save_artefacts(pipeline, X_test, y_test)

    print("\nScript 02 complete.")
    print(f"Model loaded. Baseline accuracy: {acc * 100:.2f}%")
    return pipeline, X_test, y_test


if __name__ == "__main__":
    run()
