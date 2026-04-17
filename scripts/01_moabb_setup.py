"""
01_moabb_setup.py - foolingAI-547
Download PhysioNet Motor Imagery dataset via MOABB and verify data integrity.
Run from project root: python scripts/01_moabb_setup.py
"""

import sys
import numpy as np
from pathlib import Path

RANDOM_SEED = 42
SUBJECTS = list(range(1, 11))  # Subjects 1-10; expand further if accuracy stays low


def load_dataset(subjects=SUBJECTS):
    """Download and load PhysioNet Motor Imagery dataset via MOABB.

    Returns:
        tuple: (X, y, metadata) where X is (n_trials, n_channels, n_times),
               y is integer class labels, metadata is a pandas DataFrame.
    """
    try:
        from moabb.datasets import PhysionetMI
    except ImportError:
        print("ERROR: MOABB not installed. Run: pip install moabb")
        sys.exit(1)

    print("Loading PhysioNet Motor Imagery dataset via MOABB...")
    print("(First run auto-downloads ~500MB to ~/mne_data/ — takes 5-15 min)")

    dataset = PhysionetMI()

    try:
        sessions = dataset.get_data(subjects=subjects)
    except Exception as e:
        print(f"ERROR downloading dataset: {e}")
        print("Check internet connection and try again.")
        sys.exit(1)

    # Flatten sessions into (X, y, subject_ids) arrays
    X_list, y_list, sid_list = [], [], []
    for subject_id, subject_data in sessions.items():
        for session_name, session_data in subject_data.items():
            for run_name, raw in session_data.items():
                try:
                    events = _extract_motor_imagery_events(raw)
                    if events is None:
                        continue
                    X_run, y_run = _epoch_raw(raw, events)
                    X_list.append(X_run)
                    y_list.append(y_run)
                    sid_list.append(np.full(len(y_run), subject_id, dtype=int))
                except Exception as e:
                    print(f"  Warning: skipping {subject_id}/{session_name}/{run_name}: {e}")

    if not X_list:
        print("ERROR: No usable data found. Check MOABB dataset config.")
        sys.exit(1)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subject_ids = np.concatenate(sid_list, axis=0)

    return X, y, subject_ids


def _extract_motor_imagery_events(raw):
    """Extract left/right hand motor imagery events from a Raw object."""
    import mne
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    # MOABB's PhysionetMI uses 'left_hand' / 'right_hand' annotation names
    mi_ids = {k: v for k, v in event_id.items() if k in ("left_hand", "right_hand")}
    if len(mi_ids) < 2:
        return None
    return events, mi_ids


def _epoch_raw(raw, events_tuple):
    """Epoch a Raw object around motor imagery events.

    Returns:
        tuple: (X, y) numpy arrays; X shape (n_epochs, n_channels, n_times)
    """
    import mne
    events, event_id = events_tuple
    tmin, tmax = 0.0, 2.0  # 2-second epochs

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )
    X = epochs.get_data(picks="eeg")  # (n_epochs, n_channels, n_times) — EEG only, drops stim channel
    y = (epochs.events[:, 2] == event_id["right_hand"]).astype(int)
    return X, y


def print_dataset_stats(X, y):
    """Print human-readable dataset statistics."""
    n_trials, n_channels, n_times = X.shape
    classes, counts = np.unique(y, return_counts=True)
    class_names = {0: "Left Hand", 1: "Right Hand"}

    print("\n" + "=" * 55)
    print("DATASET STATISTICS")
    print("=" * 55)
    print(f"  Trials:          {n_trials}")
    print(f"  EEG Channels:    {n_channels}")
    print(f"  Time Points:     {n_times}  (~{n_times/160:.1f}s @ 160 Hz)")
    print(f"  Classes:         {len(classes)}")
    for cls, count in zip(classes, counts):
        print(f"    [{cls}] {class_names.get(cls, cls)}: {count} trials")
    print(f"  Signal range:    [{X.min():.4f}, {X.max():.4f}] V")
    print(f"  Signal mean:     {X.mean():.6f} V")
    print("=" * 55)


def verify_data_integrity(X, y):
    """Basic sanity checks on loaded data.

    Raises:
        AssertionError: If data fails a sanity check.
    """
    assert X.ndim == 3, f"Expected 3D array (trials, channels, times), got {X.ndim}D"
    assert len(X) == len(y), "X and y length mismatch"
    assert not np.isnan(X).any(), "NaN values detected in X"
    assert not np.isinf(X).any(), "Inf values detected in X"
    assert set(np.unique(y)).issubset({0, 1}), f"Unexpected class labels: {np.unique(y)}"
    print("  Data integrity checks: PASSED")


def save_raw_data(X, y, subject_ids, out_dir=Path("data/raw")):
    """Save raw data arrays for use by subsequent scripts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_raw.npy", X)
    np.save(out_dir / "y_raw.npy", y)
    np.save(out_dir / "subject_ids.npy", subject_ids)
    print(f"  Raw data saved to {out_dir}/")


def run():
    np.random.seed(RANDOM_SEED)

    print("=" * 55)
    print("foolingAI-547  |  Script 01: MOABB Dataset Setup")
    print("=" * 55)

    X, y, subject_ids = load_dataset(subjects=SUBJECTS)
    print_dataset_stats(X, y)

    print("\nVerifying data integrity...")
    verify_data_integrity(X, y)

    print("\nSaving raw data...")
    save_raw_data(X, y, subject_ids)

    print("\nScript 01 complete.")
    print(f"Dataset loaded: {X.shape[0]} trials, {X.shape[1]} channels, "
          f"Motor Imagery (2 classes)")
    return X, y, subject_ids


if __name__ == "__main__":
    run()
