# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**foolingAI-547** — Vulnerability assessment of EEG-based Brain-Computer Interfaces (BCIs).  
Research question: How drastically do EEG classifications change when only 1–2 sample points are modified?  
Expected finding: 1–2 sample point perturbations at 1–10 µV cause dramatic misclassification despite being imperceptible (SNR > 20 dB).

---

## Run Commands

Always run from project root (`foolingAI-547/`). Scripts must run in order:

```bash
pip install -r requirements.txt

python scripts/01_moabb_setup.py       # Downloads PhysioNet dataset (~500 MB, first run only)
python scripts/02_load_pretrained.py   # Trains CSP+LDA, saves artefacts to results/
python scripts/03_perturbation.py      # Tests PerturbationEngine (standalone, no deps)
python scripts/04_integration_test.py  # End-to-end pipeline check
```

---

## Architecture & Data Flow

```
PhysioNet (online)
    │ MOABB auto-download
    ▼
01_moabb_setup.py  →  data/raw/X_raw.npy, y_raw.npy
    ▼
02_load_pretrained.py
    [4th-order Butterworth bandpass 8-30 Hz → 75/25 train/test split → CSP(4) + LDA fit]
    →  results/model.pkl, X_test.npy, y_test.npy
    ▼
04_integration_test.py  ←── dynamically imports PerturbationEngine from 03_perturbation.py
    [For each trial: predict original → perturb 1pt@1µV → predict again → report rate]
```

Script 04 uses `importlib.util` to load `PerturbationEngine` from script 03 because `scripts/` has no `__init__.py`.

**Model:** CSP (4 components) → LDA. Input: `(n_trials, 64_channels, n_times)` bandpass-filtered EEG. Output: binary (0=Left Hand, 1=Right Hand). Expected accuracy: 60–85%.

**Signal units:** MOABB/MNE stores EEG in **volts (V)**. All perturbation magnitudes are in **µV** and converted internally (`1µV = 1e-6 V`).

---

## Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| `RANDOM_SEED` | `42` | All scripts |
| `SUBJECTS` | `[1, 2, 3]` | `01_moabb_setup.py` |
| `TEST_SIZE` | `0.25` | `02_load_pretrained.py` |
| `MAGNITUDE_SWEEP_UV` | `[0.5, 1.0, 2.0, 5.0, 10.0]` µV | `03_perturbation.py` |
| `SNR_THRESHOLD_DB` | `20.0` dB | `03_perturbation.py` |
| `EXPECTED_ACCURACY_MIN` | `0.55` | `02_load_pretrained.py` |
| `PERTURBATION_MAGNITUDE_UV` | `1.0` µV | `04_integration_test.py` |
| `N_POINTS` | `1` | `04_integration_test.py` |

---

## PerturbationEngine API

```python
# Direct import (if running from project root with scripts/ on path)
from scripts.03_perturbation import PerturbationEngine

# Dynamic import (as used in 04_integration_test.py)
import importlib.util
spec = importlib.util.spec_from_file_location("pm", "scripts/03_perturbation.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
engine = mod.PerturbationEngine(random_seed=42)

indices = engine.select_random_samples(trial, num_points=1)   # trial: (n_channels, n_times)
perturbed = engine.apply_perturbation(trial, indices, magnitude_uv=1.0)
metrics = engine.verify_imperceptibility(trial, perturbed)
# metrics: {"snr_db": float, "imperceptible": bool, "l2_norm": float, "max_delta_uv": float}
```

---

## Experiment Loop (Week 2–3)

150 signals × 5 magnitudes × 2 attack types = 1,500 experiments. Results written to `results/experiment_log.csv` with columns: `trial_id, n_points, magnitude_uv, orig_pred, pert_pred, misclassified, snr_db`.

---

## Common Errors

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: moabb` | `pip install moabb` |
| `FileNotFoundError: X_raw.npy` | Run `01_moabb_setup.py` first |
| `FileNotFoundError: model.pkl` | Run `02_load_pretrained.py` first |
| `ValueError: n_components > ...` | Add more subjects to `SUBJECTS` in script 01 (109 available) |
| Accuracy < 55% | Expand `SUBJECTS`; default is only 3 subjects |
| `SNR = inf` | Perturbation was exactly 0 — sign cancellation bug in `apply_perturbation` |
| CSP convergence warning | Normal for small subject subsets; accuracy still valid |
