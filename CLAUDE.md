# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**foolingAI-547** — Vulnerability assessment of EEG-based Brain-Computer Interfaces (BCIs).  
Research question: How drastically do EEG classifications change when only 1–2 sample points are modified?  
Expected finding: 1–2 sample point perturbations at 1–10 µV cause dramatic misclassification despite being imperceptible (SNR > 20 dB).

---

## Dataset Provenance: High-Gamma Dataset (Schirrmeister et al., 2017)

**Source:**
- **Dataset:** High-Gamma Dataset (Schirrmeister et al., 2017)
- **Paper:** *Deep learning with convolutional neural networks for EEG decoding and visualization*, Human Brain Mapping, 38(11), 5391–5420
- **Repository:** G-Node GIN (German Neuroinformatics Node)
- **Raw format:** `.edf` files, 128-channel high-density EEG, 250 Hz, 4-class motor imagery (Left Hand, Right Hand, Feet, Rest), 14 total subjects

**Custom Preprocessing Pipeline:**
- **Trial selection:** 300 trials sequentially extracted from Subject 1 and Subject 2 only
- **Spatial downsampling:** Raw 128-channel montage reduced to 22-channel standard 10-20 layout to match BCI Competition IV 2a baseline
- **Temporal padding:** Raw 4-second epochs (1000 time points) zero-padded to 1001 time points for shape uniformity
- **Final shape:** `(300, 22, 1001)` → `FINAL_dataset_547_data_300.npy`
- **Labels:** `NEW_dataset_547_labels_300.npy` (string labels: feet/left_hand/rest/right_hand)

**Why 89.5% accuracy:** Data from only 2 subjects makes this effectively a within-subject task. Cross-subject (all 14) would yield ~50–60%. This must be stated explicitly in the paper methods section.

**Citation:** Schirrmeister, R.T., Springenberg, J.T., Fiederer, L.D.J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F., Burgard, W., & Ball, T. (2017).

---

## Run Commands

Always run from project root (`foolingAI-547/`). Scripts run in order:

```bash
pip install -r requirements.txt

python scripts/02_load_pretrained.py   # Trains FBCSP+SVM, saves artefacts to results/
python scripts/03_perturbation.py      # Tests PerturbationEngine (standalone)
python scripts/04_integration_test.py  # End-to-end pipeline check (random + targeted @ 1µV)
python scripts/05_run_experiments.py   # Full sweep: all trials × magnitudes × attack types → experiment_log.csv
python scripts/06_plot_results.py      # Reads experiment_log.csv, writes results/figures/*.png
```

---

## Architecture & Data Flow

```
data/raw/FINAL_dataset_547_data_300.npy   (µV, rescaled to V on load)
data/raw/NEW_dataset_547_labels_300.npy   (string labels: feet/left_hand/rest/right_hand)
    ▼
02_load_pretrained.py
    [filter to 3 classes → 75/25 stratified split → FBCSP + StandardScaler + RBF-SVM fit]
    →  results/model.pkl, X_test.npy, y_test.npy          (baseline accuracy ~89%)
    ▼
03_perturbation.py    PerturbationEngine / TargetedPerturbationEngine
04_integration_test.py  ←── dynamically imports from 03_perturbation.py
    [random attack @ 1µV + magnitude sweep random vs targeted]
    ▼
05_run_experiments.py
    [51 correctly-classified trials × [5,10,15,20]% of peak-to-peak × [random,targeted] × [1,2] points]
    →  results/experiment_log.csv  (1,224 rows)
    ▼
06_plot_results.py  →  results/figures/fig1_misclassification_rate.png
                        results/figures/fig2_snr.png
                        results/figures/fig3_heatmap.png
```

Scripts 02–06 all do `sys.path.insert(0, scripts/)` and `from fbcsp import FilterBankCSP` — required so pickle can resolve `model.pkl`.  
Scripts 04/05 use `importlib.util` to load `PerturbationEngine` from `03_perturbation.py` because `scripts/` has no `__init__.py`.

**Model:** FBCSP (6 bands × 6 components = 36 features) → StandardScaler → RBF-SVM (C=100).  
**Classes:** `feet=0, left_hand=1, rest=2` (right_hand excluded — poor separability).  
**Signal units:** dataset stored in µV; loaded as V via `× 1e-6`. All perturbation magnitudes in µV, converted internally.

---

## Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| `RANDOM_SEED` | `42` | All scripts |
| `TEST_SIZE` | `0.25` | `02_load_pretrained.py` |
| `FREQ_BANDS` | `[(4,8),(8,12),(12,16),(16,20),(20,24),(24,30)]` | `fbcsp.py` |
| `CSP_COMPONENTS_PER_BAND` | `6` | `fbcsp.py` |
| `PERTURBATION_PCTS` | `[5.0, 10.0, 15.0, 20.0]` % of peak-to-peak | `05_run_experiments.py`, `06_plot_results.py` |
| `MAGNITUDE_SWEEP_UV` | `[0.5, 1.0, 2.0, 5.0, 10.0]` µV | `03_perturbation.py` (script 04 only) |
| `SNR_THRESHOLD_DB` | `20.0` dB | `03_perturbation.py` |
| `EXPECTED_ACCURACY_MIN` | `0.60` | `02_load_pretrained.py` |
| `PERTURBATION_MAGNITUDE_UV` | `1.0` µV | `04_integration_test.py` |
| `N_POINTS` | `1` | `04_integration_test.py` |

---

## PerturbationEngine API

```python
# Dynamic import (as used in scripts 04/05)
import importlib.util
spec = importlib.util.spec_from_file_location("pm", "scripts/03_perturbation.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
engine = mod.PerturbationEngine(random_seed=42)

indices = engine.select_random_samples(trial, num_points=1)   # trial: (n_channels, n_times)
perturbed = engine.apply_perturbation(trial, indices, magnitude_uv=1.0)
metrics = engine.verify_imperceptibility(trial, perturbed)
# metrics: {"snr_db": float, "imperceptible": bool, "l2_norm": float, "max_delta_uv": float}

# Targeted attack
tengine = mod.TargetedPerturbationEngine(random_seed=42)
perturbed = tengine.apply_targeted_perturbation(trial, model, magnitude_uv=1.0, num_points=1)
```

---

## Experiment Grid

`51 correctly-classified trials × 4 magnitudes × 2 attack types × 2 n_points = 1,224 experiments`.  
Knob: `PERTURBATION_PCTS = [5.0, 10.0, 15.0, 20.0]` — percentage of each trial's peak-to-peak amplitude.  
Results in `results/experiment_log.csv`: `trial_id, attack_type, n_points, perturbation_pct, magnitude_uv, orig_pred, pert_pred, misclassified, snr_db, trial_range_uv`.

---

## Common Errors

| Error | Fix |
|-------|-----|
| `FileNotFoundError: model.pkl` | Run `02_load_pretrained.py` first |
| `FileNotFoundError: experiment_log.csv` | Run `05_run_experiments.py` first |
| `AttributeError: FilterBankCSP` on pickle load | Ensure `from fbcsp import FilterBankCSP` is at top of script |
| `KeyError: 'lda'` | Pipeline uses `'svm'` step, not `'lda'` |
| `SNR = inf` | Perturbation was exactly 0 — sign cancellation in `apply_perturbation` |
| Accuracy < 60% | Check dataset path; confirm µV→V rescaling (`× 1e-6`) applied |
