"""
03_perturbation.py - foolingAI-547
PerturbationEngine: select sample points, apply µV-scale perturbations,
and verify imperceptibility via SNR.
Run from project root: python scripts/03_perturbation.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fbcsp import FilterBankCSP  # noqa: F401 — required for pickle to resolve model.pkl

RANDOM_SEED = 42
MAGNITUDE_SWEEP_UV = [0.5, 1.0, 2.0, 5.0, 10.0]  # µV
SNR_THRESHOLD_DB = 20.0  # Perturbation is considered imperceptible above this


class PerturbationEngine:
    """Applies minimal adversarial perturbations to EEG signals.

    All magnitudes are specified in microvolts (µV). EEG signals loaded via
    MOABB / MNE are in volts (V), so magnitudes are converted internally.

    Args:
        random_seed: Seed for reproducible sample selection.
    """

    UV_TO_V = 1e-6  # Conversion factor: µV → V

    def __init__(self, random_seed: int = RANDOM_SEED):
        self.rng = np.random.default_rng(random_seed)

    def select_random_samples(self, signal: np.ndarray, num_points: int = 1) -> np.ndarray:
        """Randomly select time-point indices to perturb.

        Args:
            signal: 1-D EEG signal (n_times,) or 2-D (n_channels, n_times).
                    Selection is over the time axis.
            num_points: Number of sample points to select (1 or 2).

        Returns:
            np.ndarray of selected time-point indices, shape (num_points,).

        Raises:
            ValueError: If num_points > signal length.
        """
        n_times = signal.shape[-1]
        if num_points > n_times:
            raise ValueError(
                f"num_points ({num_points}) exceeds signal length ({n_times})"
            )
        indices = self.rng.choice(n_times, size=num_points, replace=False)
        return indices

    def apply_perturbation(
        self,
        signal: np.ndarray,
        indices: np.ndarray,
        magnitude_uv: float,
        channel: int | None = None,
    ) -> np.ndarray:
        """Apply additive perturbation at specified time points.

        Perturbation values are drawn from ±magnitude_uv uniformly at random,
        then added to the signal at the selected indices.

        Args:
            signal: EEG array. Shape (n_times,) or (n_channels, n_times).
            indices: Time-point indices returned by select_random_samples().
            magnitude_uv: Perturbation magnitude in µV.
            channel: For 2-D signals, which channel to perturb.
                     If None, all channels are perturbed at the selected indices.

        Returns:
            Perturbed copy of signal (same shape, same dtype).
        """
        perturbed = signal.copy().astype(np.float64)
        delta_v = magnitude_uv * self.UV_TO_V

        # Generate signed perturbation values (±delta)
        signs = self.rng.choice([-1.0, 1.0], size=len(indices))
        deltas = signs * delta_v

        if perturbed.ndim == 1:
            perturbed[indices] += deltas
        elif perturbed.ndim == 2:
            if channel is not None:
                perturbed[channel, indices] += deltas
            else:
                # Perturb all channels at these time points
                perturbed[:, indices] += deltas[np.newaxis, :]
        else:
            raise ValueError(f"signal must be 1-D or 2-D, got {perturbed.ndim}-D")

        return perturbed

    def verify_imperceptibility(
        self, original: np.ndarray, perturbed: np.ndarray
    ) -> dict:
        """Compute SNR and assess imperceptibility.

        SNR = 10 * log10(power(signal) / power(noise))
        where noise = perturbed - original.

        Args:
            original: Original EEG signal.
            perturbed: Perturbed EEG signal (same shape).

        Returns:
            dict with keys:
                snr_db       (float)  — signal-to-noise ratio
                imperceptible (bool)  — True if SNR > SNR_THRESHOLD_DB
                l2_norm       (float) — ||perturbation||_2 in volts
                max_delta_uv  (float) — max abs perturbation in µV
        """
        noise = perturbed - original
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power == 0:
            snr_db = float("inf")
        else:
            snr_db = 10 * np.log10(signal_power / noise_power)

        return {
            "snr_db": snr_db,
            "imperceptible": snr_db > SNR_THRESHOLD_DB,
            "l2_norm": float(np.linalg.norm(noise)),
            "max_delta_uv": float(np.max(np.abs(noise)) / self.UV_TO_V),
        }


class TargetedPerturbationEngine(PerturbationEngine):
    """Gradient-based targeted adversarial attack for CSP+LDA pipelines.

    Computes the analytic gradient of the LDA decision score with respect to
    each (channel, time) input sample, selects the time point with the largest
    gradient norm, and perturbs in the direction that flips the classification.

    Math:
        CSP computes Z = filters_ @ trial, then v_i = mean(Z[i,:]^2), f_i = log(v_i).
        LDA score = w_lda @ f + b.
        Gradient: grad = filters_.T @ diag(2*w_lda / (v*T)) @ Z   shape (n_ch, n_t)
        Best time point: t* = argmax_t ||grad[:, t]||_2
        Perturbation at t*: -sign(score) * magnitude * grad[:,t*] / ||grad[:,t*]||
    """

    def compute_gradient(self, trial: np.ndarray, pipeline) -> np.ndarray:
        """Return ∂(margin probability)/∂trial, shape (n_channels, n_times).

        Strategy:
          1. Numerical gradient of (proba[runner_up] - proba[predicted]) w.r.t.
             the 36-dimensional scaled FBCSP features — only 72 SVM evaluations.
          2. Backpropagate analytically through StandardScaler and FBCSP.
        """
        fbcsp  = pipeline.named_steps["fbcsp"]
        scaler = pipeline.named_steps["scaler"]
        svm    = pipeline.named_steps["svm"]

        feat_raw    = fbcsp.transform(trial[np.newaxis])       # (1, n_feat)
        feat_scaled = scaler.transform(feat_raw)               # (1, n_feat)

        proba       = svm.predict_proba(feat_scaled)[0]        # (n_classes,)
        ranked      = np.argsort(proba)
        predicted   = ranked[-1]
        runner_up   = ranked[-2]

        # Numerical gradient in scaled feature space (finite differences)
        eps = 1e-4
        n_feat = feat_scaled.shape[1]
        grad_feat_scaled = np.zeros(n_feat)
        for i in range(n_feat):
            fp = feat_scaled.copy(); fp[0, i] += eps
            fm = feat_scaled.copy(); fm[0, i] -= eps
            pp = svm.predict_proba(fp)[0]
            pm = svm.predict_proba(fm)[0]
            grad_feat_scaled[i] = (
                (pp[runner_up] - pm[runner_up]) - (pp[predicted] - pm[predicted])
            ) / (2 * eps)

        # Backpropagate through StandardScaler: grad_raw = grad_scaled / scale
        grad_feat_raw = grad_feat_scaled / scaler.scale_       # (n_feat,)

        # Backpropagate analytically through FBCSP (sum across bands)
        n_comp = fbcsp.n_components
        grad_trial = np.zeros_like(trial, dtype=np.float64)

        for band_idx, ((low, high), csp) in enumerate(
            zip(fbcsp.freq_bands, fbcsp.csps_)
        ):
            X_band = fbcsp._bandpass(trial[np.newaxis], low, high)[0]
            W = csp.filters_[:n_comp, :]                       # (n_comp, n_channels)
            Z = W @ X_band                                     # (n_comp, n_times)
            T = Z.shape[1]
            v = (Z ** 2).mean(axis=1)                          # (n_comp,)
            w_k = grad_feat_raw[band_idx * n_comp: (band_idx + 1) * n_comp]
            alpha = 2.0 * w_k / (v * T)
            grad_trial += W.T @ (alpha[:, np.newaxis] * Z)

        return grad_trial

    def apply_targeted_perturbation(
        self,
        trial: np.ndarray,
        pipeline,
        magnitude_uv: float,
        num_points: int = 1,
    ) -> np.ndarray:
        """Perturb the highest-impact time point(s) to flip the LDA decision.

        Args:
            trial: EEG array (n_channels, n_times), already bandpass-filtered.
            pipeline: Fitted sklearn Pipeline with 'csp' and 'lda' named steps.
            magnitude_uv: Perturbation magnitude in µV.
            num_points: Number of time points to perturb.

        Returns:
            Perturbed copy of trial (same shape).
        """
        grad = self.compute_gradient(trial, pipeline)

        importance = np.linalg.norm(grad, axis=0)          # (n_times,)
        top_indices = np.argsort(importance)[-num_points:]  # highest-impact time points

        # Gradient already points toward runner-up; always add it (flip_sign = +1)
        flip_sign = 1.0

        perturbed = trial.copy().astype(np.float64)
        delta_v = magnitude_uv * self.UV_TO_V

        for t in top_indices:
            direction = flip_sign * grad[:, t]
            norm = np.linalg.norm(direction)
            if norm > 0:
                perturbed[:, t] += delta_v * direction / norm

        return perturbed


# ---------------------------------------------------------------------------
# Standalone tests
# ---------------------------------------------------------------------------

def _make_synthetic_signal(n_channels=64, n_times=321, seed=0):
    """Generate a synthetic EEG-like signal (µV-scale, bandlimited noise)."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(loc=0, scale=20e-6, size=(n_channels, n_times))
    return signal


def test_1d_perturbation(engine: PerturbationEngine):
    """Test 1-point and 2-point perturbations on a 1-D signal."""
    print("\n[Test] 1-D signal perturbation")
    signal_1d = np.sin(np.linspace(0, 4 * np.pi, 321)) * 50e-6  # 50 µV sine

    for n_pts in (1, 2):
        for mag in (1.0, 5.0):
            idx = engine.select_random_samples(signal_1d, num_points=n_pts)
            perturbed = engine.apply_perturbation(signal_1d, idx, magnitude_uv=mag)
            metrics = engine.verify_imperceptibility(signal_1d, perturbed)
            status = "✓" if metrics["imperceptible"] else "✗"
            print(
                f"  n_pts={n_pts}, mag={mag}µV | "
                f"SNR={metrics['snr_db']:.1f} dB | "
                f"Imperceptible: {status}"
            )


def test_2d_perturbation(engine: PerturbationEngine):
    """Test perturbation on a 2-D (channels × times) signal."""
    print("\n[Test] 2-D signal perturbation")
    signal_2d = _make_synthetic_signal()

    for n_pts in (1, 2):
        idx = engine.select_random_samples(signal_2d, num_points=n_pts)
        perturbed = engine.apply_perturbation(signal_2d, idx, magnitude_uv=1.0)
        metrics = engine.verify_imperceptibility(signal_2d, perturbed)
        status = "✓" if metrics["imperceptible"] else "✗"
        print(
            f"  n_pts={n_pts}, mag=1µV | "
            f"SNR={metrics['snr_db']:.1f} dB | "
            f"max Δ={metrics['max_delta_uv']:.3f}µV | "
            f"Imperceptible: {status}"
        )


def test_magnitude_sweep(engine: PerturbationEngine):
    """Run the full magnitude sweep on a synthetic signal."""
    print("\n[Test] Magnitude sweep")
    signal = _make_synthetic_signal()[0]  # Single channel

    print(f"  {'Magnitude (µV)':<18} {'SNR (dB)':<12} {'Imperceptible'}")
    print("  " + "-" * 42)
    for mag in MAGNITUDE_SWEEP_UV:
        idx = engine.select_random_samples(signal, num_points=1)
        perturbed = engine.apply_perturbation(signal, idx, magnitude_uv=mag)
        m = engine.verify_imperceptibility(signal, perturbed)
        flag = "✓" if m["imperceptible"] else "✗"
        print(f"  {mag:<18} {m['snr_db']:<12.2f} {flag}")


def test_real_eeg(engine: PerturbationEngine):
    """Test on real EEG data from data/raw/ if available."""
    x_path = Path("data/raw/X_raw.npy")
    if not x_path.exists():
        print("\n[Test] Real EEG: skipped (run 01_moabb_setup.py first)")
        return

    print("\n[Test] Real EEG signal perturbation")
    X = np.load(x_path)
    trial = X[0]  # First trial: (n_channels, n_times)

    idx = engine.select_random_samples(trial, num_points=1)
    perturbed = engine.apply_perturbation(trial, idx, magnitude_uv=1.0)
    metrics = engine.verify_imperceptibility(trial, perturbed)

    print(f"  Trial shape:    {trial.shape}")
    print(f"  Perturbed idx:  {idx}")
    print(f"  SNR:            {metrics['snr_db']:.2f} dB")
    print(f"  max Δ:          {metrics['max_delta_uv']:.4f} µV")
    status = "✓" if metrics["imperceptible"] else "✗"
    print(f"  Imperceptible:  {status}")


def test_targeted_vs_random(random_engine: PerturbationEngine):
    """Compare targeted vs random attack misclassification rates across magnitudes."""
    import pickle
    model_path = Path("results/model.pkl")
    x_path = Path("results/X_test.npy")

    if not model_path.exists() or not x_path.exists():
        print("\n[Test] Targeted vs random: skipped (run scripts 01-02 first)")
        return

    model = pickle.load(open(model_path, "rb"))
    X_test = np.load(x_path)
    targeted_engine = TargetedPerturbationEngine(random_seed=RANDOM_SEED)

    print("\n[Test] Targeted vs random attack — misclassification rate")
    print(f"  {'Magnitude (µV)':<16} {'Random':<12} {'Targeted'}")
    print("  " + "-" * 40)

    for mag in MAGNITUDE_SWEEP_UV:
        rand_mc = targeted_mc = 0
        for trial in X_test:
            orig = model.predict(trial[np.newaxis])[0]

            idx = random_engine.select_random_samples(trial, num_points=1)
            if model.predict(random_engine.apply_perturbation(trial, idx, mag)[np.newaxis])[0] != orig:
                rand_mc += 1

            tgt = targeted_engine.apply_targeted_perturbation(trial, model, mag)
            if model.predict(tgt[np.newaxis])[0] != orig:
                targeted_mc += 1

        n = len(X_test)
        print(f"  {mag:<16} {rand_mc/n*100:<11.1f}% {targeted_mc/n*100:.1f}%")


def run():
    np.random.seed(RANDOM_SEED)

    print("=" * 55)
    print("foolingAI-547  |  Script 03: Perturbation Framework")
    print("=" * 55)

    engine = PerturbationEngine(random_seed=RANDOM_SEED)

    test_1d_perturbation(engine)
    test_2d_perturbation(engine)
    test_magnitude_sweep(engine)
    test_real_eeg(engine)
    test_targeted_vs_random(engine)

    # Report summary using the canonical 1µV / 1-point case
    signal = _make_synthetic_signal()[0]
    idx = engine.select_random_samples(signal, num_points=1)
    perturbed = engine.apply_perturbation(signal, idx, magnitude_uv=1.0)
    m = engine.verify_imperceptibility(signal, perturbed)

    print("\nScript 03 complete.")
    print(
        f"Perturbation system tested. "
        f"SNR: {m['snr_db']:.1f} dB, "
        f"Imperceptible: {'True ✓' if m['imperceptible'] else 'False ✗'}"
    )
    return engine


if __name__ == "__main__":
    run()
