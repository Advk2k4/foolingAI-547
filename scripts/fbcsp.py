"""
fbcsp.py - foolingAI-547
FilterBankCSP transformer — shared module imported by scripts 02, 03, and 04.
Keeping it here ensures pickle can resolve the class when loading model.pkl.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin

SFREQ = 250.0
FREQ_BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]
CSP_COMPONENTS_PER_BAND = 6


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Filter Bank CSP: applies CSP independently across multiple frequency bands
    and concatenates log-variance features.

    Each band produces n_components features; outputs are concatenated into a
    single feature vector of length (n_bands * n_components).

    Args:
        freq_bands: List of (low_hz, high_hz) tuples.
        n_components: CSP components per band.
        sfreq: Sampling frequency in Hz.
        reg: CSP covariance regularisation (passed to MNE CSP).
    """

    def __init__(self, freq_bands=None, n_components=CSP_COMPONENTS_PER_BAND,
                 sfreq=SFREQ, reg="ledoit_wolf"):
        self.freq_bands = freq_bands if freq_bands is not None else FREQ_BANDS
        self.n_components = n_components
        self.sfreq = sfreq
        self.reg = reg

    def _bandpass(self, X, low, high):
        sos = butter(4, [low, high], btype="bandpass", fs=self.sfreq, output="sos")
        return sosfiltfilt(sos, X, axis=-1)

    def fit(self, X, y):
        from mne.decoding import CSP
        self.csps_ = []
        for low, high in self.freq_bands:
            X_band = self._bandpass(X, low, high)
            csp = CSP(n_components=self.n_components, reg=self.reg,
                      log=True, norm_trace=False)
            csp.fit(X_band, y)
            self.csps_.append(csp)
        return self

    def transform(self, X):
        features = []
        for (low, high), csp in zip(self.freq_bands, self.csps_):
            X_band = self._bandpass(X, low, high)
            features.append(csp.transform(X_band))
        return np.concatenate(features, axis=1)
