import time

import numpy as np

from patrec_ts.feature_extraction.fe_classes import DecompositionResult, FEComponentType
from patrec_ts.preprocessing.manual_methods.season.base_season_preprocessor import BaseSeasonPreprocessor


class SeasonDFTPreprocessor(BaseSeasonPreprocessor):
    """Seasonality decomposition using Discrete Fourier Transform."""

    def __init__(
            self,
            n_freqs: int | str = 'auto',
            fs: float = 1.0,
            min_power_ratio: float = 0.1,
            min_freq: float | None = None,
            max_freq: float | None = None
    ) -> None:
        """
        Args:
            n_freqs: Number of frequencies to keep ('auto' or integer)
            fs: Sampling frequency
            min_power_ratio: Minimum power ratio for auto frequency selection
            min_freq: Minimum frequency to consider
            max_freq: Maximum frequency to consider
        """
        super().__init__()

        self.n_freqs = n_freqs
        self.fs = fs
        self.min_power_ratio = min_power_ratio
        self.min_freq = min_freq
        self.max_freq = max_freq

    def decompose(self, data: np.ndarray, **kwargs) -> DecompositionResult:
        self._check_data_input(data)

        start_time = time.time()

        n = len(data)

        # Compute DFT
        dft = np.fft.fft(data)
        magnitude = np.abs(dft)
        power = magnitude ** 2
        freq = np.fft.fftfreq(n, d=1/self.fs)

        # Filter frequencies by specified range
        pos_mask = freq > 0
        if self.min_freq is not None:
            pos_mask &= (freq >= self.min_freq)
        if self.max_freq is not None:
            pos_mask &= (freq <= self.max_freq)

        pos_freq = freq[pos_mask]
        pos_power = power[pos_mask]

        # Auto-select number of frequencies
        if self.n_freqs == 'auto':
            total_power = np.sum(pos_power)
            sorted_idx = np.argsort(pos_power)[::-1]
            cum_power = np.cumsum(pos_power[sorted_idx]) / total_power
            n_freqs = np.argmax(cum_power > self.min_power_ratio) + 1
        else:
            n_freqs = min(self.n_freqs, len(pos_freq))

        # Select most significant frequencies
        significant_idx = np.argpartition(pos_power, -n_freqs)[-n_freqs:]

        # Reconstruct seasonal component
        reconstructed = np.zeros(n, dtype=complex)
        freq_vals = []
        for idx in np.where(pos_mask)[0][significant_idx]:
            freq_val = freq[idx]
            freq_vals.append(freq_val)
            reconstructed += (dft[idx] * np.exp(2j*np.pi*freq_val*np.arange(n)/n)) / n
            # Add symmetric component for real signals
            if freq_val != 0:
                reconstructed += (dft[-idx] * np.exp(-2j*np.pi*freq_val*np.arange(n)/n) / n)

        seasonal = np.real(reconstructed)

        return DecompositionResult(
            component=seasonal,
            component_type=FEComponentType.SEASONAL,
            method_name='dft_seasonality',
            params={
                'n_freqs': n_freqs,
                'fs': self.fs,
                'min_power_ratio': self.min_power_ratio,
                'min_freq': self.min_freq,
                'max_freq': self.max_freq,
                'freq_vals': freq_vals,
                'pos_power':pos_power,
            },
            stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            }
        )