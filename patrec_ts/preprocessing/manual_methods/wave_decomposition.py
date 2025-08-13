from typing import Optional, Union, List
import time

import numpy as np
import matplotlib.pyplot as plt
import pywt

from patrec_ts.feature_extraction.fe_classes import DecompositionResult, FEComponentType


# from tslearn.shapelets import LearningShapelets


class FFT_Decomposer:
    def __init__(self, signal, fs=None, pad_factor=None, window_type=None):
        """
        Initialize FFT decomposer with a signal.
        
        Parameters:
        signal: Input time-series signal
        fs: Sampling frequency (optional)
        pad_factor: Zero-padding factor (optional)
        window_type: Window function type (optional)
        """
        self.original_signal = signal.copy()
        self.fs = fs if fs is not None else len(signal)
        self.pad_factor = pad_factor
        self.window_type = window_type
        
        self.signal = signal.copy()
        
        # Preprocessing
        if pad_factor is not None:
            self.signal = self._pad_signal(signal)
        if window_type is not None:
            self.signal = self._apply_window(self.signal)
        
        # Compute FFT
        self.fft_signal, self.freqs, self.magnitudes = self._compute_fft(self.signal)
        
        # Positive frequencies
        self.pos_freqs, self.pos_mags = self._get_positive_frequencies(self.freqs, self.magnitudes)
        
        # Sorted by power
        self.sorted_freqs, self.sorted_mags = self._sort_by_power(self.pos_freqs, self.pos_mags)
    
    def _compute_fft(self, signal, n=None):
        """Compute FFT and return all frequencies and magnitudes."""
        if n is None:
            n = len(signal)
        fft_signal = np.fft.fft(signal, n)
        return fft_signal, np.fft.fftfreq(n, d=1/self.fs), np.abs(fft_signal)
    
    def _get_positive_frequencies(self, freqs, magnitudes):
        """Return only positive frequencies and their magnitudes."""
        pos_mask = freqs >= 0
        return freqs[pos_mask], magnitudes[pos_mask]
    
    def _sort_by_power(self, freqs, magnitudes):
        """Sort frequencies by their power (magnitude squared)."""
        power = magnitudes ** 2
        sorted_idx = np.argsort(power)[::-1]
        return freqs[sorted_idx], magnitudes[sorted_idx]
    
    def _apply_window(self, signal):
        """Apply window function to reduce edge effects"""
        window = getattr(np, self.window_type)(len(signal))
        return signal * window
    
    def _pad_signal(self, signal):
        """Add zeros to both ends before FFT"""
        return np.pad(signal, self.pad_factor, mode='constant')
    
    def _trim_signal(self, signal):
        """Trim padded signal after reconstruction"""
        return signal[self.pad_factor:-self.pad_factor]
    
    def filter_signal(self, threshold=None, freq_range=None, exact_freqs=None, mode='highpass'):
        """
        Filter signal with proper frequency handling.
        
        Parameters:
        threshold: Magnitude threshold
        freq_range: Tuple (min_freq, max_freq) to keep
        exact_freqs: List of exact frequencies to keep
        mode: 'highpass' or 'lowpass' (for threshold only)
        """
        mask = np.ones_like(self.fft_signal, dtype=bool)
        
        # Validate inputs
        if sum(x is not None for x in [threshold, freq_range, exact_freqs]) > 1:
            raise ValueError("Can only specify one filtering method at a time")
        
        # Threshold filtering
        if threshold is not None:
            if mode == 'highpass':
                mask = self.magnitudes > threshold
            elif mode == 'lowpass':
                mask = self.magnitudes < threshold
            else:
                raise ValueError("Mode must be 'highpass' or 'lowpass'")
        
        # Frequency range filtering - FIXED
        elif freq_range is not None:
            min_freq, max_freq = freq_range
            mask = np.zeros_like(self.fft_signal, dtype=bool)
            
            # Find indices within frequency range (positive and negative)
            pos_mask = (self.freqs >= min_freq) & (self.freqs <= max_freq)
            neg_mask = (self.freqs <= -min_freq) & (self.freqs >= -max_freq)
            
            # Handle DC component (0Hz) specially
            if min_freq == 0:
                pos_mask |= (self.freqs == 0)
            
            mask[pos_mask | neg_mask] = True
        
        # Exact frequencies filtering
        elif exact_freqs is not None:
            mask = np.zeros_like(self.fft_signal, dtype=bool)
            for freq in exact_freqs:
                # Find closest frequency bins
                idx = np.argmin(np.abs(self.freqs - freq))
                mask[idx] = True
                # Keep negative frequency unless it's DC
                if freq != 0:
                    neg_idx = np.argmin(np.abs(self.freqs - (-freq)))
                    mask[neg_idx] = True
        
        filtered_signal = np.fft.ifft(self.fft_signal * mask).real
        return self._trim_signal(filtered_signal) if self.pad_factor is not None else filtered_signal

    def plot_frequency_components(self, min_freq=None, max_freq=None, x_tick_step=1, show=True):
        """
        Plot frequency components with increased spacing between x-axis points.
        
        Parameters:
            min_freq (float): Minimum frequency to display (default: 0)
            max_freq (float): Maximum frequency to display (default: max of positive frequencies)
            x_tick_step (int or float): Step size for x-axis ticks (in Hz)
            show (bool): Whether to show the plot immediately
        """
        if min_freq is None:
            min_freq = 0
        if max_freq is None:
            max_freq = np.max(self.pos_freqs)

        fig, ax = plt.subplots(figsize=(12, 5))  # Wider figure for readability
        
        ax.stem(self.pos_freqs, self.pos_mags, linefmt='b-', markerfmt=' ', basefmt=' ')

        # Calculate statistics within visible range
        freq_mask = (self.pos_freqs >= min_freq) & (self.pos_freqs <= max_freq)
        mean_mag = np.mean(self.pos_mags[freq_mask])
        median_mag = np.median(self.pos_mags[freq_mask])

        ax.axhline(mean_mag, color='r', linestyle='--', label=f'Mean [{min_freq}-{max_freq}] = {mean_mag:.2f}')
        ax.axhline(median_mag, color='g', linestyle='--', label=f'Median [{min_freq}-{max_freq}] = {median_mag:.2f}')

        ax.set_title("Frequency Components (Sorted by Significance)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.legend()
        ax.set_xlim(min_freq, max_freq)

        # # Set x-axis ticks at specified step intervals
        tick_values = np.arange(0, self.pos_freqs.shape[0], x_tick_step)
        ax.set_xticks(tick_values, self.pos_freqs)

        # Optional: rotate labels for better visibility
        plt.xticks(rotation=90, fontsize=7) 

        if show:
            plt.tight_layout()
            plt.show()
            return None
        return fig, ax
    
    def plot_comparison(self, filtered_signal, title='Original vs Filtered', show=True):
        """Plot original vs filtered signals.
        
        Returns:
        fig, ax matplotlib objects if show=False, otherwise shows plot and returns None
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.original_signal, label='Original')
        ax.plot(filtered_signal, label='Filtered')
        ax.set_title(title)
        ax.legend()
        
        if show:
            plt.show()
            return None
        return fig, ax


class DWT_wave_decomposer(BaseDecomposer):
    """Seasonality decomposition using Discrete Wavelet Transform."""
    
    def __init__(self, wavelet: str = 'db8', mode: str = 'symmetric',
                 level: Union[int, str] = 'auto', threshold: Union[float, str] = 'universal',
                 seasonality_levels: Optional[List[int]] = None):
        """
        Args:
            wavelet: Wavelet type ('db8', 'sym5', etc.)
            mode: Signal extension mode ('symmetric', 'zero', etc.)
            level: Decomposition level ('auto' or integer)
            threshold: Threshold value or method ('universal')
            seasonality_levels: Which wavelet levels contain seasonality
        """
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        self.threshold = threshold
        self.seasonality_levels = seasonality_levels
        
    def decompose(self, data: np.ndarray) -> DecompositionResult:
        start_time = time.time()
        n = len(data)
        
        # Auto-select decomposition level
        if self.level == 'auto':
            level = pywt.dwt_max_level(n, self.wavelet)
        else:
            level = self.level
            
        # Wavelet decomposition
        coeffs = pywt.wavedec(data, self.wavelet, level=level, mode=self.mode)
        
        # Auto-select threshold
        if self.threshold == 'universal':
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(n))
        else:
            threshold = self.threshold
            
        # Determine which levels contain seasonality
        if self.seasonality_levels is None:
            seasonality_levels = list(range(1, len(coeffs)))  # Exclude approximation
        else:
            seasonality_levels = self.seasonality_levels
            
        # Process coefficients
        seasonal_coeffs = []
        for i, c in enumerate(coeffs):
            if i in seasonality_levels:
                seasonal_coeffs.append(pywt.threshold(c, threshold, 'soft'))
            else:
                seasonal_coeffs.append(np.zeros_like(c))
        
        # Reconstruct seasonal component
        seasonal_coeffs[0] = np.zeros_like(coeffs[0])  # Ignore approximation
        seasonal = pywt.waverec(seasonal_coeffs, self.wavelet)
        seasonal = seasonal[:n]  # Trim to original length
        
        return DecompositionResult(
            component=seasonal,
            component_type=FEComponentType.SEASONAL,
            method_name='dwt_seasonality',
            params={
                'wavelet': self.wavelet,
                'level': level,
                'threshold': threshold,
                'seasonality_levels': seasonality_levels,
                'mode': self.mode
            },
            stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            }
        )


# class Shapelet_wave_decomposer(BaseDecomposer):
#     """Shapelet-based pattern decomposition (requires labeled data)."""
    
#     def __init__(self, n_shapelets: int = 5, min_len: int = 10,
#                  max_len: int = 50, max_iter: int = 100):
#         """
#         Args:
#             n_shapelets: Number of shapelets to learn
#             min_len: Minimum shapelet length
#             max_len: Maximum shapelet length
#             max_iter: Maximum training iterations
#         """
#         self.n_shapelets = n_shapelets
#         self.len_bounds = (min_len, max_len)
#         self.max_iter = max_iter
        
#     def decompose(self, data: np.ndarray, y: np.ndarray) -> DecompositionResult:
#         """
#         Args:
#             data: Input time series (1D array)
#             y: Class labels for supervised shapelet learning
            
#         Returns:
#             DecompositionResult containing shapelet distances
#         """
#         start_time = time.time()
        
#         # Initialize and fit shapelet model
#         model = LearningShapelets(
#             n_shapelets=self.n_shapelets,
#             max_iter=self.max_iter,
#             verbose=0
#         )
#         model.fit(data.reshape(1, -1, 1), y)  # Requires 3D input
        
#         # Transform data to shapelet distance space
#         distances = model.transform(data.reshape(1, -1, 1))
        
#         return DecompositionResult(
#             component=distances.flatten(),
#             component_type=FEComponentType.SEASONAL,
#             method_name='shapelet_decomposition',
#             params={
#                 'n_shapelets': self.n_shapelets,
#                 'length_bounds': self.len_bounds,
#                 'max_iter': self.max_iter
#             },
#             stats={
#                 'execution_time_sec': time.time() - start_time,
#                 'input_shape': data.shape
#             }
#         )

#### raw fft #### 
# import numpy as np
# import matplotlib.pyplot as plt

# def compute_fft(signal, fs=None, n=None):
#     """Compute FFT and return all frequencies and magnitudes."""
#     if fs is None:
#         fs = len(signal)
#     if n is None:
#         n = len(signal)

#     fft_signal = np.fft.fft(signal, n)
#     return fft_signal, np.fft.fftfreq(n, d=1/fs), np.abs(fft_signal)

# def get_positive_frequencies(freqs, magnitudes):
#     """Return only positive frequencies and their magnitudes."""
#     pos_mask = freqs >= 0
#     return freqs[pos_mask], magnitudes[pos_mask]

# def sort_by_power(freqs, magnitudes):
#     """Sort frequencies by their power (magnitude squared)."""
#     power = magnitudes ** 2
#     sorted_idx = np.argsort(power)[::-1]
#     return freqs[sorted_idx], magnitudes[sorted_idx]

# def plot_frequency_components(freqs, magnitudes, min_freq=None, max_freq=None):
#     """Plot frequency components with mean/median lines."""
    
#     if min_freq is None:
#         min_freq = 0 
#     if max_freq is None:
#         max_freq = np.max(freqs)
        
#     plt.figure(figsize=(10, 5))
#     plt.stem(freqs, magnitudes, linefmt='b-', markerfmt=' ', basefmt=' ')
    
#     mean_mag = np.mean(magnitudes[min_freq:max_freq])
#     median_mag = np.median(magnitudes[min_freq:max_freq])
    
#     plt.axhline(mean_mag, color='r', linestyle='--', label=f'Mean[{min_freq}:{max_freq}] = {mean_mag:.2f}')
#     plt.axhline(median_mag, color='g', linestyle='--', label=f'Median[{min_freq}:{max_freq}] = {median_mag:.2f}')
    
#     plt.title("Frequency Components (Sorted by Significance)")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Magnitude")
#     plt.legend()

#     plt.xlim(min_freq, max_freq)
#     plt.show()

# def filter_signal(fft_signal, magnitudes, threshold=None, inxs=None, mode='highpass'):
#     """
#     Filter signal in frequency domain.
    
#     Parameters:
#     fft_signal: FFT of the original signal (complex values)
#     magnitudes: Magnitudes of the FFT (absolute values)
#     threshold: Threshold value for magnitude filtering
#     inxs: Custom indices to keep (array of indices)
#     mode: 'highpass' (keep above threshold) or 'lowpass' (keep below) - only used with threshold
    
#     Returns:
#     Filtered time-domain signal (real part)
#     """
#     mask = np.ones_like(fft_signal, dtype=bool)  # Default: keep all frequencies
    
#     if threshold is not None and inxs is not None:
#         raise ValueError("Cannot specify both threshold and inxs - choose one filtering method")
    
#     if threshold is not None:    
#         if mode == 'highpass':
#             mask = magnitudes > threshold
#         elif mode == 'lowpass':
#             mask = magnitudes < threshold
#         else:
#             raise ValueError("Mode must be 'highpass' or 'lowpass'")
    
#     if inxs is not None:
#         # Only keep specified indices
#         mask = np.zeros_like(fft_signal, dtype=bool)
#         mask[inxs] = True  
    
#     return np.fft.ifft(fft_signal * mask).real


# def apply_window(signal, window_type='hanning'):
#     """Apply window function to reduce edge effects"""
#     window = getattr(np, window_type)(len(signal))
#     return signal * window

# def pad_signal(signal, pad_factor=2):
#     """Add zeros to both ends before FFT"""
#     return np.pad(signal, pad_factor, mode='constant')

# def trim_signal(signal, pad_factor=2):
#     return signal[pad_factor:-pad_factor]

# # Remember to trim after reconstruction
# x1 = data.copy()
# pad_val = len(x1)//2
# x = pad_signal(x1, pad_val)

# # 1. Compute FFT (keeping all frequencies)
# fft_decomp, all_freqs, all_mags = compute_fft(x)

# # 2. Get positive frequencies for visualization
# pos_freqs, pos_mags = get_positive_frequencies(all_freqs, all_mags)

# # 3. Sort by power
# sorted_freqs, sorted_mags = sort_by_power(pos_freqs, pos_mags)

# # 4. Plot with mean/median lines
# plot_frequency_components(sorted_freqs, sorted_mags, max_freq=100, min_freq=0)

# # 5. Filter signal (choose one)
# # threshold = all_mags[20]
# inxs = np.asarray(list(range(35, 45))) 
# # threshold = np.median(all_mags) 
# threshold = np.mean(pos_freqs) 
# filter_mode = 'highpass' 
# # filter_mode = 'lowpass'

# # filtered_signal = filter_signal(fft_decomp, all_mags, threshold=None, inxs=inxs, mode=filter_mode)
# filtered_signal = filter_signal(fft_decomp, all_mags, threshold=threshold, inxs=None, mode=filter_mode)
# result_signal = trim_signal(filtered_signal, pad_val)

# # Plot original vs filtered
# plt.figure(figsize=(10, 4))
# plt.plot(x1, label='Original')
# # plt.plot(x1 - result_signal, label='New')
# plt.plot(result_signal, label=f'Filtered ({filter_mode})')
# plt.legend()
# plt.show()