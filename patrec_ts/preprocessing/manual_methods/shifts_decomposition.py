from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import time

import numpy as np

from scipy import signal, stats

import ruptures as rpt

from hmmlearn import hmm

from ..old_fe_classes import BaseDecomposer, DecompositionResult, FEComponentType



class Peaks_break_detector(BaseDecomposer):
    """
    Detects structural breaks by analyzing significant peaks in smoothed time series.
    
    Algorithm steps:
    1. Apply moving average smoothing
    2. Identify prominent peaks using scipy.find_peaks
    3. Calculate differences between consecutive peaks
    4. Return locations where differences exceed variance threshold
    
    Parameters:
        smoothing_window: Size of moving average window (must be odd)
        peak_prominence: Minimum peak prominence (relative to std dev)
    """
    
    def __init__(self, smoothing_window: int = 5, peak_prominence: float = 0.1):
        if smoothing_window % 2 == 0:
            raise ValueError("Smoothing window must be odd")
        self.smoothing_window = smoothing_window
        self.peak_prominence = peak_prominence
        
    def decompose(self, data: np.ndarray) -> DecompositionResult:
        start_time = time.time()
        
        # Apply moving average smoothing
        kernel = np.ones(self.smoothing_window)/self.smoothing_window
        smoothed = np.convolve(data, kernel, mode='same')
        
        # Detect peaks
        peaks, _ = signal.find_peaks(
            smoothed, 
            prominence=self.peak_prominence * np.std(smoothed)
        )
        
        # Calculate peak differences and find significant breaks
        peak_values = smoothed[peaks]
        peak_diffs = np.abs(np.diff(peak_values))
        threshold = self.peak_prominence * np.var(smoothed)
        significant_breaks = peaks[:-1][peak_diffs > threshold]
        
        return DecompositionResult(
            component=significant_breaks,
            component_type=FEComponentType.SHIFT,
            method_name='peaks_break_detector',
            params={
                'smoothing_window': self.smoothing_window,
                'peak_prominence': self.peak_prominence
            },
            stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            features={
                'peaks_indices': peaks,
                'peak_differences': peak_diffs,
                'num_peaks': len(peaks)
            }
        )


class ACF_break_detector(BaseDecomposer):
    """
    Detects structural breaks using autocorrelation function analysis.
    
    Method:
    1. Compute normalized autocorrelation function (ACF)
    2. Check ACF at specified lag against threshold
    3. If exceeded, locate abrupt changes in ACF as break points
    
    Parameters:
        lag: Time lag for ACF calculation
        threshold: Minimum ACF value to consider break
    """
    
    def __init__(self, lag: int = 5, threshold: float = 0.7):
        self.lag = lag
        self.threshold = threshold
        
    def decompose(self, data: np.ndarray) -> DecompositionResult:
        start_time = time.time()
        
        # Compute autocorrelation
        centered = data - np.mean(data)
        autocorr = np.correlate(centered, centered, mode='full')
        normalized_acf = autocorr[len(data)-1:] / np.max(autocorr)
        
        # Detect breaks
        breaks = []
        if normalized_acf[self.lag] > self.threshold:
            diff_acf = np.diff(normalized_acf[:self.lag+1])
            breaks = np.where(diff_acf > self.threshold/2)[0]
        
        return DecompositionResult(
            component=np.array(breaks),
            component_type=FEComponentType.SHIFT,
            method_name='acf_break_detector',
            params={
                'lag': self.lag,
                'threshold': self.threshold
            },
            stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            features={
                'acf_values': normalized_acf[:self.lag+1],
                'significant_lag': self.lag,
                'lag_acf_value': normalized_acf[self.lag]
            }
        )


class Pelt_break_detector(BaseDecomposer):
    """
    Structural break detection using PELT (Pruned Exact Linear Time) algorithm.
    
    Advantages:
    - Exact change point detection
    - Linear time complexity
    - Handles multiple change points
    
    Parameters:
        model: Cost function ("l1", "l2", "rbf")
        penalty: Penalty for adding break points
        min_size: Minimum segment length
        jump: Step size for faster computation
    """
    
    def __init__(self, model: str = "l2", penalty: float = 10.0, 
                 min_size: int = 3, jump: int = 5):
        self.model = model
        self.penalty = penalty
        self.min_size = min_size
        self.jump = jump
        
    def decompose(self, data: np.ndarray) -> DecompositionResult:
        start_time = time.time()
        
        algo = rpt.Pelt(
            model=self.model,
            min_size=self.min_size,
            jump=self.jump
        ).fit(data)
        breaks = algo.predict(pen=self.penalty)
        
        return DecompositionResult(
            component=np.array(breaks),
            component_type=FEComponentType.SHIFT,
            method_name='pelt_break_detector',
            params={
                'model': self.model,
                'penalty': self.penalty,
                'min_size': self.min_size,
                'jump': self.jump
            },
            stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            features={
                'num_breaks': len(breaks) - 1,
                'segments': [data[i:j] for i,j in zip([0]+breaks, breaks+[len(data)])]
            }
        )


class CUSUM_break_detector(BaseDecomposer):
    """
    Cumulative Sum (CUSUM) method for change point detection.
    
    Method:
    1. Accumulate deviations from mean
    2. Detect when cumulative sum exceeds threshold
    
    Parameters:
        threshold: Number of std deviations for detection
        drift: Expected drift magnitude
    """
    
    def __init__(self, threshold: float = 3.0, drift: float = 0.01):
        self.threshold = threshold
        self.drift = drift
        
    def decompose(self, data: np.ndarray) -> DecompositionResult:
        start_time = time.time()
        
        residuals = data - np.mean(data)
        cumsum = np.cumsum(residuals - self.drift)
        threshold = self.threshold * np.std(residuals)
        
        # Find threshold crossings
        breaks = np.where(np.diff(np.sign(np.abs(cumsum) - threshold)))[0]
        
        return DecompositionResult(
            component=breaks,
            component_type=FEComponentType.SHIFT,
            method_name='cusum_break_detector',
            params={
                'threshold': self.threshold,
                'drift': self.drift
            },
            stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            features={
                'max_cusum': np.max(np.abs(cumsum)),
                'cumsum_values': cumsum
            }
        )


class HMM_break_detector(BaseDecomposer):
    """
    Hidden Markov Model based structural break detection.
    
    Models different regimes as hidden states,
    with transitions indicating breaks.
    
    Parameters:
        n_states: Number of hidden states
        n_iter: Maximum EM iterations
        covariance_type: Covariance matrix type
    """
    
    def __init__(self, n_states: int = 2, n_iter: int = 50, 
                 covariance_type: str = "diag"):
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        
    def decompose(self, data: np.ndarray) -> DecompositionResult:
        start_time = time.time()
        
        model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter
        )
        
        with np.errstate(all='ignore'):
            model.fit(data.reshape(-1, 1))
            states = model.predict(data.reshape(-1, 1))
        
        breaks = np.where(np.diff(states) != 0)[0] + 1
        
        return DecompositionResult(
            component=breaks,
            component_type=FEComponentType.SHIFT,
            method_name='hmm_break_detector',
            params={
                'n_states': self.n_states,
                'n_iter': self.n_iter,
                'covariance_type': self.covariance_type
            },
            stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            features={
                'states': states,
                'transition_matrix': model.transmat_.tolist(),
                'converged': model.monitor_.converged
            }
        )