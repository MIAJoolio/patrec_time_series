from typing import Dict, List, Any, Optional, Union
import time

import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import STL as StatsmodelsSTL
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.filters.cf_filter import cffilter

from scipy import stats
from scipy.interpolate import LSQUnivariateSpline

import pywt

from ..old_fe_classes import Base_decomposer, Decomposition_result, FE_component_type, Base_optimizer

__all__ = [
    'Lin_trend_decomposer',
    'Poly_trend_decomposer',
    'SMA_trend_decomposer',
    'EMA_trend_decomposer',
    'HP_trend_decomposer',
    'CF_trend_decomposer',
    'STL_trend_decomposer',
    'Wavelet_trend_decomposer',
    'Spline_trend_decomposer'
]


class Lin_trend_decomposer(Base_decomposer):
    """Linear trend decomposition with optimization and feature extraction"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        
    @classmethod
    def optimize_params(cls, data: np.ndarray, n_trials: int = 20) -> Dict[str, Any]:
        """Optimize alpha parameter"""
        param_grid = {
            'alpha': {'type': 'float', 'low': 0.01, 'high': 0.5}
        }
        return Base_optimizer.optimize(cls, data, param_grid, n_trials)
        
    def _extract_features(self, trend: np.ndarray, data: np.ndarray) -> Dict[str, float]:
        """Feature extraction specific for linear trend"""
        x = np.arange(len(trend))
        slope, intercept, r_value, _, _ = stats.linregress(x, trend)
        return {
            'trend_slope': slope,
            'trend_intercept': intercept,
            'trend_linearity': r_value**2,
            'trend_residual_std': np.std(data - trend)
        }
        
    def decompose(self, data: np.ndarray, x: Optional[np.ndarray] = None) -> Decomposition_result:
        start_time = time.time()
        n = len(data)
        x = np.arange(n) if x is None else x
        
        slope, intercept, _, _, _ = stats.linregress(x=x, y=data)
        trend = slope * x + intercept
        
        features = self._extract_features(trend, data)
        
        return Decomposition_result(
            component=trend,
            component_type=FE_component_type.TREND,
            method_name='linear_trend',
            params={'alpha': self.alpha, 'x_provided': x is not None},
            stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape,
                'features': features
            }
        )


class Poly_trend_decomposer(Base_decomposer):
    """Polynomial trend decomposition with optimization and feature extraction"""
    
    def __init__(self, degree: int = 1):
        self.degree = degree
        
    @classmethod
    def optimize_params(cls, data: np.ndarray, n_trials: int = 20) -> Dict[str, Any]:
        """Optimize polynomial degree"""
        param_grid = {
            'degree': {'type': 'int', 'low': 1, 'high': 5}
        }
        return Base_optimizer.optimize(cls, data, param_grid, n_trials)
        
    def _extract_features(self, trend: np.ndarray, data: np.ndarray) -> Dict[str, float]:
        """Feature extraction specific for polynomial trend"""
        curvature = np.mean(np.abs(np.diff(np.diff(trend))))
        return {
            'trend_slope': np.mean(trend),
            'trend_poly_degree': self.degree,
            'trend_curvature': curvature
        }
        
    def decompose(self, data: np.ndarray, x: Optional[np.ndarray] = None) -> Decomposition_result:
        start_time = time.time()
        x = np.arange(len(data)) if x is None else x
        
        coefficients = np.polyfit(x, data, deg=self.degree)
        trend = np.polyval(coefficients, x)
        
        features = self._extract_features(trend, data)
        
        return Decomposition_result(
            component=trend,
            component_type=FE_component_type.TREND,
            method_name='polynomial_trend',
            params={'degree': self.degree, 'x_provided': x is not None},
            stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape,
                'features': features
            }
        )


class SMA_trend_decomposer(Base_decomposer):
    """SMA trend decomposition with optimization and feature extraction"""
    
    def __init__(self, window: int = 5):
        self.window = window
        
    @classmethod
    def optimize_params(cls, data: np.ndarray, n_trials: int = 20) -> Dict[str, Any]:
        """Optimize window size"""
        param_grid = {
            'window': {'type': 'int', 'low': 3, 'high': 21, 'step': 2}
        }
        return Base_optimizer.optimize(cls, data, param_grid, n_trials)
        
    def _extract_features(self, trend: np.ndarray, data: np.ndarray) -> Dict[str, float]:
        """Feature extraction specific for SMA trend"""
        smoothness = 1 - (np.std(np.diff(trend)) / (np.std(np.diff(data)) + 1e-10))
        return {
            'trend_slope': np.mean(trend),
            'trend_smoothness': smoothness,
            'trend_window_size': self.window,
            'trend_lag_correlation': np.corrcoef(data[self.window:], trend[:-self.window])[0,1]
        }
        
    def decompose(self, data: np.ndarray) -> Decomposition_result:
        start_time = time.time()
        sma = pd.Series(data).rolling(window=self.window, center=True).mean()
        sma = sma.fillna(method='bfill').fillna(method='ffill')
        trend = sma.values
        
        features = self._extract_features(trend, data)
        
        return Decomposition_result(
            component=trend,
            component_type=FE_component_type.TREND,
            method_name='sma_trend',
            params={'window': self.window},
            stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape,
                'features': features
            }
        )


class HP_trend_decomposer(Base_decomposer):
    """HP filter decomposition with optimization and feature extraction"""
    
    def __init__(self, lamb: float = 1600, regime: str = 'none'):
        self.lamb = lamb
        self.regime = regime
        
    @classmethod
    def optimize_params(cls, data: np.ndarray, n_trials: int = 20) -> Dict[str, Any]:
        """Optimize lambda parameter"""
        param_grid = {
            'lamb': {'type': 'float', 'low': 10, 'high': 10000, 'log': True},
            'regime': {'type': 'categorical', 'values': ['none', 'annual', 'monthly']}
        }
        return Base_optimizer.optimize(cls, data, param_grid, n_trials)
        
    def _extract_features(self, trend: np.ndarray, data: np.ndarray) -> Dict[str, float]:
        """Feature extraction specific for HP filter"""
        cycles = data - trend
        return {
            'trend_slope': np.mean(trend),
            'trend_smoothness': 1 - (np.std(np.diff(trend)) / (np.std(np.diff(data)) + 1e-10)),
            'cycle_std': np.std(cycles),
            'trend_lambda': self.lamb if self.regime == 'none' else 
                          (1600/4**4 if self.regime == 'annual' else 1600*3**4)
        }
        
    def decompose(self, data: np.ndarray) -> Decomposition_result:
        start_time = time.time()
        if self.regime == 'annual':
            lamb = 1600/4**4
        elif self.regime == 'monthly':
            lamb = 1600*3**4
        else:
            lamb = self.lamb
        
        trend, _ = hpfilter(data, lamb=lamb)
        
        features = self._extract_features(trend, data)
        
        return Decomposition_result(
            component=trend,
            component_type=FE_component_type.TREND,
            method_name='hpfilter_trend',
            params={'lambda': lamb, 'regime': self.regime},
            stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape,
                'features': features
            }
        )


class EMA_trend_decomposer(Base_decomposer):
    """EMA trend decomposition with optimization and feature extraction"""
    
    def __init__(self, span: int = 10):
        self.span = span
        
    @classmethod
    def optimize_params(cls, data: np.ndarray, n_trials: int = 20) -> Dict[str, Any]:
        """Optimize span parameter"""
        param_grid = {
            'span': {'type': 'int', 'low': 2, 'high': 30}
        }
        return Base_optimizer.optimize(cls, data, param_grid, n_trials)
        
    def _extract_features(self, trend: np.ndarray, data: np.ndarray) -> Dict[str, float]:
        """Feature extraction specific for EMA trend"""
        responsiveness = np.mean(np.abs(trend - data))
        smoothness = 1 - (np.std(np.diff(trend)) / (np.std(np.diff(data)) + 1e-10))
        lag_corr = np.corrcoef(data[1:], trend[:-1])[0,1] if len(data) > 1 else 0
        
        return {
            'trend_slope': np.mean(trend),
            'trend_responsiveness': float(responsiveness),
            'trend_smoothness': float(smoothness),
            'trend_span': float(self.span),
            'trend_lag_correlation': float(lag_corr)
        }
        
    def decompose(self, data: np.ndarray) -> Decomposition_result:
        start_time = time.time()
        ema = pd.Series(data).ewm(span=self.span, adjust=False).mean()
        trend = ema.values
        
        features = self._extract_features(trend, data)
        
        return Decomposition_result(
            component=trend,
            component_type=FE_component_type.TREND,
            method_name='ema_trend',
            params={'span': self.span},
            stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape,
                'features': features
            }
        )


class CF_trend_decomposer(Base_decomposer):
    """Christiano-Fitzgerald filter with optimization and feature extraction"""
    
    def __init__(self, low: int = 6, high: int = 32, drift: bool = True):
        self.low = low
        self.high = high
        self.drift = drift
        
    @classmethod
    def optimize_params(cls, data: np.ndarray, n_trials: int = 20) -> Dict[str, Any]:
        """Optimize frequency band parameters"""
        param_grid = {
            'low': {'type': 'int', 'low': 2, 'high': 12},
            'high': {'type': 'int', 'low': 12, 'high': 36},
            'drift': {'type': 'categorical', 'values': [True, False]}
        }
        return Base_optimizer.optimize(cls, data, param_grid, n_trials)
        
    def _extract_features(self, trend: np.ndarray, data: np.ndarray) -> Dict[str, float]:
        """Feature extraction specific for CF filter"""
        cycles = data - trend
        smoothness = 1 - (np.std(np.diff(trend)) / (np.std(np.diff(data)) + 1e-10))
        
        return {
            'trend_slope': np.mean(trend),
            'trend_band_low': float(self.low),
            'trend_band_high': float(self.high),
            'cycle_std': float(np.std(cycles)),
            'trend_drift': float(self.drift),
            'trend_smoothness': float(smoothness)
        }
        
    def decompose(self, data: np.ndarray) -> Decomposition_result:
        start_time = time.time()
        _, trend = cffilter(data, low=self.low, high=self.high, drift=self.drift)
        
        features = self._extract_features(trend, data)
        
        return Decomposition_result(
            component=trend,
            component_type=FE_component_type.TREND,
            method_name='cffilter_trend',
            params={'low': self.low, 'high': self.high, 'drift': self.drift},
            stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape,
                'features': features
            }
        )


class STL_trend_decomposer(Base_decomposer):
    """STL decomposition with optimization and feature extraction"""
    
    def __init__(self, period: Optional[int] = None, 
                 trend_deg: int = 1,
                 seasonal_deg: int = 1):
        self.period = period
        self.trend_deg = trend_deg
        self.seasonal_deg = seasonal_deg
        
    @classmethod
    def optimize_params(cls, data: np.ndarray, n_trials: int = 20) -> Dict[str, Any]:
        """Optimize STL parameters"""
        max_period = min(100, len(data)//2)
        param_grid = {
            'period': {'type': 'int', 'low': 2, 'high': max_period},
            'trend_deg': {'type': 'int', 'low': 1, 'high': 2},
            'seasonal_deg': {'type': 'int', 'low': 1, 'high': 2}
        }
        return Base_optimizer.optimize(cls, data, param_grid, n_trials)
        
    def _extract_features(self, trend: np.ndarray, data: np.ndarray) -> Dict[str, float]:
        """Feature extraction specific for STL"""
        seasonal = data - trend - (data - trend).mean()  # approximate seasonal
        smoothness = 1 - (np.std(np.diff(trend)) / (np.std(np.diff(data)) + 1e-10))
        seasonal_strength = max(0, 1 - np.var(data - trend - seasonal)/np.var(data - trend))
        
        return {
            'trend_slope': np.mean(trend),
            'trend_smoothness': float(smoothness),
            'seasonal_strength': float(seasonal_strength),
            'trend_deg': float(self.trend_deg),
            'seasonal_deg': float(self.seasonal_deg)
        }
        
    def decompose(self, data: np.ndarray) -> Decomposition_result:
        start_time = time.time()
        stl = StatsmodelsSTL(data, period=self.period, 
                           trend_deg=self.trend_deg,
                           seasonal_deg=self.seasonal_deg)
        result = stl.fit()
        trend = result.trend
        
        features = self._extract_features(trend, data)
        
        return Decomposition_result(
            component=trend,
            component_type=FE_component_type.TREND,
            method_name='stl_trend',
            params={
                'period': self.period,
                'trend_deg': self.trend_deg,
                'seasonal_deg': self.seasonal_deg
            },
            stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape,
                'features': features
            }
        )


class Wavelet_trend_decomposer(Base_decomposer):
    """Wavelet decomposition with optimization and feature extraction"""
    
    def __init__(self, wavelet: str = 'db4', level: Optional[int] = None,
                 mode: str = 'symmetric', energy_threshold: float = 0.9):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.energy_threshold = energy_threshold
        
    @classmethod
    def optimize_params(cls, data: np.ndarray, n_trials: int = 20) -> Dict[str, Any]:
        """Optimize wavelet parameters"""
        max_level = min(5, pywt.dwt_max_level(len(data), 'db4'))
        param_grid = {
            'wavelet': {'type': 'categorical', 'values': ['db4', 'sym5', 'haar']},
            'level': {'type': 'int', 'low': 1, 'high': max_level},
            'energy_threshold': {'type': 'float', 'low': 0.7, 'high': 0.99}
        }
        return Base_optimizer.optimize(cls, data, param_grid, n_trials)
        
    def _extract_features(self, trend: np.ndarray, data: np.ndarray) -> Dict[str, float]:
        """Feature extraction specific for Wavelet"""
        details_energy = np.var(data - trend)
        energy_ratio = np.var(trend)/(np.var(data) + 1e-10)
        smoothness = 1 - (np.std(np.diff(trend)) / (np.std(np.diff(data)) + 1e-10))
        
        return {
            'trend_slope': np.mean(trend),
            'wavelet_type': str(self.wavelet),
            'decomposition_level': float(self.level if self.level is not None else 0),
            'details_energy': float(details_energy),
            'energy_ratio': float(energy_ratio),
            'trend_smoothness': float(smoothness)
        }
        
    def decompose(self, data: np.ndarray) -> Decomposition_result:
        start_time = time.time()
        
        # Auto-select level if not specified
        if self.level is None:
            max_level = pywt.dwt_max_level(len(data), self.wavelet)
            coeffs = pywt.wavedec(data, self.wavelet, level=max_level, mode=self.mode)
            energy = np.cumsum([np.sum(c**2) for c in coeffs[::-1]])[::-1]
            level = np.argmax(energy/energy[0] > self.energy_threshold) + 1
        else:
            level = self.level
        
        # Perform decomposition
        coeffs = pywt.wavedec(data, self.wavelet, level=level, mode=self.mode)
        
        # Reconstruct trend
        trend_coeffs = [coeffs[0]] + [None]*(len(coeffs)-1)
        trend = pywt.waverec(trend_coeffs, self.wavelet, mode=self.mode)
        trend = trend[:len(data)]
        
        features = self._extract_features(trend, data)
        
        return Decomposition_result(
            component=trend,
            component_type=FE_component_type.TREND,
            method_name='wavelet_trend',
            params={
                'wavelet': self.wavelet,
                'level': level,
                'mode': self.mode,
                'energy_threshold': self.energy_threshold
            },
            stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape,
                'features': features
            }
        )


class Spline_trend_decomposer(Base_decomposer):
    """Spline decomposition with optimization and feature extraction"""
    
    def __init__(self, degree: int = 3, n_knots: Optional[int] = None,
                 knot_selection: str = 'quantile'):
        self.degree = degree
        self.n_knots = n_knots
        self.knot_selection = knot_selection
        
    @classmethod
    def optimize_params(cls, data: np.ndarray, n_trials: int = 20) -> Dict[str, Any]:
        """Optimize spline parameters"""
        max_knots = max(3, min(20, len(data)//10))
        param_grid = {
            'degree': {'type': 'int', 'low': 2, 'high': 5},
            'n_knots': {'type': 'int', 'low': 3, 'high': max_knots},
            'knot_selection': {'type': 'categorical', 'values': ['quantile', 'uniform']}
        }
        return Base_optimizer.optimize(cls, data, param_grid, n_trials)
        
    def _extract_features(self, trend: np.ndarray, data: np.ndarray) -> Dict[str, float]:
        """Feature extraction specific for Spline"""
        curvature = np.mean(np.abs(np.diff(np.diff(trend))))
        flexibility = np.var(data - trend)/np.var(data)
        
        return {
            'trend_slope': np.mean(trend),
            'spline_degree': float(self.degree),
            'n_knots': float(self.n_knots if self.n_knots is not None else 0),
            'trend_curvature': float(curvature),
            'trend_flexibility': float(flexibility),
            'knot_selection': str(self.knot_selection)
        }
        
    def decompose(self, data: np.ndarray) -> Decomposition_result:
        start_time = time.time()
        x = np.arange(len(data))
        
        # Auto-select knots if not specified
        n_knots = self.n_knots if self.n_knots is not None else max(3, min(20, len(data)//10))
        
        # Select knot positions
        if self.knot_selection == 'quantile':
            knots = np.quantile(x, np.linspace(0, 1, n_knots))
        else:  # uniform
            knots = np.linspace(x.min(), x.max(), n_knots)
        
        # Fit spline
        spline = LSQUnivariateSpline(x, data, knots[1:-1], k=self.degree)
        trend = spline(x)
        
        features = self._extract_features(trend, data)
        
        return Decomposition_result(
            component=trend,
            component_type=FE_component_type.TREND,
            method_name='spline_trend',
            params={
                'degree': self.degree,
                'n_knots': n_knots,
                'knot_selection': self.knot_selection
            },
            stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape,
                'features': features
            }
        )
        