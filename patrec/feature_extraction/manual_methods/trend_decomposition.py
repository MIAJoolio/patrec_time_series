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

from sklearn.metrics import mean_squared_error

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
    """Linear trend decomposition with params dict and opt flag"""
    def __init__(self, alpha: float = 0.05, x: np.ndarray = None, params: Dict[str, Any] = None):
        self.params = {'alpha': alpha, 'x': x}
        if params:
            self.params.update(params)

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

    def decompose(self, data: np.ndarray, opt: bool = False, **kwargs) -> Decomposition_result:
        """
        Decomposes time series into linear trend component with optional parameter optimization.
        
        Args:
            data: Input time series (1D numpy array)
            opt: Whether to optimize parameters before decomposition
        
        Returns:
            Decomposition_result object containing extracted trend and metadata
        """
        if not isinstance(data, np.ndarray) or data.ndim != 1:
            raise ValueError(f"data must be a 1D numpy array. Got shape {data.shape}, type {type(data)}")

        if len(data) < 2:
            raise ValueError(f"data must contain at least 2 points for linear regression. Got {len(data)}")

        # Получаем x из params или создаём стандартный
        x = self.params.get('x', None)

        # Если x не задан — создаём от 0 до len(data)-1
        if x is None:
            x = np.arange(len(data))
        else:
            # Пробуем привести к numpy array
            try:
                x = np.array(x)
            except Exception as e:
                raise ValueError(f"Failed to convert 'x' to numpy array. Got type {type(x)}") from e

        # Проверяем размерность
        if x.ndim != 1:
            raise ValueError(f"x must be a 1D array. Got shape {x.shape}")
        if len(x) != len(data):
            raise ValueError(f"x and data must have the same length. "
                            f"data.shape={data.shape}, x.shape={x.shape}")

        slope, intercept, _, _, _ = stats.linregress(x=x, y=data)
        trend = slope * x + intercept
        features = self._extract_features(trend, data)

        return Decomposition_result(
            component=trend,
            component_type=FE_component_type.TREND,
            method_name='linear_trend',
            params=self.params.copy(),
            stats={'features': features}
        )


class Poly_trend_decomposer(Base_decomposer):
    """Polynomial trend decomposition with params dict"""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = {'degree': 1, 'x': None}
        if params:
            self.params.update(params)

    def _extract_features(self, trend: np.ndarray, data: np.ndarray) -> Dict[str, float]:
        """Feature extraction specific for polynomial trend"""
        curvature = np.mean(np.abs(np.diff(np.diff(trend))))
        return {
            'trend_slope': np.mean(trend),
            'trend_poly_degree': self.params['degree'],
            'trend_curvature': curvature,
            'trend_residual_std': np.std(data - trend)
        }

    def decompose(self, data: np.ndarray, opt: bool = True, **kwargs) -> Decomposition_result:
        """
        Decomposes time series into polynomial trend component with optional parameter optimization.
        """
        # Input validation (остается без изменений)
        if not isinstance(data, np.ndarray) or data.ndim != 1:
            raise ValueError(f"data must be a 1D numpy array. Got shape {data.shape}, type {type(data)}")

        if len(data) < 2:
            raise ValueError(f"data must contain at least 2 points for polynomial fitting. Got {len(data)}")

        x = kwargs.get('x', self.params.get('x'))
        if x is None:
            x = np.arange(len(data))
        else:
            x = np.asarray(x, dtype=np.float64)
            if x.ndim != 1 or len(x) != len(data):
                raise ValueError(f"x must be 1D array with same length as data. Got shape {x.shape}")

        if opt:
            self._optimize_params(data, x)

        degree = kwargs.get('degree', self.params['degree'])
        coeffs = np.polyfit(x, data, deg=degree)
        trend = np.polyval(coeffs, x)
        features = self._extract_features(trend, data)

        return Decomposition_result(
            component=trend,
            component_type=FE_component_type.TREND,
            method_name='polynomial_trend',
            params=self.params.copy(),
            stats={'features': features}
        )


    def _optimize_params(self, data: np.ndarray, x: np.ndarray):
        """Safe parameter optimization"""
        param_grid = {
            'degree': {
                'type': 'int',
                'low': 1,
                'high': 3,
            }
        }
        
        result = Base_optimizer.optimize(
            decomposer_class=lambda **p: Poly_trend_decomposer({'x': x, **p}),
            data=data,
            param_grid=param_grid,
            n_trials=min(50, len(data)), 
            timeout=30, 
            max_failures=5 
        )
        
        if result['best_score'] == float('inf'):
            raise RuntimeError("Parameter optimization failed")
            
        self.params['degree'] = result['best_params']['degree']
        

class SMA_trend_decomposer(Base_decomposer):
    """SMA trend decomposition with params dict"""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = {'window': 5}
        if params:
            self.params.update(params)
    
    def _extract_features(self, trend: np.ndarray, data: np.ndarray) -> Dict[str, float]:
        """Feature extraction specific for SMA trend"""
        smoothness = 1 - (np.std(np.diff(trend)) / (np.std(np.diff(data)) + 1e-10))
        return {
            'trend_slope': np.mean(trend),
            'trend_smoothness': smoothness,
            'trend_window_size': self.window,
            'trend_lag_correlation': np.corrcoef(data[self.window:], trend[:-self.window])[0,1]
        }
        
    def decompose(self, data: np.ndarray, opt: bool = True, **kwargs) -> Decomposition_result:
        if opt:
            self._optimize_params(data)
            
        window = self.params['window']
        sma = pd.Series(data).rolling(window=window, center=True).mean()
        trend = sma.fillna(method='bfill').fillna(method='ffill').values
        
        features = self._extract_features(trend, data)
        
        return Decomposition_result(
            component=trend,
            component_type=FE_component_type.TREND,
            method_name='sma_trend',
            params=self.params.copy(),
            stats={'features': features}
        )
    
    def _optimize_params(self, data: np.ndarray):
        param_grid = {'window': {'type': 'int', 'low': 3, 'high': 21, 'step': 2}}
        best_params = Base_optimizer.optimize(self.__class__, data, param_grid)
        self.params.update(best_params['best_params'])

class EMA_trend_decomposer(Base_decomposer):
    """EMA trend decomposition with params dict"""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = {'span': 10}
        if params:
            self.params.update(params)
    
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
        
    def decompose(self, data: np.ndarray, opt: bool = True, **kwargs) -> Decomposition_result:
        if opt:
            self._optimize_params(data)
            
        span = self.params['span']
        trend = pd.Series(data).ewm(span=span, adjust=False).mean().values
        
        features = self._extract_features(trend, data)
        
        return Decomposition_result(
            component=trend,
            component_type=FE_component_type.TREND,
            method_name='ema_trend',
            params=self.params.copy(),
            stats={'features': features}
        )
    
    def _optimize_params(self, data: np.ndarray):
        param_grid = {'span': {'type': 'int', 'low': 2, 'high': 30}}
        best_params = Base_optimizer.optimize(self.__class__, data, param_grid)
        self.params.update(best_params['best_params'])

class HP_trend_decomposer(Base_decomposer):
    """HP filter decomposition with params dict"""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = {'lamb': 1600, 'regime': 'none'}
        if params:
            self.params.update(params)
    
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
        
    def decompose(self, data: np.ndarray, opt: bool = True, **kwargs) -> Decomposition_result:
        if opt:
            self._optimize_params(data)
            
        lamb = self._get_effective_lambda()
        trend, _ = hpfilter(data, lamb=lamb)
        
        features = self._extract_features(trend, data)
        
        return Decomposition_result(
            component=trend,
            component_type=FE_component_type.TREND,
            method_name='hpfilter_trend',
            params=self.params.copy(),
            stats={'features': features}
        )
    
    def _get_effective_lambda(self):
        regime = self.params['regime']
        if regime == 'annual': return 1600/4**4
        if regime == 'monthly': return 1600*3**4
        return self.params['lamb']
    
    def _optimize_params(self, data: np.ndarray):
        param_grid = {
            'lamb': {'type': 'float', 'low': 10, 'high': 10000, 'log': True},
            'regime': {'type': 'categorical', 'values': ['none', 'annual', 'monthly']}
        }
        best_params = Base_optimizer.optimize(self.__class__, data, param_grid)
        self.params.update(best_params['best_params'])

class CF_trend_decomposer(Base_decomposer):
    """CF filter decomposition with params dict"""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = {'low': 6, 'high': 32, 'drift': True}
        if params:
            self.params.update(params)
        
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

    def decompose(self, data: np.ndarray, opt: bool = True, **kwargs) -> Decomposition_result:
        if opt:
            self._optimize_params(data)
            
        low = self.params['low']
        high = self.params['high']
        drift = self.params['drift']
        _, trend = cffilter(data, low=low, high=high, drift=drift)
        
        features = self._extract_features(trend, data)
        
        return Decomposition_result(
            component=trend,
            component_type=FE_component_type.TREND,
            method_name='cffilter_trend',
            params=self.params.copy(),
            stats={'features': features}
        )
    
    def _optimize_params(self, data: np.ndarray):
        param_grid = {
            'low': {'type': 'int', 'low': 2, 'high': 12},
            'high': {'type': 'int', 'low': 12, 'high': 36},
            'drift': {'type': 'categorical', 'values': [True, False]}
        }
        best_params = Base_optimizer.optimize(self.__class__, data, param_grid)
        self.params.update(best_params['best_params'])


class STL_trend_decomposer(Base_decomposer):
    """STL decomposition with trend-aware optimization"""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = {
            'period': None, 
            'trend_deg': 1, 
            'seasonal_deg': 1,
            'robust': True
        }
        if params:
            self.params.update(params)
    
    def _extract_features(self, trend: np.ndarray, data: np.ndarray) -> Dict[str, float]:
        """Enhanced feature extraction for STL"""
        residuals = data - trend
        seasonal = residuals - residuals.mean()
        
        # Trend quality metrics
        trend_slope = np.polyfit(np.arange(len(trend)), trend, 1)[0]
        trend_lin_coef = np.corrcoef(np.arange(len(trend)), trend)[0,1]**2
        trend_smoothness = 1 - (np.std(np.diff(trend))/(np.std(np.diff(data)) + 1e-10))
        
        # Seasonality metrics
        seasonal_strength = max(0, min(1, 1 - np.var(residuals - seasonal)/np.var(residuals)))
        
        return {
            'trend_slope': float(trend_slope),
            'trend_linearity': float(trend_lin_coef),
            'trend_smoothness': float(trend_smoothness),
            'seasonal_strength': float(seasonal_strength),
            'trend_deg': int(self.params['trend_deg']),
            'seasonal_deg': int(self.params['seasonal_deg'])
        }
        
    def decompose(self, data: np.ndarray, opt: bool = True, **kwargs) -> Decomposition_result:
        if opt:
            self._optimize_params(data)
            
        try:
            from statsmodels.tsa.seasonal import STL
            stl = STL(data,
                     period=self.params['period'],
                     trend_deg=self.params['trend_deg'],
                     seasonal_deg=self.params['seasonal_deg'],
                     robust=self.params['robust'])
            res = stl.fit()
            trend = res.trend
        except ImportError:
            # Fallback to simple decomposition if STL not available
            trend = self._fallback_trend(data)
        
        features = self._extract_features(trend, data)
        
        return Decomposition_result(
            component=trend,
            component_type=FE_component_type.TREND,
            method_name='stl_trend',
            params=self.params.copy(),
            stats={'features': features}
        )
    
    def _optimize_params(self, data: np.ndarray):
        """Trend-aware parameter optimization"""
        max_period = min(24, len(data)//2)  # Practical limit for period
        
        def trend_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            """Custom metric evaluating trend quality"""
            # 1. Compare smoothed versions
            smooth_true = Base_optimizer._prepare_series(y_true)
            smooth_pred = Base_optimizer._prepare_series(y_pred)
            
            # 2. Combine RMSE with trend characteristics
            rmse = np.sqrt(mean_squared_error(smooth_true, smooth_pred))
            trend_slope = np.polyfit(np.arange(len(y_pred)), y_pred, 1)[0]
            
            # Penalize excessive curvature
            curvature = np.mean(np.abs(np.diff(np.diff(y_pred))))
            
            return rmse + 0.1*abs(trend_slope) + 0.5*curvature
        
        param_grid = {
            'period': {
                'type': 'int', 
                'low': 2, 
                'high': max_period,
                'values': [7, 12, 24, 30, 52, 365]  # Common periods
            },
            'trend_deg': {'type': 'int', 'low': 1, 'high': 2},
            'seasonal_deg': {'type': 'int', 'low': 1, 'high': 2},
            'robust': {'type': 'categorical', 'values': [True, False]}
        }
        
        result = Base_optimizer.optimize(
            decomposer_class=lambda **p: STL_trend_decomposer(p),
            data=data,
            param_grid=param_grid,
            metric_func=trend_metric,
            n_trials=30,
            timeout=60  # 1 minute timeout
        )
        
        if result['best_score'] == float('inf'):
            raise RuntimeError("STL optimization failed")
            
        self.params.update(result['best_params'])
        

class Wavelet_trend_decomposer(Base_decomposer):
    """Wavelet decomposition with params dict"""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = {
            'wavelet': 'db4', 
            'level': None,
            'energy_threshold': 0.9,
            'mode': 'symmetric'
        }
        if params:
            self.params.update(params)
    
    def _extract_features(self, trend: np.ndarray, data: np.ndarray) -> Dict[str, float]:
        """Feature extraction specific for Wavelet"""
        details_energy = np.var(data - trend)
        energy_ratio = np.var(trend)/(np.var(data) + 1e-10)
        smoothness = 1 - (np.std(np.diff(trend)) / (np.std(np.diff(data)) + 1e-10))
        
        return {
            'trend_slope': np.mean(trend),
            'decomposition_level': float(self.level if self.level is not None else 0),
            'details_energy': float(details_energy),
            'energy_ratio': float(energy_ratio),
            'trend_smoothness': float(smoothness)
        }

    def decompose(self, data: np.ndarray, opt: bool = True, **kwargs) -> Decomposition_result:
        if opt:
            self._optimize_params(data)
            
        level = self._determine_level(data)
        coeffs = pywt.wavedec(data, self.params['wavelet'], level=level, mode=self.params['mode'])
        trend = pywt.waverec([coeffs[0]] + [None]*(len(coeffs)-1), 
                            self.params['wavelet'],
                            mode=self.params['mode'])[:len(data)]
        
        features = self._extract_features(trend, data)
        
        return Decomposition_result(
            component=trend,
            component_type=FE_component_type.TREND,
            method_name='wavelet_trend',
            params=self.params.copy(),
            stats={'features': features}
        )
    
    def _determine_level(self, data: np.ndarray) -> int:
        if self.params['level'] is not None:
            return self.params['level']
        
        max_level = pywt.dwt_max_level(len(data), self.params['wavelet'])
        coeffs = pywt.wavedec(data, self.params['wavelet'], level=max_level, mode=self.params['mode'])
        energy = np.cumsum([np.sum(c**2) for c in coeffs[::-1]])[::-1]
        return np.argmax(energy/energy[0] > self.params['energy_threshold']) + 1
    
    def _optimize_params(self, data: np.ndarray):
        max_level = min(5, pywt.dwt_max_level(len(data), 'db4'))
        param_grid = {
            'wavelet': {'type': 'categorical', 'values': ['db4', 'sym5', 'haar']},
            'level': {'type': 'int', 'low': 1, 'high': max_level},
            'energy_threshold': {'type': 'float', 'low': 0.7, 'high': 0.99}
        }
        best_params = Base_optimizer.optimize(self.__class__, data, param_grid)
        self.params.update(best_params['best_params'])


class Spline_trend_decomposer(Base_decomposer):
    """Spline decomposition with params dict"""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = {
            'degree': 3, 
            'n_knots': None,
            'knot_selection': 'quantile'
        }
        if params:
            self.params.update(params)

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
            
    def decompose(self, data: np.ndarray, opt: bool = True, **kwargs) -> Decomposition_result:
        if opt:
            self._optimize_params(data)
            
        x = np.arange(len(data))
        n_knots = self.params['n_knots'] or max(3, min(20, len(data)//10))
        
        if self.params['knot_selection'] == 'quantile':
            knots = np.quantile(x, np.linspace(0, 1, n_knots))
        else:
            knots = np.linspace(x.min(), x.max(), n_knots)
        
        spline = LSQUnivariateSpline(x, data, knots[1:-1], k=self.params['degree'])
        trend = spline(x)
        
        features = self._extract_features(trend, data)
        
        return Decomposition_result(
            component=trend,
            component_type=FE_component_type.TREND,
            method_name='spline_trend',
            params=self.params.copy(),
            stats={'features': features}
        )
    
    def _optimize_params(self, data: np.ndarray):
        max_knots = max(3, min(20, len(data)//10))
        param_grid = {
            'degree': {'type': 'int', 'low': 2, 'high': 5},
            'n_knots': {'type': 'int', 'low': 3, 'high': max_knots},
            'knot_selection': {'type': 'categorical', 'values': ['quantile', 'uniform']}
        }
        best_params = Base_optimizer.optimize(self.__class__, data, param_grid)
        self.params.update(best_params['best_params'])