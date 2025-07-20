from typing import Dict, List, Any, Optional, Union
import time

import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import STL as StatsmodelsSTL
from sklearn.metrics import mean_squared_error

from ..old_fe_classes import Base_decomposer, Decomposition_result, FE_component_type, Base_optimizer


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