from typing import Any

import numpy as np
import pandas as pd

from patrec_ts.feature_extraction.fe_classes import DecompositionResult, FEComponentType, BaseOptimizer
from patrec_ts.preprocessing.manual_methods.parametric.base_parametric_method import BaseParametricPreprocessor


class TrendEMAPreprocessor(BaseParametricPreprocessor):
    """EMA trend decomposition with params dict"""

    def __init__(self, params: dict[str, Any] = None):
        self.params = {'span': 10} | (params or {})

    def _extract_features(self, trend: np.ndarray, data: np.ndarray) -> dict[str, float]:
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

    def decompose(self, data: np.ndarray, opt: bool = True, **kwargs) -> DecompositionResult:
        if opt:
            self._optimize_params(data)

        span = self.params['span']
        trend = pd.Series(data).ewm(span=span, adjust=False).mean().values

        features = self._extract_features(trend, data)

        return DecompositionResult(
            component=trend,
            component_type=FEComponentType.TREND,
            method_name='ema_trend',
            params=self.params.copy(),
            stats={'features': features}
        )

    def _optimize_params(self, data: np.ndarray):
        param_grid = {'span': {'type': 'int', 'low': 2, 'high': 30}}
        best_params = BaseOptimizer.optimize(self.__class__, data, param_grid)
        self.params.update(best_params['best_params'])
