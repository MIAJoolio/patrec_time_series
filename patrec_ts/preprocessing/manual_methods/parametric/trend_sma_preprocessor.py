from typing import Any

import numpy as np
import pandas as pd

from patrec_ts.feature_extraction.fe_classes import DecompositionResult, FEComponentType, BaseOptimizer
from patrec_ts.preprocessing.manual_methods.parametric.base_parametric_method import BaseParametricPreprocessor


class TrendSMAPreprocessor(BaseParametricPreprocessor):
    """SMA trend decomposition with params dict"""

    def __init__(self, params: dict[str, Any] = None):
        self.params = {'window': 5} | (params or {})

    def _extract_features(self, trend: np.ndarray, data: np.ndarray) -> dict[str, float]:
        """Feature extraction specific for SMA trend"""
        smoothness = 1 - (np.std(np.diff(trend)) / (np.std(np.diff(data)) + 1e-10))
        return {
            'trend_slope': np.mean(trend),
            'trend_smoothness': smoothness,
            'trend_window_size': self.window,
            'trend_lag_correlation': np.corrcoef(data[self.window:], trend[:-self.window])[0,1]
        }

    def decompose(self, data: np.ndarray, opt: bool = True, **kwargs) -> DecompositionResult:
        if opt:
            self._optimize_params(data)

        window = self.params['window']
        sma = pd.Series(data).rolling(window=window, center=True).mean()
        trend = sma.fillna(method='bfill').fillna(method='ffill').values

        features = self._extract_features(trend, data)

        return DecompositionResult(
            component=trend,
            component_type=FEComponentType.TREND,
            method_name='sma_trend',
            params=self.params.copy(),
            stats={'features': features}
        )

    def _optimize_params(self, data: np.ndarray):
        param_grid = {'window': {'type': 'int', 'low': 3, 'high': 21, 'step': 2}}
        best_params = BaseOptimizer.optimize(self.__class__, data, param_grid)
        self.params.update(best_params['best_params'])