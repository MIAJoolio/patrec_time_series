from typing import Any

import numpy as np
from scipy.stats import stats

from patrec_ts.feature_extraction.fe_classes import DecompositionResult, FEComponentType
from patrec_ts.preprocessing.manual_methods.parametric.base_parametric_method import BaseParametricPreprocessor


class TrendLinearPreprocessor(BaseParametricPreprocessor):
    """Linear trend decomposition with params dict and opt flag"""
    def __init__(self, alpha: float = 0.05, params: dict[str, Any] = None):
        self.params = {'alpha': alpha} | (params or {})

    def _extract_features(self, trend: np.ndarray, data: np.ndarray) -> dict[str, np.floating]:
        """Feature extraction specific for linear trend"""
        x = np.arange(len(trend))
        slope, intercept, r_value, _, _ = stats.linregress(x, trend)
        return {
            'trend_slope': slope,
            'trend_intercept': intercept,
            'trend_linearity': r_value**2,
            'trend_residual_std': np.std(data - trend)
        }

    def decompose(self, data: np.ndarray, opt: bool = False, **kwargs) -> DecompositionResult:
        """
        Decomposes time series into linear trend component with optional parameter optimization.

        Args:
            data: Input time series (1D numpy array)
            opt: Whether to optimize parameters before decomposition

        Returns:
            DecompositionResult object containing extracted trend and metadata
        """
        self._check_data_input(data)

        # Получаем x из params или создаём стандартный
        x = np.arange(data.shape[0])

        slope, intercept, _, _, _ = stats.linregress(x=x, y=data)
        trend = slope * x + intercept
        features = self._extract_features(trend, data)

        return DecompositionResult(
            component=data - trend,
            component_type=FEComponentType.TREND,
            method_name='linear_trend',
            params=self.params.copy(),
            stats={'features': features}
        )