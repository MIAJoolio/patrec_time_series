from typing import Any

import numpy as np

from patrec_ts.feature_extraction.fe_classes import DecompositionResult, FEComponentType, BaseOptimizer
from patrec_ts.preprocessing.manual_methods.parametric.base_parametric_method import BaseParametricPreprocessor


class TrendPolyPreprocessor(BaseParametricPreprocessor):
    """Polynomial trend decomposition with params dict"""

    def __init__(self, params: dict[str, Any] = None):
        self.params = {'degree': 1} | (params or {})

    def _extract_features(self, trend: np.ndarray, data: np.ndarray) -> dict[str, float]:
        """Feature extraction specific for polynomial trend"""
        curvature = np.mean(np.abs(np.diff(np.diff(trend))))
        return {
            'trend_slope': np.mean(trend),
            'trend_poly_degree': self.params['degree'],
            'trend_curvature': curvature,
            'trend_residual_std': np.std(data - trend)
        }

    def decompose(self, data: np.ndarray, opt: bool = False, **kwargs) -> DecompositionResult:
        """
        Decomposes time series into polynomial trend component with optional parameter optimization.
        """
        self._check_data_input(data)
        x = np.arange(len(data))

        if opt:
            self._optimize_params(data, x)

        degree: int = self.params['degree']

        coeffs = np.polyfit(x, data, deg=degree)

        trend = np.polyval(coeffs, x)
        features = self._extract_features(trend, data)

        return DecompositionResult(
            component=data - trend,
            component_type=FEComponentType.TREND,
            method_name='polynomial_trend',
            params=self.params.copy(),
            stats={'features': features}
        )

    # def _optimize_params(self, data: np.ndarray, x: np.ndarray):
    #     """Safe parameter optimization"""
    #     param_grid = {
    #         'degree': {
    #             'type': 'int',
    #             'low': 1,
    #             'high': 3,
    #         }
    #     }
    #
    #     result = BaseOptimizer.optimize(
    #         decomposer_class=lambda **p: Poly_trend_decomposer({'x': x, **p}),
    #         data=data,
    #         param_grid=param_grid,
    #         n_trials=min(50, len(data)),
    #         timeout=30,
    #         max_failures=5
    #     )
    #
    #     if result['best_score'] == float('inf'):
    #         raise RuntimeError("Parameter optimization failed")
    #
    #     self.params['degree'] = result['best_params']['degree']