import numpy as np
from aeon.transformations.collection.base import BaseCollectionTransformer
# from aeon.transformations.collection import (
#     MinMaxScaler,
#     Centerer, 
#     Normalizer, 
#     # полезная штука очень наглядная для небольших рядов
#     AutocorrelationFunctionTransformer,
#     # что-то есть
#     ARCoefficientTransformer,
#     # уменьшение длины ряда 
#     DownsampleTransformer,
#     # не понятно как считается и что дальше делать (просто fe)
#     DWTTransformer,
#     # (просто fe)
#     HOG1DTransformer,
#     # (просто fe), но и значения адеватно делят
#     PeriodogramTransformer,
#     MatrixProfile,
#     # базовые fe функции
#     SlopeTransformer,
#     # для заполнения пропусков (таких нет)
#     SimpleImputer
# )


class EMA_Transformer(BaseCollectionTransformer):
    """Exponential Moving Average (EMA) transformer for time series collections.
    
    Applies exponential moving average smoothing to each time series in the collection.
    The smoothing factor alpha controls the degree of smoothing (0 < alpha <= 1).
    Higher values of alpha reduce the smoothing effect.

    Parameters
    ----------
    alpha : float, default=0.3
        Smoothing factor between 0 and 1. Higher values mean less smoothing.
    adjust : bool, default=False
        Whether to use the adjusted EMA formula that accounts for series beginnings.
    """

    _tags = {
        "X_inner_type": ["numpy3D", "np-list"],
        "fit_is_empty": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
    }

    def __init__(self, alpha=0.3, adjust=False):
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.adjust = adjust
        super().__init__()

    def _transform(self, X, y=None) -> np.ndarray:
        """
        Apply EMA smoothing to the time series collection.

        Parameters
        ----------
        X: np.ndarray or list
            Collection to transform. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        y: None
           Ignored.

        Returns
        -------
        np.ndarray
            The smoothed time series collection with same shape as input.
        """
        if isinstance(X, list):
            Xt = [self._apply_ema(x) for x in X]
            return Xt
        else:
            Xt = np.zeros_like(X)
            for i in range(X.shape[0]):
                Xt[i] = self._apply_ema(X[i])
            return Xt

    def _apply_ema(self, series):
        """Apply EMA to a single series (n_channels, n_timepoints)."""
        result = np.zeros_like(series)
        alpha = self.alpha
        
        for channel in range(series.shape[0]):
            if self.adjust:
                # Adjusted EMA version
                weights = (1 - alpha) ** np.arange(len(series[channel])-1, -1, -1)
                weights /= weights.sum()
                result[channel] = np.convolve(series[channel], weights, mode='full')[:len(series[channel])]
            else:
                # Standard EMA
                result[channel, 0] = series[channel, 0]
                for t in range(1, series.shape[1]):
                    result[channel, t] = alpha * series[channel, t] + (1 - alpha) * result[channel, t-1]
        
        return result


class SMA_Transformer(BaseCollectionTransformer):
    """Simple Moving Average (SMA) transformer for time series collections.
    
    Applies simple moving average smoothing to each time series in the collection.
    The window size controls the number of points used for averaging.

    Parameters
    ----------
    window_size : int, default=3
        Size of the moving window. Must be at least 1.
    center : bool, default=False
        If True, the result is centered in the window. Otherwise, it's the trailing average.
    """

    _tags = {
        "X_inner_type": ["numpy3D", "np-list"],
        "fit_is_empty": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
    }

    def __init__(self, window_size=3, center=False):
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.window_size = window_size
        self.center = center
        super().__init__()

    def _transform(self, X, y=None) -> np.ndarray:
        """
        Apply SMA smoothing to the time series collection.

        Parameters
        ----------
        X: np.ndarray or list
            Collection to transform. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        y: None
           Ignored.

        Returns
        -------
        np.ndarray
            The smoothed time series collection with same shape as input.
        """
        if isinstance(X, list):
            Xt = [self._apply_sma(x) for x in X]
            return Xt
        else:
            Xt = np.zeros_like(X)
            for i in range(X.shape[0]):
                Xt[i] = self._apply_sma(X[i])
            return Xt

    def _apply_sma(self, series):
        """Apply SMA to a single series (n_channels, n_timepoints)."""
        result = np.zeros_like(series)
        window = self.window_size
        half_window = window // 2
        
        for channel in range(series.shape[0]):
            if self.center:
                # Centered moving average
                for t in range(series.shape[1]):
                    start = max(0, t - half_window)
                    end = min(series.shape[1], t + half_window + 1)
                    result[channel, t] = np.mean(series[channel, start:end])
            else:
                # Trailing moving average
                for t in range(series.shape[1]):
                    start = max(0, t - window + 1)
                    end = t + 1
                    result[channel, t] = np.mean(series[channel, start:end])
        
        return result