from typing import Tuple, Dict, Union
import warnings
# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.simplefilter('ignore', InterpolationWarning)

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

def get_hurst_exponent(ts, max_lag=20):
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    
    return np.polyfit(np.log(lags), np.log(tau), 1)[0].round(4)

def hurst_exponent(ts, lags=None, weights=None):
    """
    Mean hurst
    """    
    if not lags:
        lags = np.asarray([20, 100, 200, 500, 1000])
        lags = lags[lags < len(ts)]

    if not weights:
        weights = [1/(len(lags)+1) if i >= 2 else 1/len(lags) for i in range(len(lags))]

    hurst_est = []
    
    for lag in lags:
        hurst_est.append(get_hurst_exponent(ts, lag))
    
    return np.mean(hurst_est).round(4), {'lags':lags, 'weights':weights, 'hurst_est':hurst_est}

def test_stationarity(ts: np.ndarray, alpha: float = 0.05) -> Tuple[bool, Dict[str, Union[float, str]]]:
    """
    Test time series stationarity using ADF, KPSS, and Hurst exponent.
    
    Args:
        ts: Time series data.
        alpha: Significance level (default: 0.05).
    
    Returns:
        Tuple of:
        - is_stationary (bool): True if series is likely stationary.
        - results (dict): Test statistics and interpretations.
    """
    # ADF Test (Null: Non-stationary)
    adf_result = adfuller(ts)
    adf_p = adf_result[1]
    adf_reject_null = adf_p <= alpha
    
    # KPSS Test (Null: Stationary)
    kpss_result = kpss(ts, regression='c')  # 'c' for constant, 'ct' for trend
    kpss_p = kpss_result[1]
    kpss_reject_null = kpss_p <= alpha
    
    # Hurst Exponent
    hurst, hurst_est = hurst_exponent(ts)
    hurst_behavior = (
        "Random Walk (H ≈ 0.5)" if 0.45 <= hurst <= 0.55 else
        "Mean-Reverting (H < 0.5)" if hurst < 0.45 else
        "Trending (H > 0.5)"
    )
    
    # Combined Decision Logic
    is_stationary = (
        (not kpss_reject_null) and  
        (adf_reject_null) and       
        (hurst < 0.5)               
    )
    
    return is_stationary, {
        "ADF Test": {
            "p-value": adf_p,
            "test_result": "Stationary" if adf_reject_null else "Non-Stationary",
        },
        "KPSS Test": {
            "p-value": kpss_p,
            "test_result": "Non-Stationary" if kpss_reject_null else "Stationary",
        },
        "Hurst Exponent": {
            "value": hurst,
            "test_result": hurst_behavior,
        },
        "combined_result": "Stationary" if is_stationary else "Non-Stationary",
    }
