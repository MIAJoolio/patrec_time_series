import os
from typing import Literal, Union, Optional

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

__all__ = [
    # stationarity test
    'test_stationarity',
    'hurst_exp',    
    
]

def hurst_exp(ts):
    lags = range(2, 100)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]


def test_stationarity(ts):
    adf_p = adfuller(ts)[1]  # Null: Non-stationary
    kpss_p = kpss(ts)[1]     # Null: Stationary
    hurst = hurst_exp(ts)
    est_stationary = (0.45 <= hurst <= 0.55) & (adf_p <= 0.05) & (kpss_p <= 0.05)
    return est_stationary, {
        "ADF p-value": adf_p,
        "KPSS p-value": kpss_p,
        "Hurst Exponent": hurst
    }


