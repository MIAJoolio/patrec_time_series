from typing import Literal
import time

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

import statsmodels.api as sm
from statsmodels.tsa.tsatools import detrend
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.filters.cf_filter import cffilter
from scipy import stats 
from scipy.interpolate import UnivariateSpline

from sklearn.linear_model import LinearRegression

import pywt

from src.feature_extraction.fe_classes import Base_extractor, FE_result

__all__ = [
    'Linregress_detrender',
    'Statsmodels_detrender',
    'Polyfit_detrender',
    'Sklearn_detrender',
    'SMA_detrender', 
    'EMA_detrender',
    'HP_filter_detrender',
    'CF_filter_detrender',
    'STL_detrender',
    'Wavelet_detrender',
    'Spline_detrender'
]


class Linregress_detrender(Base_extractor):
    """Детрендинг с использованием линейной регрессии."""
    def __init__(self, alternative='two-sided', alpha=0.05):
        self.alternative = alternative
        self.alpha = alpha
        
    def extract(self, data: np.ndarray, x=None) -> FE_result:
        start_time = time.time()
        n = len(data)
        if x is None:
            x = np.arange(n)

        slope, intercept, rvalue, pvalue, stderr = stats.linregress(x=x, y=data)
        intercept_stderr = np.sqrt(np.sum((x - np.mean(x))**2) / n) * stderr
        trend = slope * x + intercept
        detrended = data - trend

        # Confidence interval
        df = n - 2
        ts = abs(stats.t.ppf(self.alpha / 2, df))
        slope_ci = (slope - ts * stderr, slope + ts * stderr)
        significant_slope = pvalue < self.alpha
        r_squared = rvalue ** 2

        # Quality checks
        quality_checks = {
            'rvalue_strong': abs(rvalue) >= 0.5,
            'pvalue_significant': pvalue < self.alpha,
            'stderr_small_relative_to_slope': stderr < 0.2 * abs(slope) if slope != 0 else False
        }

        return FE_result(
            component=trend,
            method_name='linregress_detrend',
            method_params={
                'x_provided': x is not None,
                'alternative': self.alternative,
                'alpha': self.alpha
            },
            execution_stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            results={
                'slope': slope,
                'intercept': intercept,
                'rvalue': rvalue,
                'r_squared': r_squared,
                'pvalue': pvalue,
                'stderr': stderr,
                'intercept_stderr': intercept_stderr,
                'slope_ci': slope_ci,
                'significant_slope': significant_slope,
                'quality_checks': quality_checks,
                'detrended': detrended
            }
        )


class Statsmodels_detrender(Base_extractor):
    """Детрендинг с использованием statsmodels.tsa.tsatools.detrend."""
    def __init__(self, order=1, axis=0):
        self.order = order
        self.axis = axis
        
    def extract(self, data: np.ndarray) -> FE_result:
        start_time = time.time()
        data = np.asarray(data)
        detrended = detrend(data, order=self.order, axis=self.axis)
        trend = data - detrended
        
        results = {}
        if self.order == 1 and self.axis == 0:
            x = np.arange(data.shape[self.axis])
            A = np.vstack([x, np.ones_like(x)]).T
            slope, intercept = np.linalg.lstsq(A, trend, rcond=None)[0]
            results.update({'slope': slope, 'intercept': intercept})
            results['detrended'] = detrended

        return FE_result(
            component=trend,
            method_name='statsmodels_detrend',
            method_params={'order': self.order, 'axis': self.axis},
            execution_stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            results=results
        )


class Polyfit_detrender(Base_extractor):
    """Детрендинг с использованием полиномиальной аппроксимации."""
    def __init__(self, degree=1):
        self.degree = degree
        
    def extract(self, data: np.ndarray, x=None) -> FE_result:
        start_time = time.time()
        n = len(data)
        if x is None:
            x = np.arange(n)

        coefficients = np.polyfit(x, data, deg=self.degree)
        trend = np.polyval(coefficients, x)
        detrended = data - trend

        # R-squared
        residuals = data - trend
        r_squared = 1 - (np.sum(residuals**2) / np.sum((data - np.mean(data))**2))

        return FE_result(
            component=trend,
            method_name='polyfit_detrend',
            method_params={'degree': self.degree, 'x_provided': x is not None},
            execution_stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            results={
                'coefficients': coefficients,
                'r_squared': r_squared,
                'detrended': detrended
            }
        )


class Sklearn_detrender(Base_extractor):
    """Детрендинг с использованием LinearRegression из sklearn."""
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        
    def extract(self, data: np.ndarray, x=None) -> FE_result:
        start_time = time.time()
        n = len(data)
        if x is None:
            x = np.arange(n).reshape(-1, 1)
        else:
            x = np.array(x).reshape(-1, 1)

        model = LinearRegression(fit_intercept=self.fit_intercept)
        model.fit(x, data)
        trend = model.predict(x)
        detrended = data - trend

        # R-squared
        r_squared = 1 - (np.sum((data - trend)**2) / np.sum((data - np.mean(data))**2))

        return FE_result(
            component=trend,
            method_name='sklearn_detrend',
            method_params={
                'x_provided': x is not None,
                'fit_intercept': self.fit_intercept
            },
            execution_stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            results={
                'slope': model.coef_[0],
                'intercept': model.intercept_ if self.fit_intercept else 0.0,
                'r_squared': r_squared,
                'detrended': detrended
            }
        )


class SMA_detrender(Base_extractor):
    """Детрендинг с использованием простого скользящего среднего."""
    def __init__(self, window=5):
        self.window = window
        
    def extract(self, data: np.ndarray) -> FE_result:
        start_time = time.time()
        data = np.array(data)
        sma = pd.Series(data).rolling(window=self.window, center=True).mean()
        sma = sma.fillna(method='bfill').fillna(method='ffill')
        detrended = data - sma

        return FE_result(
            component=sma.values,
            method_name='sma_detrend',
            method_params={'window': self.window},
            execution_stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            results={'detrended': detrended}
        )


class EMA_detrender(Base_extractor):
    """Детрендинг с использованием экспоненциального скользящего среднего."""
    def __init__(self, span=10):
        self.span = span
        
    def extract(self, data: np.ndarray) -> FE_result:
        start_time = time.time()
        data = np.array(data)
        ema = pd.Series(data).ewm(span=self.span, adjust=False).mean()
        detrended = data - ema

        return FE_result(
            component=ema.values,
            method_name='ema_detrend',
            method_params={'span': self.span},
            execution_stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            results={'detrended': detrended}
        )


class HP_filter_detrender(Base_extractor):
    """Детрендинг с использованием фильтра Ходрика-Прескотта."""
    def __init__(self, lamb=1600, regime='none'):
        self.lamb = lamb
        self.regime = regime
        
    def extract(self, data: np.ndarray) -> FE_result:
        start_time = time.time()
        if self.regime == 'annual':
            lamb = 1600/4**4
        elif self.regime == 'monthly':
            lamb = 1600*3**4
        else:
            lamb = self.lamb
        
        trend, cycle = hpfilter(data, lamb=lamb)
        
        return FE_result(
            component=trend,
            method_name='hpfilter_detrend',
            method_params={'lambda': lamb, 'regime': self.regime},
            execution_stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            results={'detrended': cycle}
        )


class CF_filter_detrender(Base_extractor):
    """Детрендинг с использованием фильтра Кристиано-Фицджеральда."""
    def __init__(self, low=6, high=32, drift=True):
        self.low = low
        self.high = high
        self.drift = drift
        
    def extract(self, data: np.ndarray) -> FE_result:
        start_time = time.time()
        cycle, trend = cffilter(data, low=self.low, high=self.high, drift=self.drift)
        
        return FE_result(
            component=trend,
            method_name='cffilter_detrend',
            method_params={'low': self.low, 'high': self.high, 'drift': self.drift},
            execution_stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            results={'detrended': cycle}
        )


class STL_detrender(Base_extractor):
    """Детрендинг с использованием STL decomposition."""
    def __init__(self, period=None, lo_frac=0.6, lo_delta=0.01):
        self.period = period
        self.lo_frac = lo_frac
        self.lo_delta = lo_delta
        
    def extract(self, data: np.ndarray) -> FE_result:
        start_time = time.time()
        stl = STL(data, period=self.period, lo_frac=self.lo_frac, lo_delta=self.lo_delta)
        result = stl.fit()
        
        return FE_result(
            component= result.trend,
            method_name='stl_detrend',
            method_params={
                'period': self.period,
                'lo_frac': self.lo_frac,
                'lo_delta': self.lo_delta
            },
            execution_stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            results={
                'detrended': data - result.trend,
                'seasonal': result.seasonal,
                'resid': result.resid
            }
        )

        
class Wavelet_detrender(Base_extractor):
    """Детрендинг через wavelet-разложение"""
    def __init__(self, wavelet='db4', level=None, threshold=0.1, mode='symmetric', 
                 threshold_type='soft', energy_threshold=0.9):
        """
        Args:
            wavelet: Тип вейвлета ('db4', 'sym5' и др.)
            level: Уровень декомпозиции (None для автоопределения)
            threshold: Порог для отсечения коэффициентов (0-1)
            mode: Режим дополнения сигнала ('symmetric', 'zero', etc.)
            threshold_type: Тип порога ('soft', 'hard')
            energy_threshold: Порог сохранения энергии (для автоопределения уровня)
        """
        self.wavelet = wavelet
        self.level = level
        self.threshold = threshold
        self.mode = mode
        self.threshold_type = threshold_type
        self.energy_threshold = energy_threshold

    def extract(self, data: np.ndarray) -> FE_result:
        start_time = time.time()
        data = np.asarray(data)
        
        # Автоподбор уровня декомпозиции
        if self.level is None:
            max_level = pywt.dwt_max_level(len(data), self.wavelet)
            # Подбираем уровень, сохраняющий 90% энергии
            coeffs = pywt.wavedec(data, self.wavelet, level=max_level, mode=self.mode)
            energy = np.cumsum([np.sum(c**2) for c in coeffs[::-1]])[::-1]
            total_energy = energy[0]
            level = np.argmax(energy/total_energy > self.energy_threshold) + 1
        else:
            level = self.level

        # Wavelet-разложение
        coeffs = pywt.wavedec(data, self.wavelet, level=level, mode=self.mode)
        
        # Анализ коэффициентов перед пороговой обработкой
        coeffs_energy = [np.sum(c**2) for c in coeffs]
        total_energy = np.sum(coeffs_energy)
        
        # Адаптивный порог для каждого уровня
        thresholds = [self.threshold * np.max(np.abs(c)) for c in coeffs]
        
        # Пороговая обработка коэффициентов
        coeffs_thresh = [
            pywt.threshold(c, thresh, self.threshold_type) 
            for c, thresh in zip(coeffs, thresholds)
        ]
        
        # Реконструкция тренда (используем только аппроксимирующие коэффициенты)
        trend_coeffs = [coeffs_thresh[0]] + [None]*(len(coeffs)-1)
        trend = pywt.waverec(trend_coeffs, self.wavelet, mode=self.mode)
        trend = trend[:len(data)]  # Обрезка до исходной длины
        
        detrended = data - trend
        
        # Анализ после обработки
        residual_energy = np.sum(detrended**2)
        energy_retained = 1 - residual_energy/total_energy

        return FE_result(
            component=trend,
            method_name='wavelet_detrend',
            method_params={
                'wavelet': self.wavelet,
                'level': level,
                'threshold': self.threshold,
                'mode': self.mode,
                'threshold_type': self.threshold_type,
                'energy_threshold': self.energy_threshold
            },
            execution_stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            results={
                'detrended': detrended,
                'coeffs_energy': coeffs_energy,
                'energy_retained': energy_retained,
                'residual_energy': residual_energy,
                'thresholds_applied': thresholds,
                'coeffs_shape': [c.shape for c in coeffs]
            }
        )


class Spline_detrender(Base_extractor):
    """Детрендинг через сплайн-аппроксимацию с автоматическим подбором параметров"""
    def __init__(self, s=None, k=3, auto_smooth=True, n_knots=None, 
                 knot_selection='quantile'):
        """
        Args:
            s: Параметр сглаживания (None для автоопределения)
            k: Степень сплайна (1-5)
            auto_smooth: Автоматически подбирать параметр сглаживания
            n_knots: Количество узлов (None для автоматического определения)
            knot_selection: Метод выбора узлов ('quantile' или 'uniform')
        """
        self.s = s
        self.k = k
        self.auto_smooth = auto_smooth
        self.n_knots = n_knots
        self.knot_selection = knot_selection

    def extract(self, data: np.ndarray) -> FE_result:
        start_time = time.time()
        x = np.arange(len(data))
        
        # Автоподбор количества узлов
        if self.n_knots is None:
            n_knots = max(3, min(20, len(data)//10))
        else:
            n_knots = self.n_knots
            
        # Выбор положения узлов
        if self.knot_selection == 'quantile':
            knots = np.quantile(x, np.linspace(0, 1, n_knots))
        else:  # uniform
            knots = np.linspace(x.min(), x.max(), n_knots)
        
        # Автоподбор параметра сглаживания
        if self.auto_smooth and self.s is None:
            # Эмпирическая формула на основе стандартного отклонения
            noise_est = np.std(data)
            self.s = noise_est * np.sqrt(len(data)) * 0.5
            
        # Создание сплайна
        spline = LSQUnivariateSpline(x, data, knots[1:-1], k=self.k)
        
        if self.auto_smooth:
            # Оптимизация параметра сглаживания
            def objective(s):
                spline.set_smoothing_factor(s)
                residuals = data - spline(x)
                return np.sum(residuals**2) + s * np.sum(spline.get_coeffs()**2)
            
            from scipy.optimize import minimize_scalar
            opt_result = minimize_scalar(objective, bounds=(0, 10*self.s), method='bounded')
            spline.set_smoothing_factor(opt_result.x)
            final_s = opt_result.x
        else:
            final_s = self.s if self.s is not None else 0.0
            spline.set_smoothing_factor(final_s)
        
        trend = spline(x)
        detrended = data - trend
        
        # Расчет метрик качества
        ss_res = np.sum(detrended**2)
        ss_tot = np.sum((data - np.mean(data))**2)
        r_squared = 1 - ss_res/ss_tot
        
        return FE_result(
            component=trend,
            method_name='spline_detrend',
            method_params={
                'smoothing_factor': final_s,
                'spline_order': self.k,
                'n_knots': n_knots,
                'knot_selection': self.knot_selection,
                'auto_smooth': self.auto_smooth
            },
            execution_stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            results={
                'detrended': detrended,
                'knots': spline.get_knots(),
                'r_squared': r_squared,
                'residual_std': np.std(detrended),
                'effective_degrees_of_freedom': spline.get_coeffs().size,
                'condition_number': np.linalg.cond(spline.get_coeffs())
            }
        )