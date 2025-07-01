from typing import Literal
import time

import numpy as np

import pywt

from tslearn.shapelets import LearningShapelets

from src.feature_extraction.fe_classes import Base_extractor, FE_result


class DFT_seasonality(Base_extractor):
    def __init__(self, n_freqs='auto', fs=1.0, min_power_ratio=0.1, min_freq=None, max_freq=None):
        self.n_freqs = n_freqs
        self.fs = fs
        self.min_power_ratio = min_power_ratio
        self.min_freq = min_freq
        self.max_freq = max_freq
        
    def extract(self, data: np.ndarray) -> FE_result:
        start_time = time.time()
        n = len(data)
        
        # Вычисление DFT
        dft = np.fft.fft(data)
        magnitude = np.abs(dft)
        power = magnitude ** 2
        freq = np.fft.fftfreq(n, d=1/self.fs)
        
        # Фильтрация частот по заданному диапазону
        pos_mask = freq > 0
        if self.min_freq is not None:
            pos_mask &= (freq >= self.min_freq)
        if self.max_freq is not None:
            pos_mask &= (freq <= self.max_freq)
            
        pos_freq = freq[pos_mask]
        pos_power = power[pos_mask]
        
        # Автоподбор числа частот
        if self.n_freqs == 'auto':
            total_power = np.sum(pos_power)
            sorted_idx = np.argsort(pos_power)[::-1]
            cum_power = np.cumsum(pos_power[sorted_idx]) / total_power
            n_freqs = np.argmax(cum_power > self.min_power_ratio) + 1
        else:
            n_freqs = min(self.n_freqs, len(pos_freq))
            
        # Выбор наиболее значимых частот
        significant_idx = np.argpartition(pos_power, -n_freqs)[-n_freqs:]
        main_freqs = pos_freq[significant_idx]
        main_phases = np.angle(dft[np.where(pos_mask)[0][significant_idx]])
        main_amplitudes = magnitude[np.where(pos_mask)[0][significant_idx]]
        
        # Реконструкция сезонного компонента
        reconstructed = np.zeros(n, dtype=complex)
        for i, idx in enumerate(np.where(pos_mask)[0][significant_idx]):
            freq_val = freq[idx]
            reconstructed += (dft[idx] * np.exp(2j*np.pi*freq_val*np.arange(n)/n)) / n
            if freq_val != 0:  # Добавляем симметричную компоненту для вещественных сигналов
                reconstructed += (dft[-idx] * np.exp(-2j*np.pi*freq_val*np.arange(n)/n) / n)
        
        seasonal_component = np.real(reconstructed)
        
        return FE_result(
            component=seasonal_component,
            method_name='dft_seasonality',
            method_params={
                'n_freqs': n_freqs,
                'fs': self.fs,
                'min_power_ratio': self.min_power_ratio,
                'min_freq': self.min_freq,
                'max_freq': self.max_freq
            },
            execution_stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            results={
                'main_frequencies': main_freqs,
                'main_phases': main_phases,
                'main_amplitudes': main_amplitudes,
                'power_spectrum': power[pos_mask]
            }
        )


class DWT_seasonality(Base_extractor):
    def __init__(self, wavelet='db8', mode='symmetric', level='auto', 
                 threshold='universal', seasonality_levels=None):
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        self.threshold = threshold
        self.seasonality_levels = seasonality_levels
        
    def extract(self, data: np.ndarray) -> FE_result:
        start_time = time.time()
        n = len(data)
        
        # Автонастройка уровня декомпозиции
        if self.level == 'auto':
            level = pywt.dwt_max_level(n, self.wavelet)
        else:
            level = self.level
            
        # Wavelet-разложение
        coeffs = pywt.wavedec(data, self.wavelet, level=level, mode=self.mode)
        
        # Автоподбор порога
        if self.threshold == 'universal':
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(n))
        else:
            threshold = self.threshold
            
        # Определение уровней, содержащих сезонность
        if self.seasonality_levels is None:
            # По умолчанию используем все уровни кроме последнего (тренд)
            seasonality_levels = list(range(1, len(coeffs)))
        else:
            seasonality_levels = self.seasonality_levels
            
        # Обработка коэффициентов
        seasonal_coeffs = []
        for i, c in enumerate(coeffs):
            if i in seasonality_levels:
                seasonal_coeffs.append(pywt.threshold(c, threshold, 'soft'))
            else:
                seasonal_coeffs.append(np.zeros_like(c))
        
        # Реконструкция сезонного компонента
        seasonal_coeffs[0] = np.zeros_like(coeffs[0])  # Игнорируем аппроксимирующие коэффициенты
        seasonal = pywt.waverec(seasonal_coeffs, self.wavelet)
        
        # Обрезка до исходной длины (wavelet может добавлять padding)
        seasonal = seasonal[:n]
        
        # Расчет энергии компонентов
        energy = [np.sum(c**2) for c in coeffs]
        seasonal_energy = np.sum(seasonal**2)
        total_energy = np.sum(data**2)
        
        return FE_result(
            component=seasonal,
            method_name='dwt_seasonality',
            method_params={
                'wavelet': self.wavelet,
                'level': level,
                'threshold': threshold,
                'seasonality_levels': seasonality_levels,
                'mode': self.mode
            },
            execution_stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            results={
                'coeffs_energy': energy,
                'seasonal_energy_ratio': seasonal_energy/total_energy,
                'threshold_value': threshold
            }
        )


class Shapelet_extractor(Base_extractor):
    def __init__(self, n_shapelets=5, min_len=10, max_len=50, max_iter=100):
        self.n_shapelets = n_shapelets
        self.len_bounds = (min_len, max_len)
        self.max_iter = max_iter
        
    def extract(self, data: np.ndarray, y=None) -> FE_result:
        """y - метки классов для обучения shapelets"""
        start_time = time.time()
        
        if y is None:
            raise ValueError("Для Shapelet-анализа требуются метки классов")
            
        # Пример реализации с использованием tslearn
        model = LearningShapelets(n_shapelets=self.n_shapelets,
                                 max_iter=self.max_iter,
                                 verbose=0)
        model.fit(data.reshape(1, -1, 1), y)  # Требуется 3D-формат
        
        # Извлечение shapelets
        shapelets = model.shapelets_.reshape(self.n_shapelets, -1)
        distances = model.transform(data.reshape(1, -1, 1))
        
        return FE_result(
            component=distances.flatten(),
            method_name='shapelet_extraction',
            method_params={
                'n_shapelets': self.n_shapelets,
                'length_bounds': self.len_bounds,
                'max_iter': self.max_iter
            },
            execution_stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            results={
                'shapelets': shapelets,
                'shapelet_distances': distances
            }
        )