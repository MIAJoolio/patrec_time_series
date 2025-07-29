import numpy as np

# trend + autocorrelation, statistics
from statsmodels.tsa.tsatools import detrend
from statsmodels.tsa.stattools import acf
from scipy.stats import skew, kurtosis

# piecewise 
# from tslearn.piecewise import PiecewiseAggregateApproximation

# анализ пиков 
from scipy.signal import argrelmin, argrelmax, find_peaks, find_peaks_cwt, peak_prominences, peak_widths
# 2
from scipy.signal import stft, find_peaks

# dft
from scipy.fft import fft, ifft

# dwt
import pywt

from statsmodels.tsa.seasonal import STL
from scipy.signal import argrelmax


__all__ = [
    # Детрендирование
    'tsa_detrend',
    # Статистические признаки
    'statistical_features',
    # PAA
    'paa_features',
    # Автокорреляция
    'tsa_acf',
    # Анализ пиков
    'signal_peaks_features',
    # STFT
    'stft_features',
    # DFT
    'dft_components',
    'dft_signal',
    'dft_approximation',
    # DWT
    'dwt_features',
    'dwt_signal',
    # STL decomposition
    'stl_trend',
    'stl_seasonal',
    'stl_noise',
    'stl_features',
    'no_fe'
]

def no_fe(series):
    return series


# 17.06 checked
def tsa_detrend(series, order=0):
    return detrend(series, order=order)


def tsa_acf(series, n_lags=10, thresh=0.5):
    acf_values = acf(series, nlags=n_lags+1)    
    return acf_values[1:]  

    
def statistical_features(series):
    return np.array([
        np.mean(series),
        np.var(series),
        skew(series),
        kurtosis(series),
        np.mean(np.abs(np.diff(series))),  # Mean absolute difference
        np.sum(np.diff(series) > 0) / len(series)  # Zero-crossing rate
    ])


def paa_features(series, n_segments=5):
    if len(series.shape) == 1:
        series = series.reshape(1, -1)

    paa = PiecewiseAggregateApproximation(n_segments=n_segments)
    return paa.fit_transform(series).flatten()


def signal_peaks_features(series):
    """
    Анализирует временной ряд и возвращает вектор признаков.

    Параметры:
    data (np.array): Временной ряд (1D массив).
    widths (np.array): Диапазон ширины пиков для find_peaks_cwt.

    Возвращает:
    features (dict): Словарь с признаками временного ряда.
    """
    
    widths=np.arange(1, len(series))
    
    
    features = []
    # 1. Нахождение относительных минимумов и максимумов
    rel_min_indices = argrelmin(series)[0]
    rel_max_indices = argrelmax(series)[0]

    features.append(len(rel_min_indices))  # Количество относительных минимумов
    features.append(len(rel_max_indices))  # Количество относительных максимумов

    # 2. Нахождение пиков с помощью find_peaks
    peaks, properties = find_peaks(series)
    num_peaks = len(peaks)
    features.append(num_peaks)  # Количество пиков

    if num_peaks > 0:
        # 3. Высота пиков
        peak_heights = series[peaks]
        features.append(np.mean(peak_heights))  # Средняя высота пиков
        features.append(np.max(peak_heights))   # Максимальная высота пиков
        features.append(np.min(peak_heights))   # Минимальная высота пиков

        # 4. Prominence (выдающаяся часть пика)
        prominences = peak_prominences(series, peaks)[0]
        features.append(np.mean(prominences))  # Средняя prominence
        features.append(np.max(prominences))   # Максимальная prominence
        features.append(np.min(prominences))   # Минимальная prominence

        # 5. Ширина пиков
        widths, _, _, _ = peak_widths(series, peaks)
        features.append(np.mean(widths))  # Средняя ширина пиков
        features.append(np.max(widths))   # Максимальная ширина пиков
        features.append(np.min(widths))   # Минимальная ширина пиков
    else:
        # Если пиков нет, добавляем нули
        features.extend([0] * 7)  # 7 признаков, связанных с пиками

    # 6. Нахождение пиков с помощью find_peaks_cwt
    cwt_peaks = find_peaks_cwt(series, widths)
    num_cwt_peaks = len(cwt_peaks)
    features.append(num_cwt_peaks)  # Количество пиков, найденных через CWT

    if num_cwt_peaks > 0:
        cwt_peak_heights = series[cwt_peaks]
        features.append(np.mean(cwt_peak_heights))  # Средняя высота CWT пиков
        features.append(np.max(cwt_peak_heights))   # Максимальная высота CWT пиков
        features.append(np.min(cwt_peak_heights))   # Минимальная высота CWT пиков
    else:
        # Если пиков нет, добавляем нули
        features.extend([0] * 3)  # 3 признака, связанных с CWT пиками

    # Преобразуем список признаков в np.ndarray
    return np.array(features, dtype=np.float32)


def stft_features(series, noverlap=None, nperseg:int=None):
    if nperseg is None: 
        nperseg= len(series)
    # time_series = time_series.reshape(time_series.shape[1], time_series.shape[0])
    _, _, Zxx = stft(series, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx).flatten()


def dft_components(series, n_freqs=10, fs=500):
    """
    Функция:
    * Применяет быстрое преобразование Фурье (FFT) к временному ряду.
    * Вычисляет амплитуды, частоты и фазы наиболее значимых компонент, т.е. n_freqs
    """
    if not isinstance(n_freqs, int):
        n_freqs = int(n_freqs)
        
    dft_values = fft(series)
    magnitude = np.abs(dft_values)
    phases =  np.angle(dft_values)
    
    n = len(series)
    freq = np.fft.fftfreq(n, d=1/fs) 

    # Только положительные частоты, т.к. в отрицательных нет информации
    positive_freq = freq[:n // 2]  
    # Соответствующие амплитуды
    positive_magnitude = magnitude[:n // 2] 
    # индексы наибольших значение мощности
    significant = np.argsort(positive_magnitude)[-n_freqs:]

    return np.array([[positive_magnitude[j], phases[j]] for j in significant]).flatten()


def dft_signal(series, n_freqs=10, fs=500):
    """
    Функция:
    * Применяет быстрое преобразование Фурье (FFT) к временному ряду.
    * Вычисляет амплитуды, частоты и фазы наиболее значимых компонент.
    * Восстанавливает временной ряд на основе этих компонент.

    Параметры:
    - time_series: Входной временной ряд.
    - n_freqs: Количество наиболее значимых частотных компонент.

    Возвращает:
    - significant_components: Наиболее значимые компоненты (амплитуды и фазы).
    - restored_signal: Приближенный временной ряд.
    """
    if not isinstance(n_freqs, int):
        n_freqs = int(n_freqs)
         
    # Применяем FFT
    dft_values = fft(series)
    magnitude = np.abs(dft_values)
    phases = np.angle(dft_values)
    
    n = len(series)
    freq = np.fft.fftfreq(n, d=1/fs)

    # Только положительные частоты
    positive_freq = freq[:n // 2]
    positive_magnitude = magnitude[:n // 2]
    positive_phases = phases[:n // 2]

    # Индексы наибольших амплитуд
    significant_indices = np.argsort(positive_magnitude)[-n_freqs:]

    # Создаем массив для восстановленного спектра
    restored_spectrum = np.zeros(n, dtype=complex)
    for idx in significant_indices:
        restored_spectrum[idx] = positive_magnitude[idx] * np.exp(1j * positive_phases[idx])
        restored_spectrum[-idx] = positive_magnitude[idx] * np.exp(-1j * positive_phases[idx]) 

    # Обратное преобразование Фурье
    restored_signal = ifft(restored_spectrum).real

    return restored_signal


def dft_approximation(series, n_freqs=10, fs=500):
    """
    Функция:
    * Применяет быстрое преобразование Фурье (FFT) к временному ряду.
    * Вычисляет амплитуды, частоты и фазы наиболее значимых компонент.
    * Строит приближенную синусоиду на основе этих компонент.

    Параметры:
    - time_series: Входной временной ряд.
    - n_freqs: Количество наиболее значимых частотных компонент.

    Возвращает:
    - significant_components: Наиболее значимые компоненты (амплитуды, частоты и фазы).
    - approximated_signal: Приближенная синусоида.
    """
    if not isinstance(n_freqs, int):
        n_freqs = int(n_freqs)
        
    # Применяем FFT
    dft_values = fft(series)
    magnitude = np.abs(dft_values)
    phases = np.angle(dft_values)
    
    n = len(series)
    freq = np.fft.fftfreq(n, d=1/fs)

    # Только положительные частоты
    positive_freq = freq[:n // 2]
    positive_magnitude = magnitude[:n // 2]
    positive_phases = phases[:n // 2]

    # Индексы наибольших амплитуд
    significant_indices = np.argsort(positive_magnitude)[-n_freqs:]

    # Собираем значимые компоненты
    significant_components = []
    for idx in significant_indices:
        amplitude = positive_magnitude[idx] * 2 / n  # Нормализация амплитуды
        frequency = positive_freq[idx]
        phase = positive_phases[idx]
        significant_components.append((amplitude, frequency, phase))

    # Строим приближенную синусоиду
    t = np.arange(n) / fs  # Временная ось
    approximated_signal = np.zeros(n)
    for amplitude, frequency, phase in significant_components:
        approximated_signal += amplitude * np.sin(2 * np.pi * frequency * t + phase)

    return approximated_signal


def dwt_features(series, wavelet='db1', mode='symmetric', level=20, n_coeffs=15):
    """
    Функция:
    * Применяет дискретное вейвлет-преобразование (DWT) к временному ряду.
    * Вычисляет наиболее значимые коэффициенты (аппроксимации и детализации).
    * Возвращает признаки, основанные на этих коэффициентах.

    Параметры:
    - time_series: Входной временной ряд.
    - wavelet: Тип вейвлета (по умолчанию 'db4' — вейвлет Добеши 4-го порядка).
    - level: Уровень декомпозиции.
    - n_coeffs: Количество наиболее значимых коэффициентов для возврата.

    Возвращает:
    - Наиболее значимые коэффициенты DWT.
    """
    if not isinstance(n_coeffs, int):
        n_coeffs = int(n_coeffs)
    if not isinstance(level, int):
        level = int(level)
        
    # Выполняем вейвлет-преобразование
    coeffs = pywt.wavedec(series, wavelet, mode=mode, level=level)
    
    # # # Объединяем все коэффициенты в один массив
    all_coeffs = np.concatenate(coeffs)
    
    # # Находим наиболее значимые коэффициенты (по абсолютной величине)
    significant_indices = np.argsort(np.abs(all_coeffs))[-n_coeffs:]
    significant_coeffs = all_coeffs[significant_indices]
    
    return significant_coeffs[:n_coeffs]


def dwt_signal(series, wavelet='db4', mode='symmetric', level=4, n_coeffs=15):
    """
    Функция:
    * Применяет дискретное вейвлет-преобразование (DWT) к временному ряду.
    * Вычисляет наиболее значимые коэффициенты (аппроксимации и детализации).
    * Восстанавливает временной ряд на основе этих коэффициентов.

    Параметры:
    - time_series: Входной временной ряд.
    - wavelet: Тип вейвлета (по умолчанию 'db4' — вейвлет Добеши 4-го порядка).
    - mode: Режим дополнения границ.
    - level: Уровень декомпозиции.
    - n_coeffs: Количество наиболее значимых коэффициентов для возврата.

    Возвращает:
    - significant_coeffs: Наиболее значимые коэффициенты DWT.
    - restored_signal: Приближенный временной ряд.
    """
    if not isinstance(n_coeffs, int):
        n_coeffs = int(n_coeffs)
    if not isinstance(level, int):
        level = int(level)
        
    # Выполняем вейвлет-преобразование
    coeffs = pywt.wavedec(series, wavelet, mode=mode, level=level)
    
    # Объединяем все коэффициенты в один массив
    all_coeffs = np.concatenate(coeffs)
    
    # Находим наиболее значимые коэффициенты (по абсолютной величине)
    significant_indices = np.argsort(np.abs(all_coeffs))[-n_coeffs:]
    significant_coeffs = all_coeffs[significant_indices]
    
    # Обнуляем менее значимые коэффициенты
    for i in range(len(coeffs)):
        coeffs[i][np.abs(coeffs[i]) < np.mean(np.abs(significant_coeffs))] = 0
    
    # Обратное вейвлет-преобразование
    restored_signal = pywt.waverec(coeffs, wavelet)
    
    return restored_signal


def stl_trend(series, period=None):
    """
    Выполняет STL-декомпозицию временного ряда.
    
    Parameters:
    - series: np.array, одномерный временной ряд.
    - period: int, период сезонности (если None, автоматически определяется через ACF).
    
    Returns:
    - trend: Трендовая компонента.
    - seasonal: Сезонная компонента.
    - resid: Остаточная компонента.
    """
    if period == 1:
        acf_values = acf(series, nlags=40)
        peaks = argrelmax(acf_values)[0]
        if len(peaks) > 0:
            period = peaks[0]
        else:
            period = 10  # Значение по умолчанию
    
    stl = STL(series, period=period)
    result = stl.fit()
    
    return result.trend

def stl_seasonal(series, period=None):
    """
    Выполняет STL-декомпозицию временного ряда.
    
    Parameters:
    - series: np.array, одномерный временной ряд.
    - period: int, период сезонности (если None, автоматически определяется через ACF).
    
    Returns:
    - trend: Трендовая компонента.
    - seasonal: Сезонная компонента.
    - resid: Остаточная компонента.
    """
    if period == 1:
        acf_values = acf(series, nlags=40)
        peaks = argrelmax(acf_values)[0]
        if len(peaks) > 0:
            period = peaks[0]
        else:
            period = 10  # Значение по умолчанию
    
    stl = STL(series, period=period)
    result = stl.fit()
    
    return result.seasonal

def stl_noise(series, period=None):
    """
    Выполняет STL-декомпозицию временного ряда.
    
    Parameters:
    - series: np.array, одномерный временной ряд.
    - period: int, период сезонности (если None, автоматически определяется через ACF).
    
    Returns:
    - trend: Трендовая компонента.
    - seasonal: Сезонная компонента.
    - resid: Остаточная компонента.
    """
    if period == 1:
        acf_values = acf(series, nlags=40)
        peaks = argrelmax(acf_values)[0]
        if len(peaks) > 0:
            period = peaks[0]
        else:
            period = 10  # Значение по умолчанию
    
    stl = STL(series, period=period)
    result = stl.fit()
    
    return result.resid


def stl_features(series, period=None):
    """
    Возвращает числовые признаки на основе STL-декомпозиции.
    """
    trend, seasonal, resid = stl_decomposition(series, period)
    features = [
        np.var(trend),
        np.var(seasonal),
        np.var(resid),
        np.max(seasonal) - np.min(seasonal),  # Амплитуда сезонности
        np.sum(np.abs(np.diff(trend)))       # Гладкость тренда
    ]
    return np.array(features)