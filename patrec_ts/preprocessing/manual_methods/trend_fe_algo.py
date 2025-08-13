from statsmodels.tsa.stattools import acf
import numpy as np


def diagnose_trend_with_acf(series, lags=20):
    """
    Анализирует временной ряд на наличие и тип тренда с использованием ACF.
    
    Параметры:
        series (np.array or list): Временной ряд.
        lags (int): Количество лагов для анализа ACF.

    Возвращает:
        dict: Результаты анализа — признаки наличия и типа тренда.
    """
    result = {}
    series = np.asarray(series)

    if len(series) < 2:
        return {"error": "Серия слишком короткая для анализа."}

    # Корректировка количества лагов
    max_valid_lag = min(lags, len(series) - 1)
    acf_values = acf(series, nlags=max_valid_lag, fft=False)

    persistence = np.mean(acf_values[1:]) if len(acf_values) > 1 else 0
    decay_rate = np.mean(np.diff(acf_values[:min(5, len(acf_values))])) if len(acf_values) >= 2 else 0

    result['has_trend'] = persistence > 0.5 and decay_rate > -0.2

    # Линейный тренд
    first_diff = np.diff(series)
    if len(first_diff) >= 1:
        acf_first_diff = acf(first_diff, nlags=min(max_valid_lag - 1, len(first_diff) - 1), fft=False)
        persistence_first_diff = np.mean(acf_first_diff[1:]) if len(acf_first_diff) > 1 else 0
        result['linear_trend_likely'] = persistence_first_diff < 0.25
    else:
        result['linear_trend_likely'] = False

    # Квадратичный тренд
    if result['linear_trend_likely'] is False and len(first_diff) >= 2:
        second_diff = np.diff(first_diff)
        if len(second_diff) >= 1:
            acf_second_diff = acf(second_diff, nlags=min(max_valid_lag - 2, len(second_diff) - 1), fft=False)
            persistence_second_diff = np.mean(acf_second_diff[1:]) if len(acf_second_diff) > 1 else 0
            result['quadratic_trend_likely'] = persistence_second_diff < 0.25
        else:
            result['quadratic_trend_likely'] = False
    else:
        result['quadratic_trend_likely'] = False

    return result


def extract_acf_statistics(series: np.ndarray, n_lags: int = 20) -> np.ndarray:
    """
    Извлекает основные статистики из ACF временного ряда.
    
    Возвращает:
        np.ndarray: [persistence, decay_rate, significant_lag_count, lag_of_max_abs_acf]
    """
    n = len(series)
    if n < 2 or n_lags >= n:
        return np.zeros(5)

    try:
        acf_values = acf(series, nlags=n_lags, fft=False)
    except Exception:
        return np.zeros(5)

    # Persistence: среднее значение ACF без нулевого лага
    persistence = np.mean(acf_values[1:]) if len(acf_values) > 1 else 0

    # Decay rate: средняя скорость снижения ACF на первых 5 лагах
    window = min(5, len(acf_values))
    decay_rate = np.mean(np.diff(acf_values[:window])) if window > 1 else 0

    # Число значимых лагов (по 95% доверительному интервалу)
    threshold = 1.96 / np.sqrt(n)
    significant_lag_count = np.sum(np.abs(acf_values[1:]) > threshold)

    # Лаг максимального |ACF| после 0
    abs_vals = np.abs(acf_values[1:])
    lag_of_max_abs_acf = np.argmax(abs_vals) + 1 if len(abs_vals) > 0 else 0

    zero_lag = -1
    # Находит первый лаг, где ACF меняет знак (пересекает ноль)
    for k in range(1, len(acf_values) - 1):
        if np.sign(acf_values[k]) != np.sign(acf_values[k + 1]):
            zero_lag = k + 1     
    
    return np.array([
        persistence,
        decay_rate,
        significant_lag_count,
        lag_of_max_abs_acf,
        zero_lag
    ])