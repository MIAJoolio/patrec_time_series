import numpy as np

__all__ = [
    "linear_trend",
    "quadratic_trend",
    "exponential_trend",
    "sin_wave_norm",
    "sin_wave",   
    "sawtooth_wave",
    "harmonic_shift",
    'sawtooth_shift',
    "random_walk"
]


def linear_trend(slope: float = 0.5, length: int = 100) -> np.ndarray:
    return slope * np.arange(length)


def quadratic_trend(a: float = 1.0, b: float = 0.5, c: float = 0.0, length: int = 100) -> np.ndarray:
    x = np.arange(length)
    return a * x**2 + b * x + c


def exponential_trend(alpha: float = 0.1, length: int = 100) -> np.ndarray:
    x = np.arange(length)
    return np.exp(alpha * x)


def sin_wave_norm(amplitude: float = 1.0, frequency: float = 1.0, phase: float = 0.0, length: int = 100) -> np.ndarray:
    """
    normalized frequency over length
    """
    x = np.arange(length)
    return amplitude * np.sin(2 * np.pi * frequency * x / length + phase)

def sin_wave(amplitude: float = 1.0, frequency: float = 1.0, phase: float = 0.0, length: int = 100) -> np.ndarray:
    x = np.arange(1, length+1)
    return amplitude * np.sin(2 * np.pi * x / frequency + phase)

def sawtooth_wave(amplitude: float = 1.0, frequency: float = 1.0, length: int = 100) -> np.ndarray:
    t = np.arange(length)
    period = length / frequency
    return amplitude * (t % period) / period


def harmonic_shift(amplitude: float = 10, frequency: float = 1, damping: float = 0.05, t_peak: float = 20.0, length: int = 100 ) -> np.ndarray:
    """
    Гармонический осциллятор с пиком в заданной точке.
    
    :param amplitude: Амплитуда
    :param frequency: Частота колебаний
    :param damping: Коэффициент затухания
    :param t_peak: Точка, в которой будет находиться пик
    :param length: Длина временного ряда
    :return: Временной ряд с пиком в нужной точке
    """
    t = np.arange(length)
    
    # Центрируем колебание на t_peak
    wave = np.sin(2 * np.pi * frequency * (t - t_peak) / length)
    
    # Затухание начинается от t_peak
    envelope = np.exp(-damping * np.abs(t - t_peak))
    
    return amplitude * envelope * wave


def sawtooth_shift(amplitude: float = 1.0, frequency: float = 1.0, length: int = 101) -> np.ndarray:
    t = np.arange(length)
    period = length / frequency  
    first_period_mask = t < period  

    # Первый период — линейный рост, остальное — ноль
    signal = np.zeros_like(t, dtype=float)
    signal[first_period_mask] = amplitude * (t[first_period_mask] / period)

    return signal


def random_walk(initial_value: float = 0.0, noise_std: float = 1.0, length: int = 100) -> np.ndarray:
    series = np.zeros(length)
    series[0] = initial_value
    noise = np.random.normal(0, noise_std, length)
    for t in range(1, length):
        series[t] = series[t - 1] + noise[t]
    return series


def main():
    return None

if __name__ == '__main__':
    main()