from typing import Dict, Optional, Union, Callable

import numpy as np

__all__ = [
    "normal_noise",
    "poisson_noise",
    "uniform_noise",
    "exponential_noise"
]


def normal_noise(data: np.ndarray, noise_pct: float) -> np.ndarray:
    """
    Генерация нормального шума.
    """
    return np.random.normal(0, np.std(data) * noise_pct, data.shape)


def poisson_noise(data: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Генерация шума Пуассона.
    """
    return np.random.poisson(lambda_, data.shape)


def uniform_noise(data: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Генерация равномерного шума.
    """
    return np.random.uniform(low, high, data.shape)


def exponential_noise(data: np.ndarray, scale: float) -> np.ndarray:
    """
    Генерация экспоненциального шума.
    """
    return np.random.exponential(scale, data.shape)
    

def main():
    return None

if __name__ == '__main__':
    main()