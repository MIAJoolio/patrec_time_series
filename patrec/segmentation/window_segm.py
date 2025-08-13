from .segmentor import Segmentor

import numpy as np

import ruptures as rpt


class WindowSegmentor(Segmentor):
    """Сегментация на основе Window-метода из ruptures."""
    
    def __init__(self, model="l2", width=100, min_size=10, jump=5):
        """
        Args:
            model: Тип модели ("l2", "l1", "rbf").
            width: Ширина окна.
            min_size: Минимальный размер сегмента.
            jump: Шаг для ускорения вычислений.
        """
        super().__init__()
        self.model = model
        self.width = width
        self.min_size = min_size
        self.jump = jump
    
    def infer(self, signal: np.ndarray, pen: float = 10.0) -> np.ndarray:
        """Поиск точек разладки с помощью Window-метода.
        
        Args:
            signal: Входной временной ряд.
            pen: Параметр штрафа за добавление новой точки.
            
        Returns:
            Массив с индексами точек разладки.
        """
        self.signal = signal
        
        # Инициализация Window
        algo = rpt.Window(
            model=self.model,
            width=self.width,
            min_size=self.min_size,
            jump=self.jump
        ).fit(signal)
        
        # Предсказание точек разладки
        self.change_points = np.array(algo.predict(pen=pen))[:-1]  # Исключаем последний элемент
        return self.change_points
    
# class + fixed через strive=0, 
def subseq_generator(data, window_size, stride=1):
    """
    Generate subsequences as a generator (memory efficient for large data).
    
    Args:
        data: list or 1D array of time-series data
        window_size: length of each subsequence
        stride: step size between windows
    
    Yields:
        One subsequence at a time
    """
    for i in range(0, len(data)-window_size+1, stride):
        yield data[i:i+window_size]
        