from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

import ruptures as rpt
import stumpy

from scipy.signal import find_peaks  

class Segmentor(ABC):
    def __init__(self, **kwargs):
        self.change_points = None
        self.signal = None
    
    @abstractmethod
    def infer(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        pass
    
    def make_segments(self, signal: np.ndarray = None) -> list[np.ndarray]:
        """Возвращает список сегментов как numpy массивы."""
        signal = self.signal if signal is None else signal
        if self.change_points is None:
            raise ValueError("Сначала вызовите infer() для поиска точек разладки.")
        
        segments = []
        start = 0
        for end in sorted(self.change_points):
            if end <= start:  # Защита от некорректных точек
                continue
            segments.append(signal[start:int(end)])
            start = int(end)
        segments.append(signal[start:])  # Последний сегмент
        return segments
    
    def plot(self, signal: np.ndarray = None, **kwargs):
        signal = self.signal if signal is None else signal
        plt.figure(figsize=(12, 6))
        plt.plot(signal, label="Временной ряд", color='blue')
        
        if self.change_points is not None:
            for cp in sorted(self.change_points):
                plt.axvline(cp, color='red', linestyle='--', alpha=0.7, label="Точка разладки" if cp == sorted(self.change_points)[0] else "")
        
        plt.title("Сегментация временного ряда")
        plt.legend()
        plt.show()
    
    def find_peaks(self, signal: np.ndarray = None, 
                  height: float = None, 
                  prominence: float = None,
                  distance: int = None) -> np.ndarray:
        """Находит пики (максимумы и минимумы) в сигнале с настройкой параметров."""
        signal = self.signal if signal is None else signal
        
        # Находим максимумы
        max_peaks, _ = find_peaks(signal, height=height, prominence=prominence, distance=distance)
        # Находим минимумы (инвертируем сигнал)
        min_peaks, _ = find_peaks(-signal, height=height, prominence=prominence, distance=distance)
        
        peaks = np.concatenate([max_peaks, min_peaks])
        peaks.sort()
        return peaks
        

class PeltSegmentor(Segmentor):
    def __init__(self, model="l2", min_size=10, jump=5):
        super().__init__()
        self.model = model
        self.min_size = min_size
        self.jump = jump
    
    def infer(self, signal: np.ndarray, pen: float = 10.0) -> np.ndarray:
        self.signal = signal
        algo = rpt.Pelt(model=self.model, min_size=self.min_size, jump=self.jump).fit(signal)
        self.change_points = np.array(algo.predict(pen=pen)[:-1])  # Исключаем последний элемент (конец ряда)
        return self.change_points
            
    
class BinSegSegmentor(Segmentor):
    def __init__(self, model="l2", n_bkps=5):
        super().__init__()
        self.model = model
        self.n_bkps = n_bkps
    
    def infer(self, signal: np.ndarray) -> np.ndarray:
        self.signal = signal
        algo = rpt.Binseg(model=self.model).fit(signal)
        self.change_points = np.array(algo.predict(n_bkps=self.n_bkps))[:-1]
        return self.change_points


class KernelCPDSegmentor(Segmentor):
    """Сегментация на основе Kernel Change Point Detection (KernelCPD) из ruptures."""
    
    def __init__(self, kernel="rbf", min_size=10, jump=5):
        """
        Args:
            kernel: Тип ядра ("rbf", "linear", "cosine").
            min_size: Минимальный размер сегмента.
            jump: Шаг для ускорения вычислений.
        """
        super().__init__()
        self.kernel = kernel
        self.min_size = min_size
        self.jump = jump
    
    def infer(self, signal: np.ndarray, pen: float = 10.0, detect_merge: bool = False) -> np.ndarray:
        """Добавлен параметр detect_merge для поиска точки объединения рядов."""
        self.signal = signal
        
        algo = rpt.KernelCPD(
            kernel=self.kernel,
            min_size=self.min_size,
            jump=self.jump
        ).fit(signal)
        
        self.change_points = np.array(algo.predict(pen=pen))[:-1]
        
        if detect_merge:
            if len(signal) > 100:  
                window_size = 30
                matrix_profile = stumpy.stump(signal, m=window_size)
                merge_point = np.argmax(matrix_profile[:, 0])
                self.change_points = np.append(self.change_points, merge_point)
                self.change_points.sort()
                
        return self.change_points
    
    
class BottomUpSegmentor(Segmentor):
    """Сегментация на основе BottomUp алгоритма из ruptures."""
    
    def __init__(self, model="l2", min_size=10, jump=5):
        """
        Args:
            model: Тип модели ("l2", "l1", "rbf").
            min_size: Минимальный размер сегмента.
            jump: Шаг для ускорения вычислений.
        """
        super().__init__()
        self.model = model
        self.min_size = min_size
        self.jump = jump
    
    def infer(self, signal: np.ndarray, n_bkps: int = 5) -> np.ndarray:
        """Поиск точек разладки с помощью BottomUp.
        
        Args:
            signal: Входной временной ряд.
            n_bkps: Количество точек разладки.
            
        Returns:
            Массив с индексами точек разладки.
        """
        self.signal = signal
        
        # Инициализация BottomUp
        algo = rpt.BottomUp(
            model=self.model,
            min_size=self.min_size,
            jump=self.jump
        ).fit(signal)
        
        # Предсказание точек разладки
        self.change_points = np.array(algo.predict(n_bkps=n_bkps))[:-1]  # Исключаем последний элемент
        return self.change_points


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
    

class StumpySegmentor(Segmentor):
    def __init__(self, window_size=50, threshold_std=2.0):
        super().__init__()
        self.window_size = window_size
        self.threshold_std = threshold_std
    
    def infer(self, signal: np.ndarray) -> np.ndarray:
        self.signal = signal
        
        # Вычисляем матричный профиль
        mp = stumpy.stump(signal, m=self.window_size)
        mp_scores = mp[:, 0].astype(float)  # Явное приведение к float
        
        # Находим аномальные пики (точки разладки)
        threshold = np.nanmean(mp_scores) + self.threshold_std * np.nanstd(mp_scores)
        self.change_points = np.where(mp_scores > threshold)[0]
        
        # Фильтрация слишком близких точек
        if len(self.change_points) > 0:
            min_dist = self.window_size
            filtered_cps = [self.change_points[0]]
            for cp in self.change_points[1:]:
                if cp - filtered_cps[-1] >= min_dist:
                    filtered_cps.append(cp)
            self.change_points = np.array(filtered_cps)
        
        return self.change_points
    
    
# from merlion.models.anomaly.change_point.bocpd import BOCPD, BOCPDConfig
# from merlion.post_process.threshold import Threshold

# class MerlionBOCPDSegmentor(Segmentor):
#     """Сегментация через Bayesian Online Change Point Detection (BOCPD)."""
    
#     def __init__(self, threshold: float = 3.0, model_type="lognormal"):
#         super().__init__()
#         self.threshold = threshold
#         self.model_type = model_type  # "lognormal", "normal", "poisson"
        
#     def infer(self, signal: np.ndarray) -> np.ndarray:
#         self.signal = signal
        
#         # Инициализация BOCPD
#         config = BOCPDConfig(
#             model_type=self.model_type,
#             threshold=Threshold(alm_threshold=self.threshold))
#         model = BOCPD(config)
        
#         # Обучение и предсказание (BOCPD работает онлайн, но принимает весь ряд)
#         train_scores = model.train(signal)
#         change_points = model.get_anomaly_label(signal)
        
#         # Индексы точек разладки (где change_points == 1)
#         self.change_points = np.where(change_points == 1)[0]
#         return self.change_points