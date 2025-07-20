from .segmentor import Segmentor

import numpy as np

import ruptures as rpt


class PeltSegmentor(Segmentor):
    def __init__(self, model="l2", min_size=10, jump=5):
        super().__init__()
        self.model = model
        self.min_size = min_size
        self.jump = jump
    
    def infer(self, signal: np.ndarray, pen: float = 10.0) -> np.ndarray:
        self.signal = signal
        algo = rpt.Pelt(model=self.model, min_size=self.min_size, jump=self.jump).fit(signal)
        self.change_points = np.array(algo.predict(pen=pen)[:-1])
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
    
    def infer(self, signal: np.ndarray, pen: float = 10.0) -> np.ndarray:
        """Поиск точек разладки с помощью KernelCPD.
        
        Args:
            signal: Входной временной ряд.
            pen: Параметр штрафа за добавление новой точки.
            
        Returns:
            Массив с индексами точек разладки.
        """
        self.signal = signal
        
        # Инициализация KernelCPD
        algo = rpt.KernelCPD(
            kernel=self.kernel,
            min_size=self.min_size,
            jump=self.jump
        ).fit(signal)
        
        # Предсказание точек разладки
        self.change_points = np.array(algo.predict(pen=pen))[:-1]  # Исключаем последний элемент
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