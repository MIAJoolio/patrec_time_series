from .segmentor import Segmentor

import numpy as np

import stumpy


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
    