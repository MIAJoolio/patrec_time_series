from abc import ABC, abstractmethod
import numpy as np

import ruptures  as rpt

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
            if end <= start: 
                continue
            segments.append(signal[start:int(end)])
            start = int(end)
        segments.append(signal[start:])  
        return segments
    
    def plot(self, signal: np.ndarray = None, true_pts: np.ndarray = None,pred_pts: np.ndarray = None, **kwargs):
        signal = self.signal if signal is None else signal
        
        if pred_pts is None:
           pred_pts = self.change_points 
                   
        return rpt.display(signal, true_pts, pred_pts)
    
    def target_paddding(self, true_cps:np.ndarray=None, pred_cps:np.ndarray=None, tolerance=5):
        if true_cps is None:
            return None

        if pred_cps is None:
            pred_cps = self.change_points

        matched = set()
        y_true = []
        y_pred = []

        # Сортируем оба списка
        true_cps = sorted(true_cps)
        pred_cps = sorted(pred_cps)

        j = 0
        for cp in true_cps:
            # Ищем ближайшую предсказанную точку в пределах tolerance
            while j < len(pred_cps) and pred_cps[j] < cp - tolerance:
                j += 1
            if j < len(pred_cps) and abs(pred_cps[j] - cp) <= tolerance:
                y_true.append(1)
                y_pred.append(1)
                matched.add(j)
                j += 1
            else:
                y_true.append(1)
                y_pred.append(0)

        # Добавляем ложные срабатывания (предсказанные, но не найденные в ground truth)
        for j in range(len(pred_cps)):
            if j not in matched and pred_cps[j] not in true_cps:
                y_true.append(0)
                y_pred.append(1)

        return y_true, y_pred