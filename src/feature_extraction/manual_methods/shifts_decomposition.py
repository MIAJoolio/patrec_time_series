from pathlib import Path
from typing import Literal
import time

import ruptures as rpt
        
from hmmlearn import hmm
import numpy as np

from src.feature_extraction.fe_classes import Base_extractor, FE_result


class StructuralBreaksDetector(Base_extractor):
    def extract(self, data: np.ndarray, model="l2", penalty=10, min_size=3) -> FE_result:
        start_time = time.time()
        
        # Алгоритм PELT (быстрый и точный)
        algo = rpt.Pelt(model=model, min_size=min_size).fit(data)
        breaks = algo.predict(pen=penalty)
        
        # Визуализация (опционально)
        # rpt.display(data, breaks)
        
        return FE_result(
            component=np.array(breaks),  # Точки разрыва
            method_name='structural_breaks_pelt',
            method_params={
                'model': model,
                'penalty': penalty,
                'min_size': min_size
            },
            execution_stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': data.shape
            },
            results={
                'break_points': breaks,
                'num_breaks': len(breaks) - 1
            }
        )
        
        
class CUSUMDetector(Base_extractor):
    def extract(self, data: np.ndarray, threshold=3, drift=0.01) -> FE_result:
        start_time = time.time()
        residuals = data - np.mean(data)
        cumsum = np.cumsum(residuals - drift)
        breaks = np.where(np.abs(cumsum) > threshold * np.std(residuals))[0]
        
        return FE_result(
            component=np.array(breaks),
            method_name='structural_breaks_cusum',
            method_params={'threshold': threshold, 'drift': drift},
            execution_stats={'execution_time_sec': time.time() - start_time},
            results={'break_points': breaks}
        )


class HMMBreakDetector(Base_extractor):
    def extract(self, data: np.ndarray, n_states=2) -> FE_result:
        start_time = time.time()
        model = hmm.GaussianHMM(n_components=n_states).fit(data.reshape(-1, 1))
        states = model.predict(data.reshape(-1, 1))
        breaks = np.where(np.diff(states) != 0)[0] + 1
        
        return FE_result(
            component=np.array(breaks),
            method_name='structural_breaks_hmm',
            method_params={'n_states': n_states},
            execution_stats={'execution_time_sec': time.time() - start_time},
            results={'break_points': breaks, 'states': states}
        )