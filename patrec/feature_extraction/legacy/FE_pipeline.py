import os
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from itertools import product

import numpy as np
import pandas as pd

from patrec.utils import load_config_file

class Feature_extractor(ABC):
    def __init__(self, config_path: str, output_dir: str = "experiments/fe_output"):
        """
        Абстрактный базовый класс для извлечения признаков временного ряда.

        Параметры:
            config_path (str): Путь к YAML-конфигурации
            output_dir (str): Каталог для сохранения результатов
        """
        # self.config = load_config_file(config_path)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def load_data(self, file_path: str):
        """Загружает и предобрабатывает данные."""
        pass

    @abstractmethod
    def extract_features(self, data, algo_params) -> np.ndarray:
        """Извлекает признаки из временного ряда."""
        pass

    @abstractmethod
    def generate_param_grid(self) -> List[Dict]:
        """Генерирует сетку параметров для grid search."""
        pass

