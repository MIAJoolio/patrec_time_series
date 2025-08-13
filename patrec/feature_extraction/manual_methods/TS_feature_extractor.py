from typing import Callable, Union, Dict, Optional, List
import inspect
from tqdm import tqdm
from itertools import product

import numpy as np
from sklearn.model_selection import ParameterGrid

from patrec.utils.files_helper import load_config_file, save_config_file


class Feature_extraction_method:
    def __init__(self, method_func, config: dict):
        self.method_func = method_func

    def infer(self, series):
        """
        Применяет метод к полному временному ряду.
        Если params не задан, применяет все комбинации из self.param_grid.
        """
        
        results = None
        for param_set in ParameterGrid(self.param_grid):
            try:
                features = self.method_func(series, **param_set)
                results = (param_set, features)
            except Exception as e:
                print(f"Ошибка при использовании {param_set}: {e}")
        return results

    def infer_segmented(self, series, n_segments=3, params=None):
        """
            !!! лучше сразу данные делить на сегменты 
        """
        if n_segments <= 0 or not isinstance(n_segments, int):
            raise ValueError("n_segments должно быть целым положительным числом")

        length = len(series)
        segment_size = length // n_segments
        remainder = length % n_segments

        all_features = []

        for i in range(n_segments):
            start = i * segment_size
            end = start + segment_size + (1 if i < remainder else 0)
            segment = series[start:end]

            # Применяем метод к сегменту
            if params is not None:
                features = self.method_func(segment, **params)
            else:
                if self.param_grid is None:
                    raise ValueError("Не загружена сетка параметров.")
                # Берём первый набор параметров из grid (можно усреднить/выбрать лучший)
                features = self.method_func(segment, **self.param_grid[0])

            all_features.append(features)

        # Объединяем все признаки в один вектор
        return np.concatenate(all_features)


class Feature_extractor_pipeline:
    def __init__(self, methods):
        """
        Args:
            methods: List of Feature_extraction_method objects
        """
        self.methods = methods

#     def extract_single_series(self, series):
#         """
#         Извлекает все возможные признаки из одного ряда по всем методам и параметрам.

#         Returns:
#             features_by_config: list of dicts
#                 [
#                     {
#                         "features": np.array,
#                         "params": {"method1": params_used, ...}
#                     },
#                     ...
#                 ]
#         """
#         features_by_config = []

#         # Получаем все возможные комбинации параметров
#         method_param_options = []
#         for method in self.methods:
#             if method.param_grid is None:
#                 method_param_options.append([{}])  # без параметров
#             else:
#                 method_param_options.append(ParameterGrid(method.param_grid))

#         # Проходим по всем комбинациям параметров
#         for param_combination in product(*method_param_options):
#             feature_vector = []
#             param_log = {}

#             for method_idx, method in enumerate(self.methods):
#                 current_params = param_combination[method_idx]
#                 try:
#                     # Применяем метод с текущими параметрами
#                     features = method.method_func(series, **current_params)
#                     feature_vector.append(features)
#                     param_log[method.method_func.__name__] = current_params
#                 except Exception as e:
#                     feature_vector.append(np.zeros(1))  # Заглушка при ошибке
#                     param_log[method.method_func.__name__] = {'error': str(e)}

#             # Сохраняем результат
#             features_by_config.append({
#                 'features': np.concatenate(feature_vector),
#                 'params': param_log
#             })

#         return features_by_config

#     def extract_single_series_segmented(self, series, n_segments=3):
#         """
#         Извлекает признаки из временного ряда, разбитого на сегменты.
#         Для каждого метода и параметра: применяет infer_segmented, объединяет признаки.

#         Returns:
#             features_by_config: list of dicts
#         """
#         features_by_config = []

#         # Получаем все возможные комбинации параметров
#         method_param_options = []
#         for method in self.methods:
#             if method.param_grid is None:
#                 method_param_options.append([{}])
#             else:
#                 method_param_options.append(ParameterGrid(method.param_grid))

#         # Проходим по всем комбинациям параметров
#         for param_combination in product(*method_param_options):
#             feature_vector = []
#             param_log = {}

#             for method_idx, method in enumerate(self.methods):
#                 current_params = param_combination[method_idx]
#                 try:
#                     # Применяем метод с сегментацией
#                     features = method.infer_segmented(series, n_segments=n_segments, params=current_params)
#                     feature_vector.append(features)
#                     param_log[method.method_func.__name__] = current_params
#                 except Exception as e:
#                     feature_vector.append(np.zeros(1))
#                     param_log[method.method_func.__name__] = {'error': str(e)}

#             # Сохраняем результат
#             features_by_config.append({
#                 'features': np.concatenate(feature_vector),
#                 'params': param_log
#             })

#         return features_by_config

#     def batch_extract(self, dataset, use_segmentation=False, n_segments=3, verbose=False):
#         """
#         Обрабатывает весь датасет и возвращает список tuple:
#             [
#                 (X: np.ndarray, params: dict),
#                 ...
#             ]

#         Где:
#             X.shape == (n_samples, n_features)
#             params — словарь {method_name: params_used}
#         """
#         n_samples = len(dataset)

#         # Первый ряд обрабатываем для определения числа конфигураций
#         if use_segmentation:
#             sample_results = self.extract_single_series_segmented(dataset[0], n_segments=n_segments)
#         else:
#             sample_results = self.extract_single_series(dataset[0])

#         n_configs = len(sample_results)

#         if verbose:
#             print(f"Обнаружено {n_configs} уникальных комбинаций параметров")

#         # Подготавливаем структуры под все комбинации
#         all_X = [[] for _ in range(n_configs)]
#         all_params = [res['params'] for res in sample_results]

#         # Обработка всего датасета
#         for i, series in enumerate(dataset):
#             if verbose:
#                 print(f"Ряд {i + 1}/{n_samples}")
#             if use_segmentation:
#                 results = self.extract_single_series_segmented(series, n_segments=n_segments)
#             else:
#                 results = self.extract_single_series(series)

#             for config_idx, result in enumerate(results):
#                 all_X[config_idx].append(result['features'])

#         # Преобразуем в массивы
#         all_X_np = [np.array(X_list, dtype=np.float32) for X_list in all_X]

#         # Возвращаем список tuple: (X_array, params)
#         return [(all_X_np[i], all_params[i]) for i in range(n_configs)]