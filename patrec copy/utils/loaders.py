from typing import Dict, List, Tuple, Optional, Union, Any, Literal
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from patrec.utils.files_helper import JSON_Handler


class StratifiedTSLoader:
    """
    Загрузчик временных рядов с стратифицированным разбиением и поддержкой масштабирования.
    Поддерживает MinMaxScaler и StandardScaler.
    """
    
    def __init__(self, filepath: str, 
                 val_size: float = 0.15, 
                 test_size: float = 0.15,
                 batch_size: int = 32,
                 random_state: int = 42,
                 scaler_type: Literal['minmax', 'standard', None] = None,
                 feature_range: tuple = (0, 1)):
        """
        Args:
            filepath: путь к JSON файлу с данными
            val_size: доля validation выборки (0-1)
            test_size: доля test выборки (0-1)
            batch_size: размер батча для итеративной загрузки
            random_state: seed для воспроизводимости
            scaler_type: тип масштабирования ('minmax', 'standard' или None)
            feature_range: диапазон для MinMaxScaler (только при scaler_type='minmax')
        """
        self.handler = JSON_Handler(filepath)
        self.data = self.handler.data['data']
        self.labels = np.array(self.handler.data['labels'])
        self.meta = self.handler.data.get('meta', [{}]*len(self.data))
        self.batch_size = batch_size
        self.random_state = random_state
        self.scaler_type = scaler_type
        self.feature_range = feature_range
        
        # Инициализация scaler
        self.scaler = None
        if self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=self.feature_range)
        elif self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        
        # Проверка входных данных
        if len(self.labels) != len(self.data):
            raise ValueError("Длина данных и меток должна совпадать")
        if len(self.meta) != len(self.data):
            self.meta = [{} for _ in range(len(self.data))]
        # Применение масштабирования
        if self.scaler is not None:
            self._fit_scaler()
            
        # Разбиение данных
        self._split_data(val_size, test_size)
    
    def _fit_scaler(self):
        """Обучение scaler на всех данных"""
        # Объединяем все временные ряды для обучения scaler
        all_data = np.concatenate([ts.reshape(-1, 1) for ts in self.data])
        self.scaler.fit(all_data)
        
        # Применяем масштабирование ко всем данным
        self.data = [self.scaler.transform(ts.reshape(-1, 1)).flatten() for ts in self.data]
    
    def _split_data(self, val_size: float, test_size: float):
        """Стратифицированное разбиение данных на train/val/test."""
        # Первое разбиение: train+val / test
        X_train_val, X_test, y_train_val, y_test, meta_train_val, meta_test = train_test_split(
            self.data, self.labels, self.meta,
            test_size=test_size,
            stratify=self.labels,
            random_state=self.random_state
        )
        
        # Второе разбиение: train / val
        val_size_adj = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
            X_train_val, y_train_val, meta_train_val,
            test_size=val_size_adj,
            stratify=y_train_val,
            random_state=self.random_state
        )
        
        self.splits = {
            'train': {'data': X_train, 'labels': y_train, 'meta': meta_train},
            'val': {'data': X_val, 'labels': y_val, 'meta': meta_val},
            'test': {'data': X_test, 'labels': y_test, 'meta': meta_test}
        }
        
    def get_batches(self, split: str = 'train') -> Tuple[List[np.ndarray], List[int]]:
        """
        Генератор батчей для указанной выборки.
        
        Args:
            split: выборка ('train', 'val' или 'test')
            
        Yields:
            tuple: (batch_data, batch_labels) где:
                batch_data: список временных рядов в батче
                batch_labels: соответствующие метки классов
        """
        if split not in self.splits:
            raise ValueError(f"Неизвестная выборка: {split}. Допустимые значения: 'train', 'val', 'test'")
            
        data = self.splits[split]['data']
        labels = self.splits[split]['labels']
        n_samples = len(data)
        
        indices = np.arange(n_samples)
        np.random.seed(self.random_state)
        np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_data = np.asarray([data[i] for i in batch_indices])
            batch_labels = np.asarray([labels[i] for i in batch_indices])
            
            yield batch_data, batch_labels
            
    def get_full_split(self, split: str) -> Dict[str, Union[List[np.ndarray], List[int], List[dict]]]:
        """
        Получить полную выборку (не батчами).
        
        Args:
            split: выборка ('train', 'val' или 'test')
            
        Returns:
            dict: {'data': [...], 'labels': [...], 'meta': [...]}
        """
        return self.splits[split]
    
    def get_class_distribution(self) -> Dict[str, Dict[Any, int]]:
        """
        Получить распределение классов по выборкам.
        
        Returns:
            dict: распределение классов в формате:
                {
                    'train': {class1: count, class2: count, ...},
                    'val': {...},
                    'test': {...}
                }
        """
        dist = {}
        for split in self.splits:
            labels = self.splits[split]['labels']
            unique, counts = np.unique(labels, return_counts=True)
            dist[split] = dict(zip(unique, counts))
        return dist
    

class ConcatenatedTSLoader:
    """
    Загрузчик временных рядов, объединяющий ряды внутри батчей в один общий ряд.
    Сохраняет метаданные о длинах исходных рядов и их метках.
    
    Формат входных данных (JSON):
    {
        "data": [[ts1], [ts2], ...],  # список временных рядов
        "labels": [label1, label2, ...],  # метки классов
        "meta": [  # метаданные (опционально)
            {"id": 1, "source": "UCR", ...},
            {"id": 2, "source": "UEA", ...},
            ...
        ]
    }
    """
    
    def __init__(self, filepath: str, 
                 batch_size: int = 32,
                 random_state: int = 42):
        """
        Args:
            filepath: путь к JSON файлу с данными
            batch_size: количество рядов для объединения в один батч
            random_state: seed для воспроизводимости
        """
        self.handler = JSON_Handler(filepath)
        self.data = [np.array(ts) for ts in self.handler.data['data']]
        self.labels = np.array(self.handler.data['labels'])
        self.meta = self.handler.data.get('meta', [{}]*len(self.data))
        self.batch_size = batch_size
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Проверка входных данных
        if len(self.data) != len(self.labels):
            raise ValueError("Длина данных и меток должна совпадать")
            
    def get_batches(self) -> Tuple[np.ndarray, Dict[str, List]]:
        """
        Генератор объединенных батчей.
        
        Yields:
            tuple: (concatenated_series, batch_meta) где:
                concatenated_series: объединенный временной ряд
                batch_meta: словарь с метаданными:
                    {
                        'lengths': длины исходных рядов,
                        'labels': метки классов,
                        'indices': индексы рядов в исходном наборе,
                        'meta': полные метаданные (если есть)
                    }
        """
        n_samples = len(self.data)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Объединение рядов в один
            concatenated = np.concatenate([self.data[i] for i in batch_indices])
            
            # Сбор метаданных
            batch_meta = {
                'lengths': [len(self.data[i]) for i in batch_indices],
                'labels': [self.labels[i] for i in batch_indices],
                'indices': batch_indices.tolist(),
                'meta': [self.meta[i] for i in batch_indices] if self.meta else None
            }
            
            yield concatenated, batch_meta
            
    def reconstruct_batch(self, concatenated: np.ndarray, lengths: List[int]) -> List[np.ndarray]:
        """
        Восстановить исходные ряды из объединенного батча.
        
        Args:
            concatenated: объединенный временной ряд
            lengths: список длин исходных рядов
            
        Returns:
            list: список восстановленных временных рядов
        """
        series = []
        start = 0
        for length in lengths:
            end = start + length
            series.append(concatenated[start:end])
            start = end
        return series