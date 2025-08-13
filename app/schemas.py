from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import numpy as np

class TimeSeriesData(BaseModel):
    """Структура для хранения одного временного ряда"""
    name: str
    values: List[float]
    labels: Optional[List[int]] = None  # Метки классов, если есть
    features: Optional[Dict[str, List[float]]] = None  # Извлеченные признаки

class DatasetInfo(BaseModel):
    """Метаинформация о датасете"""
    name: str
    source: str  # "UCR" или "UEA"
    n_samples: int
    n_features: int
    n_classes: Optional[int] = None
    length: int  # Длина временного ряда

class JSON_Handler(BaseModel):
    """Основной контейнер для данных"""
    datasets: Dict[str, TimeSeriesData]  # Ключ - имя датасета
    metadata: Dict[str, DatasetInfo]  # Метаданные для каждого датасета