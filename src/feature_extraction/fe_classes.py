from typing import Dict, List, Any, Optional
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import statsmodels.api as sm
from scipy import stats

@dataclass
class FE_result:
    """
    Контейнер для хранения результатов выделения признаков.
    """
    component: np.ndarray  # Выделенный компонент (тренд, сезонность и т.д.)
    method_name: str  # Название метода
    method_params: Dict[str, Any]  # Параметры метода
    execution_stats: Dict[str, Any]  # Статистика выполнения (время и др.)
    results: Dict[str, Any]  # Дополнительные результаты работы метода

class Base_extractor(ABC):
    """
    Базовый класс для экстракторов признаков.
    """
    @abstractmethod
    def extract(self, data: np.ndarray, **kwargs) -> FE_result:
        """
        Основной метод для выделения признаков.
        
        Args:
            data: Входные данные временного ряда
            **kwargs: Дополнительные параметры метода
            
        Returns:
            FE_result: Результат выделения признаков
        """
        pass

class FE_pipeline:
    """
    Пайплайн для выделения признаков из временных рядов.
    """
    def __init__(self, steps: List[Base_extractor]):
        """
        Инициализация пайплайна.
        
        Args:
            steps: Список экстракторов признаков в порядке их применения
        """
        self.steps = steps
        self.results = []
        
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Применяет пайплайн к данным.
        
        Args:
            data: Входные данные временного ряда
            
        Returns:
            Преобразованные данные после всех этапов
        """
        self.results = []  # Сбрасываем предыдущие результаты
        current_data = data.copy()
        
        for step in self.steps:
            result = step.extract(current_data)
            self.results.append(result)
            current_data -= result.component  # Используем компонент для следующего шага
            
        return current_data
    
    def get_results(self) -> List[FE_result]:
        """
        Возвращает детальные результаты работы пайплайна.
        
        Returns:
            Список результатов для каждого шага
        """
        return self.results

# # Пример использования
# if __name__ == "__main__":
#     # Создаем тестовые данные
#     np.random.seed(42)
#     t = np.arange(100)
#     data = 0.1 * t + 5 * np.sin(2 * np.pi * t / 12) + np.random.normal(size=100)
    
#     print("=== Пример с одним методом (линейный детрендинг) ===")
#     pipeline_single = FE_pipeline([LinregressDetrender()])
#     processed_data = pipeline_single.apply(data)
#     results = pipeline_single.get_results()
    
#     print(f"Метод: {results[0].method_name}")
#     print(f"Время выполнения: {results[0].execution_stats['execution_time_sec']:.4f} сек")
#     print(f"Наклон: {results[0].results['slope']:.4f}")
#     print(f"Перехват: {results[0].results['intercept']:.4f}")
#     print(f"R-квадрат: {results[0].results['r_squared']:.4f}")
    
#     print("\n=== Пример с двумя методами (детрендинг + STL) ===")
#     pipeline_double = FE_pipeline([
#         LinregressDetrender(),
#         STLDetrender(period=12)
#     ])
#     processed_data = pipeline_double.apply(data)
#     results = pipeline_double.get_results()
    
#     for i, result in enumerate(results):
#         print(f"\nШаг {i+1}: {result.method_name}")
#         print(f"Время выполнения: {result.execution_stats['execution_time_sec']:.4f} сек")
#         print(f"Параметры метода: {result.method_params}")
#         if i == 0:
#             print(f"Наклон: {result.results['slope']:.4f}")
#         elif i == 1:
#             print(f"Сезонная амплитуда: {np.max(result.results['seasonal']) - np.min(result.results['seasonal']):.4f}")