from .base import Base_Clustering_Model
from sklearn.cluster import KMeans    
import logging
import numpy as np
from tslearn.clustering import TimeSeriesKMeans, KShape
from aeon.clustering import TimeSeriesCLARA
from sklearn.base import BaseEstimator, ClusterMixin
from typing import Optional, Tuple, Union

class Sklearn_KMeans(Base_Clustering_Model):
    def __init__(self):
        super().__init__()
        self.default_params = {
            'n_clusters': 8,
            'init': 'k-means++',
            'n_init': 10,
            'max_iter': 300,
            'tol': 1e-4,
            'random_state': None
        }
        self.model = KMeans(**self.default_params)
    
    def load_model_parameters(self, **params):
        """Обновляет параметры модели и пересоздает экземпляр KMeans с новыми параметрами."""
        new_params = self.default_params.copy()
        new_params.update(params)
        self.default_params = new_params
        
        self.model = KMeans(**self.default_params)
        return new_params
        
    def fit_predict(self, X_train, X_test=None):
        """
        Обучает модель KMeans и предсказывает метки кластеров.
        
        Args:
            X_train (np.ndarray): Обучающие данные формы (n_samples, n_features)
            X_test (np.ndarray, optional): Тестовые данные. Если None, используется X_train.
            
        Returns:
            tuple: (метки для обучающих данных, метки для тестовых данных или None)
        """
        if X_test is None:
            X_test = X_train
            
        self.model.fit(X_train)
        
        train_labels = self.model.predict(X_train)
        test_labels = self.model.predict(X_test) if X_test is not None else None
        
        return train_labels, test_labels
    
    def get_cluster_centers(self):
        """
        Возвращает центры кластеров после обучения модели.
        
        Returns:
            np.ndarray: Массив центров кластеров формы (n_clusters, n_features)
        """
        if not hasattr(self.model, 'cluster_centers_'):
            raise ValueError("Модель не обучена. Сначала вызовите fit_predict().")
        return self.model.cluster_centers_
    
    def get_inertia(self):
        """
        Возвращает сумму квадратов расстояний образцов до ближайшего центра кластера.
        
        Returns:
            float: Значение инерции
        """
        if not hasattr(self.model, 'inertia_'):
            raise ValueError("Модель не обучена. Сначала вызовите fit_predict().")
        return self.model.inertia_

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesKMeansCustom(BaseEstimator, ClusterMixin):
    """Кастомный KMeans для временных рядов с поддержкой DTW и Euclidean метрик."""
    
    def __init__(self, 
                 n_clusters: int = 8,
                 metric: str = 'dtw',
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 n_init: int = 10,
                 random_state: Optional[int] = None,
                 verbose: bool = False):
        """
        Args:
            n_clusters: Количество кластеров
            metric: 'dtw' или 'euclidean'
            max_iter: Максимальное число итераций
            tol: Допуск сходимости
            n_init: Число инициализаций
            random_state: Seed для генератора случайных чисел
            verbose: Логирование процесса
        """
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'TimeSeriesKMeansCustom':
        """Обучение модели."""
        logger.info(f"Fitting TimeSeriesKMeans with {self.n_clusters} clusters using {self.metric} metric")
        self.model = TimeSeriesKMeans(
            n_clusters=self.n_clusters,
            metric=self.metric,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=self.n_init,
            random_state=self.random_state,
            verbose=self.verbose
        )
        self.model.fit(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание меток кластеров."""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.predict(X)
    
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Обучение и предсказание в одном методе."""
        self.fit(X)
        return self.predict(X)
    
    def get_cluster_centers(self) -> np.ndarray:
        """Получение центроидов кластеров."""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.cluster_centers_
    
    def get_inertia(self) -> float:
        """Получение значения инерции."""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.inertia_


class KShapeWrapper(BaseEstimator, ClusterMixin):
    """Обертка для KShape алгоритма из tslearn."""
    
    def __init__(self,
                 n_clusters: int = 8,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 n_init: int = 10,
                 random_state: Optional[int] = None,
                 verbose: bool = False):
        """
        Args:
            n_clusters: Количество кластеров
            max_iter: Максимальное число итераций
            tol: Допуск сходимости
            n_init: Число инициализаций
            random_state: Seed для генератора случайных чисел
            verbose: Логирование процесса
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'KShapeWrapper':
        """Обучение модели."""
        logger.info(f"Fitting KShape with {self.n_clusters} clusters")
        self.model = KShape(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=self.n_init,
            random_state=self.random_state,
            verbose=self.verbose
        )
        self.model.fit(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание меток кластеров."""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.predict(X)
    
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Обучение и предсказание в одном методе."""
        self.fit(X)
        return self.predict(X)
    
    def get_cluster_centers(self) -> np.ndarray:
        """Получение центроидов кластеров."""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.cluster_centers_
    
    def get_inertia(self) -> float:
        """Получение значения инерции."""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.inertia_


class TimeSeriesCLARAWrapper(BaseEstimator, ClusterMixin):
    """Обертка для TimeSeriesCLARA алгоритма из aeon."""
    
    def __init__(self,
                 n_clusters: int = 8,
                 init_algorithm: str = "random",
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 n_sampling: int = 40,
                 n_samples: int = 5,
                 random_state: Optional[int] = None,
                 verbose: bool = False):
        """
        Args:
            n_clusters: Количество кластеров
            init_algorithm: Алгоритм инициализации ('random' или 'k-means++')
            max_iter: Максимальное число итераций
            tol: Допуск сходимости
            n_sampling: Количество выборок для CLARA
            n_samples: Количество образцов в каждой выборке
            random_state: Seed для генератора случайных чисел
            verbose: Логирование процесса
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_sampling = n_sampling
        self.n_samples = n_samples
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'TimeSeriesCLARAWrapper':
        """Обучение модели."""
        logger.info(f"Fitting TimeSeriesCLARA with {self.n_clusters} clusters")
        self.model = TimeSeriesCLARA(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            n_sampling_iters=self.n_sampling,
            n_samples=self.n_samples,
            random_state=self.random_state,
            verbose=self.verbose
        )
        self.model.fit(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание меток кластеров."""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.predict(X)
    
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Обучение и предсказание в одном методе."""
        self.fit(X)
        return self.predict(X)
    
    def get_cluster_centers(self) -> np.ndarray:
        """Получение центроидов кластеров."""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.cluster_centers_
    
    def get_inertia(self) -> float:
        """Получение значения инерции (не поддерживается в CLARA)."""
        raise NotImplementedError("CLARA does not provide inertia value")