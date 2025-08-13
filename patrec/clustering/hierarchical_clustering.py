
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

from .base import Base_Clustering_Model

class AC_model(Base_Clustering_Model):
    """
    Реализация кластеризации с использованием алгоритма Agglomerative Clustering.
    """

    def __init__(self):
        super().__init__()
        self.default_params = {
            "n_clusters": 2,
            "metric": "euclidean",
            "linkage": "ward",
            "compute_full_tree": False
        }

    def fit_predict(self, X_train, X_test=None):
        """
        Обучает AgglomerativeClustering на обучающих данных и предсказывает кластеры.

        Parameters:
            X_train (np.ndarray): Данные для обучения.
            X_test (np.ndarray): Данные для предсказания (опционально).

        Returns:
            tuple: (метки кластеров, обученная модель)
        """
        # X_train = self.normalize(X_train)

        model_params = self.default_params.copy()
        self.model = AgglomerativeClustering(**model_params)

        if X_test is None:
            labels = self.model.fit_predict(X_train)
        else:
            # Agglomerative не поддерживает fit + predict отдельно → нужно обучать на X_train
            self.model.fit(X_train)
            labels = self.model.fit_predict(X_test)

        return labels, self.model

# чет можно еще для одного ряда побаловаться - https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-py 


class AC_hier_model(Base_Clustering_Model):
    """Agglomerative Clustering с поддержкой дендрограмм и измерения времени выполнения."""
    
    def __init__(self):
        super().__init__()
        self.default_params = {
            "n_clusters": 2,
            "metric": "euclidean",
            "linkage": "ward",
            "compute_full_tree": True,  # Для построения дендрограммы
            "compute_distances": True   # Для отображения расстояний
        }
        self._init_model()
        self.distance_threshold = None
    
    def _init_model(self):
        self.model = AgglomerativeClustering(**self.default_params)
    
    def fit_predict(self, 
                  X_train: np.ndarray, 
                  X_test: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Особенности реализации:
        - Сохраняет данные для построения дендрограммы
        - Для тестовых данных выполняется переобучение
        """
        self.X_train = X_train  # Сохраняем для построения дендрограммы
        
        if X_test is None:
            train_labels = self.model.fit_predict(X_train)
            return train_labels, None
        else:
            self.model.fit(X_train)
            train_labels = self.model.labels_
            self.model.fit(X_test)
            test_labels = self.model.labels_
            return train_labels, test_labels
    
    def plot_dendrogram(self, 
                       method: str = 'ward',
                       metric: str = 'euclidean',
                       color_threshold: Optional[float] = None,
                       truncate_mode: str = 'lastp',
                       p: int = 12,
                       show: bool = True,
                       save_path: Optional[str] = None,
                       figsize: tuple = (12, 6)) -> None:
        """
        Визуализация дендрограммы для иерархической кластеризации.
        
        Args:
            method: Метод связи ('ward', 'single', 'complete', 'average')
            metric: Метрика расстояния
            color_threshold: Порог для окрашивания ветвей
            truncate_mode: Режим усечения ('lastp', 'level', None)
            p: Количество листьев для отображения при усечении
            show: Показывать график
            save_path: Путь для сохранения (если None - не сохранять)
            figsize: Размер фигуры
        """
        if not hasattr(self, 'X_train'):
            raise ValueError("Сначала обучите модель методом fit_predict()")
            
        # Вычисляем матрицу связей
        Z = linkage(self.X_train, method=method, metric=metric)
        
        # Настройка порога окрашивания
        if color_threshold is None and hasattr(self.model, 'n_clusters'):
            color_threshold = self.distance_threshold or (Z[-self.model.n_clusters, 2] + Z[-self.model.n_clusters+1, 2])/2
        
        plt.figure(figsize=figsize)
        dendrogram(
            Z,
            truncate_mode=truncate_mode,
            p=p,
            color_threshold=color_threshold,
            show_leaf_counts=True,
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True
        )
        
        plt.title(f'Дендрограмма (метод связи: {method})')
        plt.xlabel('Индекс образца или размер кластера')
        plt.ylabel('Расстояние')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Дендрограмма сохранена в {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def set_distance_threshold(self, threshold: float) -> None:
        """Устанавливает порог расстояния для автоматического определения числа кластеров"""
        self.distance_threshold = threshold
        self.default_params['n_clusters'] = None
        self.default_params['distance_threshold'] = threshold
        self._init_model()