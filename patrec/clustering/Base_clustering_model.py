from abc import ABC, abstractmethod
import numpy as np

from sklearn.preprocessing import StandardScaler

# Импорты из ранее созданного ClusteringScorer
from patrec.clustering.Clustering_scorer import Clustering_scorer

import warnings
from sklearn.exceptions import ConvergenceWarning

# Отключение только ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler


class Base_clustering_model(ABC):
    def __init__(self):
        self.model = None
        self.default_params = {}
        self.scaler = StandardScaler()
        self.config = {}  # <-- добавили config

    def normalize(self, X):
        return self.scaler.fit_transform(X)

    def load_model_parameters(self, **params):
        """Применяет новые параметры к модели и возвращает их."""
        new_params = self.default_params.copy()
        new_params.update(params)
        self.default_params = new_params
        return new_params

    @abstractmethod
    def fit_predict(self, X_train, X_test=None):
        pass