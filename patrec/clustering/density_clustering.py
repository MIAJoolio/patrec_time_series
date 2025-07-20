from sklearn.cluster import DBSCAN, Birch

from patrec.clustering.Base_clustering_model import Base_clustering_model

from sklearn.cluster import DBSCAN
from .Base_clustering_model import Base_clustering_model


class DBSCAN_model(Base_clustering_model):
    def __init__(self):
        super().__init__()
        self.default_params = {
            "eps": 0.5,
            "min_samples": 5,
            "metric": "euclidean"
        }

    def fit_predict(self, X_train, X_test=None):
        # X_train = self.normalize(X_train)
        self.model = DBSCAN(**self.default_params)
        labels = self.model.fit_predict(X_train)
        return labels, self.model
    
class BIRCH_model(Base_clustering_model):
    """
    Реализация кластеризации с использованием алгоритма BIRCH.
    """

    def __init__(self):
        super().__init__()
        self.default_params = {
            "threshold": 0.5,
            "branching_factor": 50,
            "n_clusters": None, 
            "compute_labels": True,
            "copy": False
        }

    def fit_predict(self, X_train, X_test=None):
        """
        Обучает BIRCH на обучающих данных и предсказывает кластеры.

        Parameters:
            X_train (np.ndarray): Данные для обучения.
            X_test (np.ndarray): Данные для предсказания (опционально).

        Returns:
            tuple: (метки кластеров, обученная модель)
        """
        # X_train = self.normalize(X_train)

        model_params = self.default_params

        self.model = Birch(**model_params)

        # Обучение
        self.model.fit(X_train)
        self.is_fitted = True

        # Предсказание
        if X_test is None:
            labels = self.model.labels_
        else:
            labels = self.model.predict(X_test)

        return labels, self.model