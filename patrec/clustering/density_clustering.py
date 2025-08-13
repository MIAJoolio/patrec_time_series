from typing import Optional, Tuple
import numpy as np

from sklearn.cluster import DBSCAN, Birch

from patrec.clustering.base import Base_Clustering_Model


class DBSCAN_model(Base_Clustering_Model):
    """DBSCAN с расширенной функциональностью базового класса."""
    
    def __init__(self):
        super().__init__()
        self.default_params = {
            "eps": 0.5,
            "min_samples": 5,
            "metric": "euclidean"
        }
        self._init_model()
    
    def _init_model(self):
        self.model = DBSCAN(**self.default_params)
    
    def fit_predict(self, 
                  X_train: np.ndarray, 
                  X_test: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        train_labels = self.model.fit_predict(X_train)
        
        if X_test is not None:
            test_labels = self.model.fit_predict(X_test)
            raise NotImplementedError("DBSCAN не поддерживает предсказание для новых данных без переобучения")
        
        return train_labels, test_labels


class BIRCH_model(Base_Clustering_Model):
    """BIRCH алгоритм с полной интеграцией в базовый интерфейс."""
    
    def __init__(self):
        super().__init__()
        self.default_params = {
            "threshold": 0.5,
            "branching_factor": 50,
            "n_clusters": None,
            "compute_labels": True,
            "copy": False
        }
        self._init_model()
    
    def _init_model(self):
        self.model = Birch(**self.default_params)
    
    def fit_predict(self, 
                  X_train: np.ndarray, 
                  X_test: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self.model.fit(X_train)
        train_labels = self.model.labels_
        
        test_labels = None
        if X_test is not None:
            test_labels = self.model.predict(X_test)
        
        return train_labels, test_labels
    
    def get_cluster_centers(self) -> np.ndarray:
        if not hasattr(self.model, 'subcluster_centers_'):
            raise ValueError("Model not fitted yet. Call fit_predict() first.")
        return self.model.subcluster_centers_
    
    def get_inertia(self) -> float:
        raise NotImplementedError("BIRCH не предоставляет инерцию")