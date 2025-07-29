from .base import Base_Clustering_Model
from sklearn.cluster import KMeans

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