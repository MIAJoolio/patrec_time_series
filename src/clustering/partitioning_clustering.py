from sklearn.cluster import KMeans
from tslearn import clustering
from sktime.clustering import k_medoids

from src.clustering.Base_clustering_model import Base_clustering_model


class Sklearn_kmeans_model(Base_clustering_model):
    def __init__(self):
        super().__init__()
        self.default_params = {
            "n_clusters": 8,
            "init": "k-means++",
            "n_init": 10,
            "max_iter": 300,
            "random_state": None
        }

    def fit_predict(self, X_train, X_test=None):
        # X_train = self.normalize(X_train)
        
        model_params = self.default_params.copy()
        self.model = KMeans(**model_params)

        if X_test is not None:
            self.model.fit(X_train)
            labels = self.model.predict(X_test)
        else:
            labels = self.model.fit_predict(X_train)

        return labels, self.model