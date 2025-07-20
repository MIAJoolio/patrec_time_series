from sklearn.cluster import AgglomerativeClustering

from patrec.clustering.Base_clustering_model import Base_clustering_model

class AC_model(Base_clustering_model):
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