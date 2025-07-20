from abc import ABC, abstractmethod
import numpy as np

from sklearn.preprocessing import StandardScaler

# Импорты из ранее созданного ClusteringScorer
from patrec.clustering.Clustering_scorer import Clustering_scorer

import warnings
from sklearn.exceptions import ConvergenceWarning

# Отключение только ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class Base_clustering_model(ABC):
    """
    Абстрактный базовый класс для моделей кластеризации.
    """

    def __init__(self):
        self.model = None
        self.is_fitted = False
        self.default_params = {}
        self.scaler = StandardScaler()
        
    def normalize(self, X_raw):
        return np.array([self.scaler.fit_transform(ts.reshape(-1, 1)).flatten() for ts in X_raw])

    def load_data(self, X_train, X_test=None):
        """
        Загружает обучающие и тестовые данные.

        Parameters:
            X_train (np.ndarray): Обучающая выборка.
            X_test (np.ndarray): Тестовая выборка (опционально).

        Returns:
            dict: Словарь с ключами 'X_train' и при наличии — 'X_test'.
        """
        data = {"X_train": X_train}
        if X_test is not None:
            data["X_test"] = X_test
        return data

    def load_model_parameters(self, **params):
        """
        Загружает или обновляет параметры модели.

        Parameters:
            params (dict): Произвольные параметры модели.

        Returns:
            dict: Объединённые параметры (дефолтные + новые).
        """
        self.default_params.update(params)
        return self.default_params

    @abstractmethod
    def fit_predict(self, X_train, X_test=None):
        """
        Обучает модель на X_train и предсказывает кластеры.
        Если есть X_test, применяет модель к тестовым данным.

        Parameters:
            X_train (np.ndarray): Данные для обучения.
            X_test (np.ndarray): Данные для предсказания (опционально).

        Returns:
            tuple: (метки кластеров, модель)
        """
        pass

    def evaluate_results(self, X, y_true=None, config=None, search_method="grid_search"):
        """
        Выполняет подбор гиперпараметров и оценку по всем метрикам:
        - silhouette_score
        - nmi
        - ari

        Parameters:
            X (np.ndarray): Входные данные.
            y_true (np.ndarray): Истинные метки (опционально).
            config (dict): сеткa параметров.
            search_method (str): Метод поиска — 'grid_search' или 'bayesian_search'.
            n_iter (int): Количество итераций для bayesian_search.

        Returns:
            pd.DataFrame: Результаты оценки по всем метрикам и комбинациям параметров.
        """
        if config is None:
            raise ValueError("Необходимо указать путь к конфигурационному файлу (config_path).")

        # Выбор метода поиска
        if search_method == "grid_search":
            df = Clustering_scorer.grid_search(model=self, X=X, y_true=y_true, config=config)
        else:
            raise ValueError(f"Метод '{search_method}' не поддерживается.")

        return df