import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd

from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score, accuracy_score, precision_score, recall_score, f1_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.base import BaseEstimator

import warnings
from sklearn.exceptions import ConvergenceWarning

# Отключение только ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class Clustering_scorer:
    """
    Класс для оценки и подбора гиперпараметров моделей кластеризации.
    """

    METRICS = {
        "silhouette": silhouette_score,
        "nmi": normalized_mutual_info_score,
        "ari": adjusted_rand_score,
        "accuracy": accuracy_score,
        "precision": lambda y_true, labels: precision_score(y_true, labels, average='macro'),
        "recall": lambda y_true, labels: recall_score(y_true, labels, average='macro'),
        "f1": lambda y_true, labels: f1_score(y_true, labels, average='macro')
    }

    @classmethod
    def _create_search_space(cls, param_grid):
        space = {}
        for name, info in param_grid.items():
            param_type = info.get("type")
            if param_type == "Integer":
                low = info["low"]
                high = info["high"]
                step = info.get("step", 1)  # по умолчанию шаг 1
                space[name] = list(range(low, high + 1, step))
            elif param_type == "Categorical":
                space[name] = info["values"]
            elif param_type == "Real":
                num_points = info.get("num", 5)
                space[name] = np.linspace(info["low"], info["high"], num=num_points).tolist()
            else:
                raise ValueError(f"Тип параметра '{param_type}' не поддерживается.")
        return space

    @classmethod
    def evaluate_metrics(cls, labels, X=None, y_true=None):
        results = {}

        unique_labels, counts = np.unique(labels, return_counts=True)
        results.update({
            "n_clusters": len(unique_labels),
            "cluster_distribution": {int(l): int(c) for l, c in zip(unique_labels, counts)}
        })

        # Silhouette Score
        if X is not None and len(unique_labels) > 1:
            try:
                results["silhouette"] = silhouette_score(X, labels)
                results["calinski_harabasz_score"] = calinski_harabasz_score(X, labels)
                results["davies_bouldin_score"] = davies_bouldin_score(X, labels)
            except:
                results["silhouette"] = None
                results["calinski_harabasz_score"] = None
                results["davies_bouldin_score"] = None
        else:
            results["silhouette"] = None
            results["calinski_harabasz_score"] = None
            results["davies_bouldin_score"] = None

        # Clustering metrics
        if y_true is not None:
            results["nmi"] = normalized_mutual_info_score(y_true, labels)
            results["ari"] = adjusted_rand_score(y_true, labels)

            # Добавляем метрики классификации
            results["accuracy"] = accuracy_score(y_true, labels)

            # Используем macro-усреднение, чтобы избежать зависимости от порядка меток
            results["precision"] = precision_score(y_true, labels, average='macro', zero_division=0)
            results["recall"] = recall_score(y_true, labels, average='macro', zero_division=0)
            results["f1"] = f1_score(y_true, labels, average='macro', zero_division=0)
        else:
            results["nmi"] = None
            results["ari"] = None
            results["accuracy"] = None
            results["precision"] = None
            results["recall"] = None
            results["f1"] = None

        return results

    @classmethod
    def grid_search(cls, model, X, y_true=None, config:dict=None):
        param_grid = config["param_grid"]
        search_space = {}

        for name, info in param_grid.items():
            param_type = info.get("type")
            if param_type == "Integer":
                low = info["low"]
                high = info["high"]
                step = info.get("step", 1)
                search_space[name] = list(range(low, high + 1, step))
            elif param_type == "Categorical":
                search_space[name] = info["values"]
            elif param_type == "Real":
                num_points = info.get("num", 5)
                search_space[name] = np.linspace(info["low"], info["high"], num=num_points).tolist()
            else:
                raise ValueError(f"Тип параметра '{param_type}' не поддерживается.")

        all_results = []
        for params in tqdm(ParameterGrid(search_space), desc="Grid Search Progress"):
            model.load_model_parameters(**params)
            labels, _ = model.fit_predict(X)

            metrics = cls.evaluate_metrics(labels, X=X, y_true=y_true)

            for metric_name in ["silhouette", "davies_bouldin_score", "calinski_harabasz_score", "nmi", "ari", "accuracy", "precision", "recall", "f1"]:
                all_results.append({
                    "parameter": str(params),
                    "metric": metric_name,
                    "value": metrics[metric_name]
                })

        df = pd.DataFrame(all_results)
        return df
