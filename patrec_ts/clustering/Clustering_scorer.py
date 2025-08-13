import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

# Метрики
from sklearn.metrics import (
    silhouette_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    confusion_matrix
)
from scipy.optimize import linear_sum_assignment


class Clustering_scorer:
    def __init__(self, metrics=None, debug=False):
        self.metrics = metrics or [
            "silhouette", "nmi", "ari", "accuracy", "precision", "recall", "f1",
            "calinski_harabasz_score", "davies_bouldin_score"
        ]
        self.debug = debug
        self.all_results = []

    def _build_search_space(self, param_grid):
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
        return search_space

    def _map_clusters(self, y_true, y_pred):
        unique_labels = np.unique(y_true)
        conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_labels)

        cost_matrix = -conf_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        mapping = {int(pred): true for pred, true in zip(col_ind, row_ind)}
        mapped_pred = np.array([mapping.get(label, -1) for label in y_pred])

        return mapped_pred

    def evaluate_metrics(self, y_true, y_pred, X=None):
        results = {}

        unique_clusters = np.unique(y_pred)
        results["n_clusters"] = len(unique_clusters)

        # Внутрикластерные метрики
        if "silhouette" in self.metrics and X is not None and len(unique_clusters) > 1:
            try:
                results["silhouette"] = silhouette_score(X, y_pred)
            except:
                results["silhouette"] = None
        else:
            results["silhouette"] = None

        if "calinski_harabasz_score" in self.metrics and X is not None and len(unique_clusters) > 1:
            try:
                results["calinski_harabasz_score"] = calinski_harabasz_score(X, y_pred)
            except:
                results["calinski_harabasz_score"] = None
        else:
            results["calinski_harabasz_score"] = None

        if "davies_bouldin_score" in self.metrics and X is not None and len(unique_clusters) > 1:
            try:
                results["davies_bouldin_score"] = davies_bouldin_score(X, y_pred)
            except:
                results["davies_bouldin_score"] = None
        else:
            results["davies_bouldin_score"] = None

        # Внешние метрики (только если есть y_true)
        if y_true is not None:
            mapped_pred = self._map_clusters(y_true, y_pred)

            if "nmi" in self.metrics:
                results["nmi"] = normalized_mutual_info_score(y_true, y_pred)
            if "ari" in self.metrics:
                results["ari"] = adjusted_rand_score(y_true, y_pred)
            if "accuracy" in self.metrics:
                results["accuracy"] = accuracy_score(y_true, mapped_pred)
            if "precision" in self.metrics:
                results["precision"] = precision_score(y_true, mapped_pred, average='macro', zero_division=0)
            if "recall" in self.metrics:
                results["recall"] = recall_score(y_true, mapped_pred, average='macro', zero_division=0)
            if "f1" in self.metrics:
                results["f1"] = f1_score(y_true, mapped_pred, average='macro', zero_division=0)
        else:
            for m in ["nmi", "ari", "accuracy", "precision", "recall", "f1"]:
                if m in self.metrics:
                    results[m] = None

        return results

    def grid_search(self, model, X, y_true=None, model_name="clustering_model", param_grid=None):
        """
        Выполняет перебор параметров и возвращает результаты в нужном формате.
        Если debug=True — сохраняет все данные в self.all_results.
        """

        if param_grid is None:
            config = getattr(model, 'config', {})
            param_grid = {}
            param_grid["param_grid"] = config.get("param_grid", {})

        search_space = self._build_search_space(param_grid['param_grid']) 
        print(search_space)
        
        all_rows = []

        for params in tqdm(ParameterGrid(search_space), desc="Grid Search"):
            try:
                applied_params = model.load_model_parameters(**params)
                pred_labels, _ = model.fit_predict(X)
                metrics = self.evaluate_metrics(y_true, pred_labels, X=X)

                row_base = {
                    "parameter": applied_params,
                    "model_name": model_name
                }

                if self.debug:
                    self.all_results.append({
                        "params": applied_params,
                        "X": X,
                        "true_labels": y_true,
                        "pred_labels": pred_labels,
                        "metrics": metrics.copy()
                    })

                for metric_name in self.metrics:
                    row = row_base.copy()
                    row["metric"] = metric_name
                    row["value"] = metrics.get(metric_name)
                    all_rows.append(row)

            except Exception as e:
                print(f"Ошибка при использовании {params}: {e}")
                continue

        return pd.DataFrame(all_rows)

    def get_debug_data(self):
        if not self.debug:
            raise RuntimeError("Debug mode выключен. Результаты не сохранены.")
        return self.all_results