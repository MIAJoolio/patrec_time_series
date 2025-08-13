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
from typing import Literal
import numpy as np

def map_clusters(y_true, y_pred, filter='pred'):
    """Сопоставляет истинные метки с предсказанными кластерами."""
    
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    if filter == "pred":
        conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_pred)
    elif filter == "true":
        conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_true)

    # Construct confusion matrix based on larger set to avoid missing classes
    if len(unique_pred) > len(unique_true):
        conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_pred)
    else:
        conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_true)

    cost_matrix = -conf_matrix.copy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {int(true): int(pred) for true, pred in zip(col_ind, row_ind)}
    # 
    y_pred_convert = np.asarray([mapping[val] for val in y_pred])
    
    return y_pred_convert, mapping, conf_matrix 


class Clustering_Metrics_Evaluator:
    def __init__(self, metrics=None, y_true=None, y_pred=None, X_feat=None):
        self.metrics = metrics or [
            "silhouette", "nmi", "ari", "accuracy", "precision", "recall", "f1",
            "calinski_harabasz_score", "davies_bouldin_score"
        ]
        self._validate_input(y_true, y_pred, X_feat)     
        self.pred_convert, self.conf_mtx, self.mapping = self._map_clusters()
           
    def _validate_input(self, y_true, y_pred, X_feat):
        """Проверяет корректность входных данных."""
        if y_true is None or y_pred is None:
            raise ValueError("y_true и y_pred должны быть предоставлены.")
        if len(y_true) != len(y_pred):
            raise ValueError(f"Размеры y_true ({len(y_true)}) и y_pred ({len(y_pred)}) не совпадают.")
        
        if X_feat is not None:
            if len(X_feat.shape) == 3:
                X_feat = X_feat.squeeze(1)
        
            # if X_feat.shape[1] > 1:
            #     raise ValueError("3D data and it's Incorrect")
            
        self.X_feat = X_feat
        self.y_true = y_true 
        self.unique_true = np.unique(self.y_true)
        self.y_pred = y_pred
        self.unique_pred = np.unique(self.y_pred)
        
        if len(self.unique_pred) < len(self.unique_true):
            print(f"Предупреждение: Количество кластеров ({len(self.unique_pred)}) < количество классов ({len(self.unique_true)}).")

    def _map_clusters(self, filter_by:Literal['pred', 'true']='true'):
        return map_clusters(self.y_true, self.y_pred, filter_by)
    
    def _norm_conf_mtx(self):
        """Преобразует матрицу ошибок в проценты."""
        total = self.conf_mtx.sum()
        return self.conf_mtx / total * 100 if total > 0 else self.conf_mtx

    # def _():
    
    def _get_cluster_analysis(self, y_true, y_pred, X, mapping):
        """Анализирует структуру кластеров и возвращает информацию о каждом из них."""
        results = {}
        unique_clusters = np.unique([mapping[y] for y in y_pred])
        for cluster_id in unique_clusters:
            indices = np.where([mapping[y] == cluster_id for y in y_pred])[0]
            if len(indices) == 0:
                continue
            cluster_data = X[indices]
            true_labels = y_true[indices]
            # Преобразуем y_pred в предсказанные метки с учётом маппинга
            mapped_y_pred = np.array([mapping[y] for y in y_pred])
            pred_labels = mapped_y_pred[indices]
            correct = np.sum(true_labels == pred_labels)
            incorrect = len(indices) - correct
            wrong_indices = np.where(pred_labels != true_labels)[0]
            results[cluster_id] = {
                "total_samples": len(indices),
                "correct": correct,
                "incorrect": incorrect,
                "wrong_samples": {
                    "indices": indices[wrong_indices],
                    "data": cluster_data[wrong_indices],
                    "true_labels": true_labels[wrong_indices],
                    "predicted_labels": pred_labels[wrong_indices]
                }
            }
        return results

    def evaluate(self, y_true, y_pred, X=None):
        results = {}
        unique_clusters = np.unique(y_pred)
        results["n_clusters"] = len(unique_clusters)
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
        if y_true is not None:
            _, mapping, _ = self._map_clusters()
            print(mapping)
            if "nmi" in self.metrics:
                results["nmi"] = normalized_mutual_info_score(y_true, y_pred)
            if "ari" in self.metrics:
                results["ari"] = adjusted_rand_score(y_true, y_pred)
            if "accuracy" in self.metrics:
                results["accuracy"] = accuracy_score(y_true, [mapping[y] for y in y_pred])
            if "precision" in self.metrics:
                results["precision"] = precision_score(y_true, [mapping[y] for y in y_pred], average='macro', zero_division=0)
            if "recall" in self.metrics:
                results["recall"] = recall_score(y_true, [mapping[y] for y in y_pred], average='macro', zero_division=0)
            if "f1" in self.metrics:
                results["f1"] = f1_score(y_true, [mapping[y] for y in y_pred], average='macro', zero_division=0)
        else:
            for m in ["nmi", "ari", "accuracy", "precision", "recall", "f1"]:
                if m in self.metrics:
                    results[m] = None
        return results

    def get_confusion_matrix(self, y_true, y_pred):
        """Возвращает матрицу ошибок для кластеров."""
        self._validate_input(y_true, y_pred)
        return confusion_matrix(y_true, y_pred)

    def get_tpr_fpr_tnr_fnr(self, y_true, y_pred):
        """Вычисляет TP, FP, TN, FN для каждого класса."""
        self._validate_input(y_true, y_pred)
        cm = self.get_confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]
        tpr = []
        fpr = []
        tnr = []
        fnr = []
        for i in range(n_classes):
            tp = cm[i, i]
            fp = sum(cm[:, i]) - tp
            fn = sum(cm[i, :]) - tp
            tn = sum(sum(cm)) - (tp + fp + fn)
            tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
            tnr.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            fnr.append(fn / (fn + tp) if (fn + tp) > 0 else 0)
        return {
            "TPR": tpr,
            "FPR": fpr,
            "TNR": tnr,
            "FNR": fnr
        }

    def find_misclassified_samples(self, y_true, y_pred, X=None):
        """
        Находит экземпляры, которые были присвоены в неправильные кластеры.
        Возвращает информацию о каждом неправильно классифицированном образце,
        включая оригинальный класс, предсказанный кластер и сопоставленный класс.
        """
        if y_true is None or X is None:
            raise ValueError("y_true и X должны быть предоставлены.")
        
        # Сопоставляем кластеры
        mapped_pred = self._map_clusters(y_true, y_pred)
        
        # Находим индексы, где предсказание не совпадает с истиной
        misclassified_indices = np.where(mapped_pred != y_true)[0]
        
        # Формируем список неправильно классифицированных точек
        results = []
        for idx in misclassified_indices:
            original_class = y_true[idx]
            predicted_cluster = y_pred[idx]
            mapped_class = mapped_pred[original_class]
            results.append({
                "index": idx,
                "original_class": original_class,
                "predicted_cluster": predicted_cluster,
                "mapped_class": mapped_class,
                "X": X[idx]
            })
        
        return results

    def analyze_cluster_structure(self, y_true, y_pred, X=None):
        """Анализирует структуру кластеров."""
        self._validate_input(y_true, y_pred)
        mapping = self._create_mapping(y_true, y_pred)
        return self._get_cluster_analysis(y_true, y_pred, X, mapping)

    def get_mapping(self, y_true, y_pred):
        """Возвращает словарь, который показывает, как истинные метки были сопоставлены с предсказанными кластерами."""
        self._validate_input(y_true, y_pred)
        return self._create_mapping(y_true, y_pred)

    def analyze_confusion_matrix(
        self,
        y_true, 
        y_pred, 
        X=None, 
        show_percentage=True, 
        filter_class=None, 
        by_original=True
    ):
        """
        Анализирует матрицу ошибок.
        
        Args:
            y_true (np.ndarray): Истинные метки.
            y_pred (np.ndarray): Предсказанные метки.
            X (np.ndarray): Матрица признаков.
            show_percentage (bool): Перевести значения в проценты.
            filter_class (int): Фильтр по оригинальному классу.
            by_original (bool): Фильтровать по оригинальным меткам (True) или предсказанным (False).
        
        Returns:
            dict: Словарь с анализом матрицы ошибок.
        """
        self._validate_input(y_true, y_pred)
        
        # Получаем маппинг
        mapping = self._map_clusters(y_true, y_pred)
        
        # Получаем исходную матрицу ошибок
        cm = confusion_matrix(y_true, y_pred)
        
        # Переводим в проценты
        if show_percentage:
            cm = self._convert_to_percentages(cm)
        
        # Если нужно фильтровать по оригинальному или предсказанному классу
        if filter_class is not None:
            if by_original:
                # Фильтруем по оригинальному классу
                filtered_cm = cm[filter_class]
            else:
                # Фильтруем по предсказанному классу
                idx = np.argmax(cm, axis=0)
                filtered_cm = cm[:, idx == filter_class]
            cm = filtered_cm
        
        # Получаем индексы неправильно классифицированных
        misclassified_indices = self._get_misclassified_indices(y_true, y_pred, mapping)
        
        # Получаем аналитику по кластерам
        cluster_analysis = self._get_cluster_analysis(y_true, y_pred, X, mapping)
        
        return {
            "mapping": mapping,
            "confusion_matrix": cm,
            "misclassified_indices": misclassified_indices,
            "cluster_analysis": cluster_analysis
        }