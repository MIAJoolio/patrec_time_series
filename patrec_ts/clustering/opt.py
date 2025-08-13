import numpy as np
from sklearn.model_selection import ParameterGrid
import optuna
from typing import Dict, Any, Tuple, List, Optional
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
from typing import Tuple, Union, List, Dict, Callable
import warnings
if optuna is not None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
from skopt.plots import plot_convergence, plot_objective


import numpy as np
import optuna
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from typing import Dict, Any, Tuple, List, Optional, Union, Callable
import warnings
from abc import ABC, abstractmethod

class BaseClusterOptimizer(ABC):
    """Базовый класс для оптимизаторов кластеризации."""
    
    def __init__(self, metric: str = 'silhouette'):
        self.metric_func, self.metric_direction = self._get_metric_config(metric)
    
    def _get_metric_config(self, metric: str) -> Tuple[Callable, str]:
        metrics = {
            'silhouette': (silhouette_score, 'maximize'),
            'calinski_harabasz': (calinski_harabasz_score, 'maximize'),
            'davies_bouldin': (davies_bouldin_score, 'minimize'),
            'nmi': (normalized_mutual_info_score, 'maximize'),
            'ari': (adjusted_rand_score, 'maximize')
        }
        if metric not in metrics:
            raise ValueError(f"Unknown metric '{metric}'. Available: {list(metrics.keys())}")
        return metrics[metric]
    
    def _compute_metric(self, X: np.ndarray, y_true: Optional[np.ndarray], labels: np.ndarray) -> float:
        if self.metric_func.__name__ in ['normalized_mutual_info_score', 'adjusted_rand_score']:
            if y_true is None:
                raise ValueError(f"Metric {self.metric_func.__name__} requires y_true")
            return self.metric_func(y_true, labels)
        return self.metric_func(X, labels)
    
    @abstractmethod
    def optimize(self, model, X: np.ndarray, y_true: Optional[np.ndarray] = None,
                param_space: Dict[str, Any] = None) -> Dict[str, Any]:
        pass


class OptunaClusterOptimizer(BaseClusterOptimizer):
    def optimize(self, 
                model, 
                X: np.ndarray, 
                y_true: Optional[np.ndarray] = None,
                param_space: Dict[str, Any] = None,
                n_trials: int = 50,
                random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Оптимизация гиперпараметров с помощью Optuna.
        
        Args:
            param_space: Пространство параметров в формате:
                {
                    'n_clusters': {'type': 'int', 'low': 2, 'high': 10},
                    'linkage': {'type': 'categorical', 'choices': ['ward', 'complete']},
                    'metric': {'type': 'categorical', 'choices': ['euclidean', 'manhattan']}
                }
        """
        def objective(trial):
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        name=param_name,
                        low=param_config['low'],
                        high=param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        name=param_name,
                        low=param_config['low'],
                        high=param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        name=param_name,
                        choices=param_config['choices']
                    )
            
            model.load_model_parameters(**params)
            labels, _ = model.fit_predict(X)
            score = self._compute_metric(X, y_true, labels)
            return score if self.metric_direction == 'maximize' else -score
        
        study = optuna.create_study(direction='maximize' if self.metric_direction == 'maximize' else 'minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }
        

class SkoptClusterOptimizer(BaseClusterOptimizer):
    """Оптимизатор с использованием Scikit-Optimize (Bayesian Optimization)."""
    
    def optimize(self, 
                model, 
                X: np.ndarray, 
                y_true: Optional[np.ndarray] = None,
                param_space: Optional[Dict[str, Dict]] = None,
                n_calls: int = 50,
                random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Оптимизация гиперпараметров с помощью Gaussian Process.
        
        Args:
            param_space: Пространство параметров в формате:
                {
                    'n_clusters': {'type': 'int', 'low': 2, 'high': 10},
                    'linkage': {'type': 'categorical', 'choices': ['ward', 'complete']},
                    'metric': {'type': 'categorical', 'choices': ['euclidean', 'manhattan']}
                }
        """
        from skopt.space import Integer, Categorical, Real
        
        if param_space is None:
            param_space = {
                'n_clusters': {'type': 'int', 'low': 2, 'high': 10}
            }

        # Преобразование param_space в формат skopt
        dimensions = []
        param_names = []
        for param_name, config in param_space.items():
            if config['type'] == 'int':
                dimensions.append(Integer(
                    low=config.get('low', 2),
                    high=config.get('high', 10),
                    name=param_name
                ))
            elif config['type'] == 'float':
                dimensions.append(Real(
                    low=config.get('low', 0.0),
                    high=config.get('high', 1.0),
                    prior='log-uniform' if config.get('log', False) else 'uniform',
                    name=param_name
                ))
            elif config['type'] == 'categorical':
                dimensions.append(Categorical(
                    categories=config['choices'],
                    name=param_name
                ))
            param_names.append(param_name)

        @use_named_args(dimensions=dimensions)
        def objective(**params):
            try:
                model.load_model_parameters(**params)
                labels, _ = model.fit_predict(X)
                score = self._compute_metric(X, y_true, labels)
                return -score if self.metric_direction == 'maximize' else score
            except Exception as e:
                return float('nan')

        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=random_state,
            n_initial_points=min(10, n_calls//2),
            verbose=False
        )

        best_params = {name: result.x[i] for i, name in enumerate(param_names)}
        best_score = -result.fun if self.metric_direction == 'maximize' else result.fun
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'result': result,
            'history': [{'params': {name: x[i] for i, name in enumerate(param_names)},
                        'score': -val if self.metric_direction == 'maximize' else val}
                       for x, val in zip(result.x_iters, result.func_vals)]
        }
        

class GridSearchClusterOptimizer(BaseClusterOptimizer):
    """Оптимизатор с использованием полного перебора параметров."""
    
    def optimize(self, 
                model, 
                X: np.ndarray, 
                y_true: Optional[np.ndarray] = None,
                param_space: Optional[Dict[str, Dict]] = None,
                verbose: bool = False,
                random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Полный перебор всех комбинаций параметров.
        
        Args:
            param_space: Пространство параметров в формате:
                {
                    'n_clusters': {'type': 'int', 'values': [2, 3, 4, 5]},
                    'linkage': {'type': 'categorical', 'choices': ['ward', 'complete']}
                }
        """
        from sklearn.model_selection import ParameterGrid
        import warnings
        
        if param_space is None:
            param_space = {
                'n_clusters': {'type': 'int', 'values': list(range(2, 11))}
            }

        # Подготовка сетки параметров
        grid = {}
        for param_name, config in param_space.items():
            if config['type'] == 'int':
                grid[param_name] = config.get('values', list(range(
                    config.get('low', 2), 
                    config.get('high', 10) + 1
                )))
            elif config['type'] == 'float':
                grid[param_name] = config['values']
            elif config['type'] == 'categorical':
                grid[param_name] = config['choices']

        param_grid = list(ParameterGrid(grid))
        
        if random_state is not None:
            np.random.seed(random_state)
            np.random.shuffle(param_grid)
        
        best_score = -np.inf if self.metric_direction == 'maximize' else np.inf
        best_params = None
        history = []
        
        for params in param_grid:
            try:
                model.load_model_parameters(**params)
                labels, _ = model.fit_predict(X)
                score = self._compute_metric(X, y_true, labels)
                
                history.append({
                    'params': params,
                    'score': score
                })
                
                if verbose:
                    print(f"Параметры: {params} -> Score: {score:.4f}")
                
                if (self.metric_direction == 'maximize' and score > best_score) or \
                   (self.metric_direction == 'minimize' and score < best_score):
                    best_score = score
                    best_params = params.copy()
                    
            except Exception as e:
                warnings.warn(f"Skipping {params}: {str(e)}")
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'history': history,
            'param_grid': param_grid
        }
        

class ClusterOptimizerFactory:
    """Фабрика для создания оптимизаторов."""
    
    @staticmethod
    def create(method: str, metric: str = 'silhouette') -> BaseClusterOptimizer:
        optimizers = {
            'optuna': OptunaClusterOptimizer,
            'skopt': SkoptClusterOptimizer,
            'grid': GridSearchClusterOptimizer
        }
        
        if method not in optimizers:
            raise ValueError(f"Unknown method '{method}'. Available: {list(optimizers.keys())}")
            
        return optimizers[method](metric)


class ClusterAnalyzer:
    def __init__(self, metrics=None):
        self.metrics = metrics or [
            "silhouette", "nmi", "ari", "accuracy", 
            "precision", "recall", "f1",
            "calinski_harabasz", "davies_bouldin"
        ]
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        results = {"n_clusters": len(np.unique(y_pred))}
        
        if "silhouette" in self.metrics and X is not None:
            results["silhouette"] = silhouette_score(X, y_pred) if len(np.unique(y_pred)) > 1 else None
        
        if "calinski_harabasz" in self.metrics and X is not None:
            results["calinski_harabasz"] = calinski_harabasz_score(X, y_pred) if len(np.unique(y_pred)) > 1 else None
            
        if "davies_bouldin" in self.metrics and X is not None:
            results["davies_bouldin"] = davies_bouldin_score(X, y_pred) if len(np.unique(y_pred)) > 1 else None
        
        if y_true is not None:
            if "nmi" in self.metrics:
                results["nmi"] = normalized_mutual_info_score(y_true, y_pred)
            
            if "ari" in self.metrics:
                results["ari"] = adjusted_rand_score(y_true, y_pred)
            
            if any(m in self.metrics for m in ["accuracy", "precision", "recall", "f1"]):
                y_pred_mapped, mapping, _ = self._map_clusters(y_true, y_pred)
                
                if "accuracy" in self.metrics:
                    results["accuracy"] = accuracy_score(y_true, y_pred_mapped)
                
                if "precision" in self.metrics:
                    results["precision"] = precision_score(y_true, y_pred_mapped, average='macro', zero_division=0)
                
                if "recall" in self.metrics:
                    results["recall"] = recall_score(y_true, y_pred_mapped, average='macro', zero_division=0)
                
                if "f1" in self.metrics:
                    results["f1"] = f1_score(y_true, y_pred_mapped, average='macro', zero_division=0)
        
        return results
    
    def _map_clusters(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        
        if len(unique_pred) > len(unique_true):
            conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_pred)
        else:
            conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_true)
        
        cost_matrix = -conf_matrix.copy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        mapping = {int(true): int(pred) for true, pred in zip(col_ind, row_ind)}
        
        y_pred_convert = np.asarray([mapping[val] for val in y_pred])
        return y_pred_convert, mapping, conf_matrix
    
    
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from typing import Dict, Any, List

def plot_optimization_history(study: 'optuna.Study') -> Figure:
    """Визуализирует историю оптимизации Optuna.
    
    Args:
        study: Объект исследования Optuna
        
    Returns:
        Figure: График истории оптимизации
    """
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    
    # Получаем все trials
    trials = study.trials
    
    # Сортируем по номеру trial
    trials.sort(key=lambda x: x.number)
    
    # Извлекаем значения метрики
    values = [t.value for t in trials if t.value is not None]
    numbers = [t.number for t in trials if t.value is not None]
    
    # Лучшее значение на каждом шаге
    best_values = [study.best_value for _ in numbers]
    
    ax.plot(numbers, values, 'b.', alpha=0.5, label='Trials')
    ax.plot(numbers, best_values, 'r-', label='Best so far')
    
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Objective value')
    ax.set_title('Optimization History')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig

def plot_parameter_importance(study: 'optuna.Study') -> Figure:
    """Визуализирует важность параметров для Optuna.
    
    Args:
        study: Объект исследования Optuna
        
    Returns:
        Figure: График важности параметров
    """
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    
    importances = optuna.importance.get_param_importances(study)
    param_names = list(importances.keys())
    importance_values = list(importances.values())
    
    y_pos = np.arange(len(param_names))
    
    ax.barh(y_pos, importance_values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Parameter Importance')
    ax.grid(True)
    
    plt.tight_layout()
    return fig

def plot_skopt_convergence(result: Dict[str, Any]) -> Figure:
    """Визуализирует сходимость для Skopt.
    
    Args:
        result: Результат оптимизации Skopt
        
    Returns:
        Figure: График сходимости
    """
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    
    plot_convergence(result['result'], ax=ax)
    ax.set_title('Convergence Plot')
    ax.grid(True)
    
    plt.tight_layout()
    return fig

def plot_grid_search_history(history: List[Dict[str, Any]], metric_name: str) -> Figure:
    """Визуализирует историю поиска по сетке.
    
    Args:
        history: История поиска по сетке
        metric_name: Название метрики для подписи оси Y
        
    Returns:
        Figure: График истории поиска
    """
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    
    n_clusters = [h['n_clusters'] for h in history]
    scores = [h['score'] for h in history]
    
    ax.plot(n_clusters, scores, 'bo-')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel(metric_name)
    ax.set_title('Grid Search Results')
    ax.grid(True)
    
    best_idx = np.argmax(scores) if len(scores) > 0 else 0
    ax.plot(n_clusters[best_idx], scores[best_idx], 'ro', markersize=10)
    
    plt.tight_layout()
    return fig