from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import time
from typing import Optional, Tuple, Any, List, Dict


class Base_Clustering_Model(ABC):
    """
    Abstract base class for clustering models used in time series analysis.
    This class provides a unified interface for training, prediction, and label mapping
    with support for evaluation via confusion matrix and cluster-to-class alignment.

    Attributes:
        model: Placeholder for the actual clustering model (e.g., KMeans, DBSCAN).
        default_params: Dictionary of default hyperparameters for the model.
        config: Configuration dictionary that may store runtime settings.
        mapping_: Dictionary mapping predicted clusters to true classes after optimization.
        labels_: Tuple of predicted labels (train, test) from fit_predict().
        confusion_matrix_: Confusion matrix between true labels and mapped predicted labels.
    """
    
    def __init__(self):
        self.model = None
        self.default_params = {}
        self.config = {}
        self.mapping_ = None
        self.labels_ = None
        self.confusion_matrix_ = None
        self.time_fitted = None  # New attribute for tracking fitting time

    def load_model_parameters(self, **params) -> Dict[str, Any]:
        """Updates model parameters and reinitializes the model."""
        new_params = self.default_params.copy()
        new_params.update(params)
        self.default_params = new_params
        self._init_model()
        return new_params

    @abstractmethod
    def _init_model(self):
        """Initialize the concrete model instance."""
        pass

    @abstractmethod
    def fit_predict(self, 
                  X_train: np.ndarray, 
                  X_test: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Trains the model and predicts clusters with time tracking.
        
        Args:
            X_train: Training data (n_samples, n_features) or (n_samples, n_timesteps, n_features)
            X_test: Optional test data
            
        Returns:
            Tuple of (train_labels, test_labels)
        """
        pass

    def timed_fit_predict(self, 
                         X_train: np.ndarray, 
                         X_test: Optional[np.ndarray] = None
                         ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Wrapper for fit_predict that measures and stores execution time.
        """
        start_time = time.time()
        train_labels, test_labels = self.fit_predict(X_train, X_test)
        self.time_fitted = time.time() - start_time
        return train_labels, test_labels

    def get_cluster_centers(self) -> np.ndarray:
        """Returns cluster centers if available."""
        if not hasattr(self.model, 'cluster_centers_'):
            raise ValueError("Model not fitted yet. Call fit_predict() first.")
        return self.model.cluster_centers_

    def get_inertia(self) -> float:
        """Returns inertia if available."""
        if not hasattr(self.model, 'inertia_'):
            raise ValueError("Model not fitted yet. Call fit_predict() first.")
        return self.model.inertia_

    def _map_clusters(self, y_true: np.ndarray, y_pred: np.ndarray, filter_by: str = 'true') -> tuple:
        """
        Maps predicted cluster labels to true class labels using the Hungarian algorithm
        to minimize misclassification cost. This allows comparison of clustering results
        against ground truth.

        Args:
            y_true: Array of true class labels (n_samples,).
            y_pred: Array of predicted cluster labels (n_samples,).
            filter_by: Controls which set of unique labels to use as reference ('true' or 'pred').

        Returns:
            tuple: (mapped_labels, mapping_dict, confusion_matrix)
                - mapped_labels: Predicted labels after optimal reassignment.
                - mapping_dict: Dictionary mapping {true_class: assigned_cluster} after optimization.
                - confusion_matrix: Confusion matrix between true and mapped predicted labels.
        """
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)

        # Construct confusion matrix based on larger set to avoid missing classes
        if len(unique_pred) > len(unique_true):
            conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_pred)
        else:
            conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_true)

        # Use negative cost matrix for maximization (Hungarian algorithm minimizes)
        cost_matrix = -conf_matrix.copy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Build mapping: true class index into predicted cluster index
        self.mapping_ = {int(true): int(pred) for true, pred in zip(col_ind, row_ind)}

        # Apply mapping to predictions
        y_pred_convert = np.asarray([self.mapping_[val] for val in y_pred])
        return y_pred_convert, self.mapping_, conf_matrix

    def predict_with_mapping(self, X_train, X_test=None, y_true=None, return_all=False):
        """
        Fits the model on training data and predicts cluster labels. If true labels are provided,
        it maps predicted clusters to true classes using optimal assignment.

        Args:
            X_train: Training time series data (n_train_samples, n_features).
            X_test: Optional test time series data (n_test_samples, n_features).
            y_true: True class labels (n_samples,) for mapping predictions to classes.
            return_all: If True, returns additional metadata (mapping, confusion matrix).

        Returns:
            If return_all is False:
                tuple: (train_labels, test_labels)
            If return_all is True:
                tuple: (train_labels, test_labels, mapping, confusion_matrix)
        """
        train_labels, test_labels = self.fit_predict(X_train, X_test)
        self.labels_ = (train_labels, test_labels)

        if y_true is not None:
            mapped_labels, mapping, conf_matrix = self._map_clusters(y_true, train_labels)
            self.mapping_ = mapping
            self.confusion_matrix_ = conf_matrix

            if return_all:
                return train_labels, test_labels, mapping, conf_matrix

        return train_labels, test_labels


def visualize_cluster_errors(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mapping: dict,
    save_dir: str = '.temp/clutering',
    prefix: str = '',
    figsize: tuple = (10, 3),
    show_legend: bool = False
) -> None:
    """
    Visualizes classification errors in clustering by plotting time series grouped by true class
    and highlighting correctly vs incorrectly assigned samples.

    Each subplot shows all time series belonging to a specific true class. Correctly classified
    series are shown in yellow. Misclassified series are colored by their predicted cluster.

    Parameters:
        X: Training time series data (n_samples, n_features).
        y_true: True class labels (n_samples,).
        y_pred: Predicted cluster labels (n_samples,).
        mapping: Dictionary mapping {true_class: predicted_cluster} after optimal assignment.
        save_dir: Directory path to save plots (created if not exists).
        figsize: Figure size (width, height) in inches.
        show_legend: Whether to display legend in plots.

    Returns:
        None. Saves plots to disk.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Invert mapping for easier lookup: {predicted_cluster: true_class}
    inv_mapping = {v: k for k, v in mapping.items()}

    # Color palette for different clusters (for wrong classifications)
    colors_meta = ["#FD3E1C", "#1CFD42", "#FD1CB2", "#1C33FD",
                   "#805380", "#41F7B1", "#DFBCFF", "#FFFD7D",
                   "#A9C7FF", "#3D3D3D"]

    for true_class in np.unique(y_true):
        plt.figure(figsize=figsize)

        # Correct classifications: true class matches predicted cluster
        correct_mask = (y_true == true_class) & (y_pred == mapping[true_class])
        correct_data = X[correct_mask]
        for series in correct_data:
            plt.plot(series, color="#FDC11C", label="Correct" if not show_legend else f"Correct (Cluster {mapping[true_class]})")

        # Incorrect classifications: true class ≠ predicted cluster
        error_mask = (y_true == true_class) & (y_pred != mapping[true_class])
        error_data = X[error_mask]
        error_clusters = y_pred[error_mask]

        for cluster in np.unique(error_clusters):
            cluster_data = error_data[error_clusters == cluster]
            color = colors_meta[cluster % len(colors_meta)]
            for series in cluster_data:
                plt.plot(series, color=color, label=f"Wrong (Cluster {cluster})" if show_legend else "")

        # Remove duplicate legend entries
        if show_legend:
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

        plt.title(f'True Class {true_class} (Mapped to Cluster {mapping[true_class]})')
        plt.tight_layout()

        save_path = f'{save_dir}/class_{true_class}_cluster_{mapping[true_class]}_{prefix}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path} | Correct: {sum(correct_mask)}, Errors: {sum(error_mask)}")


def visualize_cluster_composition(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mapping: dict,
    save_dir: str = '.temp/clutering',
    prefix: str = '',
    figsize: tuple = (10, 3),
    show_legend: bool = False
) -> None:
    """
    Visualizes the composition of each predicted cluster by showing the distribution of true classes.

    For each predicted cluster, all time series are plotted, colored by their true class.
    A black dashed line shows the median series of the cluster for visual reference.

    Parameters:
        X: Full dataset (n_samples, n_features) — can be train or test.
        y_true: True class labels (n_samples,).
        y_pred: Predicted cluster labels (n_samples,).
        mapping: Dictionary mapping {true_class: predicted_cluster} after optimal assignment.
        save_dir: Directory to save generated plots.
        figsize: Figure size (width, height) in inches.
        show_legend: Whether to include legend in plots.

    Returns:
        None. Saves plots to disk.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Invert mapping: {predicted_cluster: true_class}
    inv_mapping = {v: k for k, v in mapping.items()}

    # Color palette for true classes
    true_colors = ["#1CFD42", "#FD3E1C", "#1C33FD", "#FD1CB2",
                   "#805380", "#41F7B1", "#DFBCFF", "#FFFD7D",
                   "#A9C7FF", "#3D3D3D"]

    for cluster in np.unique(y_pred):
        plt.figure(figsize=figsize)

        # Get all data points in this cluster
        cluster_data = X[y_pred == cluster]
        true_labels = y_true[y_pred == cluster]

        # Plot each time series by true class
        for true_class in np.unique(true_labels):
            class_data = cluster_data[true_labels == true_class]
            color = true_colors[true_class % len(true_colors)]
            for series in class_data:
                label = f"True class {true_class}" if show_legend else ""
                plt.plot(series, color=color, label=label)

        # Add median series as reference (black dashed line)
        if len(cluster_data) > 0:
            median_series = np.median(cluster_data, axis=0)
            plt.plot(median_series, color='black', linewidth=2, linestyle='--',
                     label='Cluster median' if show_legend else "")

        # Handle legend duplicates
        if show_legend:
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

        # Title includes purity metric
        main_true_class = inv_mapping.get(cluster, -1)
        title = f"Cluster {cluster} (Mapped from class {main_true_class})"
        if main_true_class != -1:
            purity = np.mean(true_labels == main_true_class)
            title += f" | Purity: {purity:.1%}"
        plt.title(title)
        plt.tight_layout()

        save_path = f'{save_dir}/cluster_{cluster}_composition_{prefix}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_path} | Total: {len(cluster_data)} | "
              f"Main class: {main_true_class} ({np.sum(true_labels == main_true_class)} samples)")