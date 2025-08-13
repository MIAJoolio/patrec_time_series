from typing import Dict, Any, Callable, Optional
import numpy as np

from sklearn.metrics import mean_squared_error

import optuna


class TrendFeatureExtractor:
    """
    Feature extraction focused solely on trend component characteristics.
    Provides statistical, shape and temporal features of the trend.
    """
    
    @staticmethod
    def extract_all(trend_component: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive set of trend-only features.
        
        Args:
            trend_component: Extracted trend component from decomposition
            
        Returns:
            Dictionary with feature names and values
        """
        features = {}
        features.update(TrendFeatureExtractor._basic_trend_stats(trend_component))
        features.update(TrendFeatureExtractor._trend_shape_features(trend_component))
        features.update(TrendFeatureExtractor._trend_temporal_features(trend_component))
        return features
    
    @staticmethod
    def _basic_trend_stats(trend: np.ndarray) -> Dict[str, float]:
        """Basic statistical properties of the trend"""
        features = {
            'trend_mean': float(np.mean(trend)),
            'trend_std': float(np.std(trend)),
            'trend_min': float(np.min(trend)),
            'trend_max': float(np.max(trend)),
        }
        return features


class TrendDecompositionOptimizer:
    """
    Optimizes trend decomposition parameters using only trend-related metrics.
    Completely separate from residual or structural break analysis.
    """
    
    def __init__(self, decomposer_class: Callable):
        """
        Args:
            decomposer_class: Trend decomposition class to optimize
        """
        self.decomposer_class = decomposer_class
        self.study = None
        self.best_params = None
        
    def optimize(self, 
                data: np.ndarray,
                param_grid: Dict[str, Dict[str, Any]],
                n_trials: int = 50,
                direction: str = 'minimize') -> Dict[str, Any]:
        """
        Optimize decomposition parameters using trend-only metrics.
        
        Args:
            data: Input time series data
            param_grid: Parameter search space configuration
            n_trials: Number of optimization trials
            direction: Optimization direction ('minimize' or 'maximize')
            
        Returns:
            Dictionary with optimization results
        """
        study = optuna.create_study(direction=direction)
        objective = self._create_objective(data, param_grid)
        study.optimize(objective, n_trials=n_trials)
        
        self.study = study
        self.best_params = study.best_params
        
        # Get best decomposition and features
        best_decomposer = self.decomposer_class(**study.best_params)
        decomposition = best_decomposer.decompose(data)
        features = TrendFeatureExtractor.extract_all(decomposition.component)
        
        return {
            'best_params': study.best_params,
            'trend_features': features,
            'trend_component': decomposition.component,
            'best_value': study.best_value
        }
    
    def _create_objective(self, 
                         data: np.ndarray,
                         param_grid: Dict[str, Dict[str, Any]]) -> Callable:
        """Create optimization objective function"""
        def objective(trial: optuna.Trial) -> float:
            params = {}
            
            # Sample parameters according to the grid
            for param_name, config in param_grid.items():
                if config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, config['values'])
                elif config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, config['low'], config['high'])
                elif config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, config['low'], config['high'], 
                        log=config.get('log', False))
            
            try:
                # Get trend component
                decomposer = self.decomposer_class(**params)
                decomposition = decomposer.decompose(data)
                trend = decomposition.component
                
                # Calculate optimization metrics
                rmse = np.sqrt(mean_squared_error(data, trend))
                features = TrendFeatureExtractor.extract_all(trend)
                
                # Composite score focusing on trend quality
                score = rmse * (1 + features['trend_avg_curvature'])
                return score
                
            except Exception as e:
                return float('inf')  # Return bad score if decomposition fails
            
        return objective