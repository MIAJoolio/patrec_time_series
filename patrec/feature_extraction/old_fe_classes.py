from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum, auto
import time 

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_squared_error

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.metrics import mean_squared_error

class FE_component_type(Enum):
    """
    Enumeration of time series component types.
    
    Attributes:
        TREND: Long-term direction of the series
        SEASONAL: Periodic fluctuations
        RESIDUAL: Irregular remainder after decomposition
        NOISE: Random variations
        SHIFT: Structural breaks/level shifts
    """
    TREND = auto()
    SEASONAL = auto()
    RESIDUAL = auto()
    NOISE = auto()
    SHIFT = auto()

@dataclass
class Decomposition_result:
    """
    Container for storing decomposition results with associated metadata.
    
    Attributes:
        component: Extracted component values (trend, seasonal, etc.)
        component_type: Type of the component (from FE_component_type)
        method_name: Name of the decomposition method used
        params: Dictionary of method parameters
        stats: Execution statistics (time, memory, etc.)
        features: Extracted features from this component (optional)
    """
    component: np.ndarray
    component_type: FE_component_type
    method_name: str
    params: Dict[str, Any]
    stats: Dict[str, Any]
    features: Optional[Dict[str, Any]] = None

class Base_decomposer(ABC):
    """
    Abstract base class for time series decomposition methods.
    
    Implementations should override the decompose() method to perform specific
    decomposition techniques (e.g., STL, wavelet, etc.).
    """
    @abstractmethod
    def decompose(self, data: np.ndarray, **kwargs) -> Union[Decomposition_result, Dict[FE_component_type, Decomposition_result]]:
        """
        Decompose the input time series into components.
        
        Args:
            data: 1D numpy array containing the time series
            **kwargs: Method-specific parameters
            
        Returns:
            Either a single Decomposition_result or a dictionary mapping 
            FE_component_type to Decomposition_result for multiple components
        """
        pass

class Base_fe_extractor(ABC):
    """
    Abstract base class for feature extraction from time series components.
    
    Implementations should override extract_features() to compute specific
    features from decomposed components.
    """
    @abstractmethod
    def extract_features(self, component: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Extract features from a time series component.
        
        Args:
            component: 1D numpy array of component values
            **kwargs: Additional parameters for feature calculation
            
        Returns:
            Dictionary of feature names to their values. Values should be
            JSON-serializable (float, int, str, bool, etc.)
        """
        pass


class Base_optimizer:
    """Base class for trend decomposition optimization with safety limits"""
    
    @staticmethod
    def optimize(
        decomposer_class: Callable,
        data: np.ndarray,
        param_grid: Dict[str, Dict[str, Any]],
        metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        n_trials: int = 50,
        direction: str = "minimize",
        timeout: Optional[float] = None,
        max_failures: int = 10
    ) -> Dict[str, Any]:
        """
        Optimize decomposition parameters with safety limits.
        
        Args:
            timeout: Maximum time in seconds (None for no limit)
            max_failures: Maximum allowed failed trials before aborting
        """
        if metric_func is None:
            metric_func = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))

        # Store optimization state
        state = {
            'failures': 0,
            'start_time': time.time(),
            'should_stop': False
        }

        def objective(trial: optuna.Trial) -> float:
            # Check stop conditions
            if state['should_stop']:
                raise optuna.TrialPruned()
                
            if timeout and (time.time() - state['start_time']) > timeout:
                state['should_stop'] = True
                raise optuna.TrialPruned()
                
            # Suggest parameters
            params = {}
            for param_name, config in param_grid.items():
                if config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, config['values'])
                elif config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, config['low'], config['high'])
                elif config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, config['low'], config['high'],
                        log=config.get('log', False))

            # Evaluate
            try:
                decomposer = decomposer_class(**params)
                result = decomposer.decompose(data, opt=False)  # Disable nested optimization
                trend = result.component if isinstance(result, Decomposition_result) else result
                return metric_func(data, trend)
            except Exception as e:
                state['failures'] += 1
                if state['failures'] >= max_failures:
                    state['should_stop'] = True
                return float('inf')

        # Configure study
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42)  # For reproducibility
        )
        
        # Run optimization
        try:
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
        except Exception as e:
            print(f"Optimization stopped: {str(e)}")

        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'n_failures': state['failures']
        }
        

class FeaturePipeline:
    """
    Orchestrates the complete feature extraction workflow:
    1. Decomposition of time series into components
    2. Feature extraction from each component
    3. Aggregation of results
    
    Attributes:
        decomposers: List of decomposition methods to apply
        feature_extractors: Dictionary mapping component types to their
                           respective feature extractors
    """
    def __init__(self, 
                 decomposers: List[Base_decomposer],
                 feature_extractors: Optional[Dict[FE_component_type, Base_fe_extractor]] = None):
        """
        Initialize the pipeline with decomposition and feature extraction methods.
        
        Args:
            decomposers: List of decomposition method instances
            feature_extractors: Optional dictionary of component type to
                              feature extractor instances. If None, decomposers
                              must include their own feature extraction.
        """
        self.decomposers = decomposers
        self.feature_extractors = feature_extractors or {}

    def process(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Execute the complete feature extraction pipeline.
        
        Args:
            data: Input time series (1D numpy array)
            
        Returns:
            Nested dictionary of features organized by:
            {decomposer_name}_{component_type: {feature_name: value}}
        """
        results = {}
        
        for decomposer in self.decomposers:
            # Step 1: Decompose time series
            decomp_result = decomposer.decompose(data)
            
            # Handle both single and multiple component returns
            components = (
                [decomp_result] if isinstance(decomp_result, Decomposition_result) 
                else list(decomp_result.values())
            )
            
            for result in components:
                # Step 2: Extract features if not already done by decomposer
                if result.features is None and result.component_type in self.feature_extractors:
                    result.features = self.feature_extractors[result.component_type].extract_features(result.component)
                
                # Step 3: Store results
                key = f"{decomposer.__class__.__name__}_{result.component_type.name.lower()}"
                if result.features:
                    results[key] = result.features
                    
        return results