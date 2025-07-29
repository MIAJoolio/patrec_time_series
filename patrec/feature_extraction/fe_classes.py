from abc import ABC, abstractmethod
from typing import Callable, Any, Optional, Union
import numpy as np

class Decomp_Model(ABC):
    @abstractmethod
    def __init__(self, series, model_params):
        self.series = series
        self.model_params = model_params
    
    @abstractmethod
    def _init_model(self):
        """
        set up model to fit and predict
        """
        pass
    
    @abstractmethod
    def _fit(self):
        """
        fit on data
        """
        pass
    
    @abstractmethod
    def _infer(self, series):
        """
        make prediction and transform it into decomposition, estimated parameters, plots and feature vectors 
        
        4-dimensions, 4-spaces
        """
        pass


class Base_Optimizer(ABC):
    """
    -> (1) best params (2) opt pic
    """
    
    def __init__(self, series:np.ndarray, model:Decomp_Model, grid_params:dict, opt_func: Callable):
        
        self.series = series        
        self.model = model
        self.grid_params = grid_params
        
        # ... full без вызова
        
    def optimize(self, trial, fitted:bool=True):
        """
        
        """
        
        params = {}
        for param_name, config in self.param_grid.items():
            if config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, config['values'])
            elif config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, config['low'], config['high'])
            elif config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, config['low'], config['high'],
                    log=config.get('log', False))

        self.model = self.model(**params)
        
        if fitted:
            self.model._fit()
        
        result = self.model._infer(self.series)
        
        return self.opt_func(result)
    
    
