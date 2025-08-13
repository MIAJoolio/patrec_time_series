from typing import Callable, Any, Optional, Union, Dict
import numpy as np

from patrec_ts.feature_extraction.fe_classes import DecompModel, BaseOptimizer
from statsmodels.tsa.seasonal import STL, MSTL


class STL_Model(DecompModel):
    def __init__(self, series: np.ndarray, model_params: Dict[str, Any]):
        super().__init__(series, model_params)
        self._init_model()
        
    def _init_model(self):
        self.model = STL(self.series, **self.model_params)
        
    def _fit(self):
        pass        
    
    def _infer(self):
        result = self.model.fit()        
         
        return {
            'seasonal': result.seasonal,
            'trend': result.trend,
            'resid': result.resid,
            'period': self.model.period if self.model.period is not None else self.model_params['period'],
        }


class STL_Optimizer(BaseOptimizer):
    def __init__(self, series: np.ndarray, model: DecompModel,
                 grid_params: Dict[str, Any], opt_func: Callable):
        self.series = series
        self.model = model
        self.grid_params = grid_params
        self.opt_func = opt_func
        
    def find_optimal_values(self, trial):
        # Implement the optimization logic
        params = {}
        for param_name, config in self.grid_params.items():
            if config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, config['values'])
            elif config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, config['low'], config['high'])
            elif config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, config['low'], config['high'],
                    log=config.get('log', False))

        self.model = STL_Model(self.series, params)
                
        result = self.model._infer()
        return self.opt_func(result)
