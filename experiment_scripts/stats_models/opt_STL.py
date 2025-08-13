from typing import Callable, Any, Optional, Union, Dict, Literal
from pathlib import Path
# stats_methods
from patrec.feature_extraction.stats_methods import seasonal_correlation, seasonal_strength, STL_Model, STL_Optimizer
# files_helper
from patrec.utils.files_helper import JSON_Handler, YAML_Config
from patrec.utils.visuals import plot_series
# other
import numpy as np
from sklearn.metrics import accuracy_score, root_mean_squared_error
import optuna


def stl_implement_exp(result_path:Path, stl_grid:Optional[Union[YAML_Config, Path]], dataset:Optional[Union[JSON_Handler, Path]], opt_func_name:Literal['seasonal_corr', 'seasonal_str']):
    """
    Эксперимент на синтетике с известными значениями периодичности
    """
    # Load your data
    if isinstance(dataset, Path):
        dataset = JSON_Handler(filepath=dataset)
    
    if isinstance(stl_grid, Path):
        stl_grid = YAML_Config(filepath=stl_grid).data
    
    # 
    opt_funcs = {
        'seasonal_corr':seasonal_correlation,
        'seasonal_str':seasonal_strength
    }
    
    # log results
    metrics = {
        'accuracy':[],
        'rmse_residual_plus':[],
        'rmse_residual_minus':[]
    }
    
    est_params = {
        'best_stl_val':[],
        'best_stl_params':[]
    }
    
    for inx, time_series in enumerate(dataset.data['data']):

        ts_obj = np.asarray(time_series)

        # Create optimizer
        optimizer = STL_Optimizer(
            series=ts_obj,
            model=STL_Model,
            grid_params=stl_grid,
            opt_func=opt_funcs[opt_func_name]
        )

        # Optimize
        study = optuna.create_study(direction="maximize")
        study.optimize(optimizer.find_optimal_values, n_trials=200 if opt_func_name == 'seasonal_corr' else 30, timeout=20)
        trial = study.best_trial


        est_params['best_stl_val'].append(trial.value)
        est_params['best_stl_params'].append(trial.params)
        
        pic_path = result_path/"pics/"
        pic_path.mkdir(parents=True, exist_ok=True)
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(pic_path/f"row_{inx}.html")
        
        metrics['accuracy'].append(np.max([accuracy_score(lbl if isinstance(lbl, list) else [lbl], [trial.params['period']]) for lbl in dataset.data['labels'][inx]]))
        
        model = STL_Model(time_series, trial.params)
        result = model._infer()
        
        metrics['rmse_residual_plus'].append(np.round(root_mean_squared_error(time_series, np.asarray(result['trend']) + np.asarray(result['seasonal']) + np.asarray(result['resid'])), 4))
        metrics['rmse_residual_minus'].append(np.round(root_mean_squared_error(time_series, np.asarray(result['trend']) + np.asarray(result['seasonal'])), 4))
        
        
        plot_series([time_series, np.asarray(result['trend']) + np.asarray(result['seasonal']) + np.asarray(result['resid'])], ['Original', 'Estimated'], save_path=pic_path/f"estimated/pic_{inx}_with_residual.png")
        plot_series([time_series, np.asarray(result['trend']) + np.asarray(result['seasonal'])], ['Original', 'Estimated'], save_path=pic_path/f"estimated/pic_{inx}_without_residual.png")
        
    metrics_avg = {}
    for k,v in metrics.items():
        metrics_avg [k+'_mean'] = np.mean(v).round(4)    
    metrics.update(metrics_avg)

    return metrics, est_params, stl_grid, dataset, pic_path


if __name__ == '__main__':
    pass