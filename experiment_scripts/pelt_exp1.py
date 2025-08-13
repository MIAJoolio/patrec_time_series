"""
min_size controls the minimum distance between change points; for instance, if min_size=10, all change points will be at least 10 samples apart.
jump controls the grid of possible change points; for instance, if jump=k, only changes at k, 2*k, 3*k,... are considered.

"l2" (по умолчанию) - для гауссовских данных
"rbf" - для нелинейных данных
"linear" - для линейных моделей
"normal" - для нормального распределения
"""
from pathlib import Path
from itertools import product
import shutil

import numpy as np

EXP_NAME = 'Pelt_sub_sample1'
EXP_PATH = Path(f'.temp/{EXP_NAME}/')

EXP_PATH.mkdir(exist_ok=True, parents=True)
PLOTS_PATH = EXP_PATH / "results/plots/"
PLOTS_PATH.mkdir(exist_ok=True, parents=True)
METRICS_PATH = EXP_PATH / "results/metrics.yaml"
PARAMS_PATH = EXP_PATH / "results/params.yaml"


p1 = [5, 9, 13, 15] #np.linspace(5, 20, 5, dtype=np.int64)
p2 = [2, 5, 10, 15] #np.linspace(5, 20, 5, dtype=np.int64)
p3 = ['l2', 'rbf', 'linear', 'normal']
p4 = np.linspace(1, 5, 3, dtype=np.int64)

grid = product(p1, p2, p3, p4)

for min_size, jump, model_name, pen in list(grid):

    init_params = {
        'model': model_name,
        'min_size': min_size,
        'jump': jump, 
    }

    pred_params = {
        'pen': pen
    }

    metrics = {
        'accuracy_score':[],
        'recall_score':[],
        'precision_score':[],
        'f1_score':[]
    }

    MATCH_ROUND = 3 
    
    params_dict = {}
    params_dict.update(init_params)
    params_dict.update(pred_params)
    params_dict.update({'match_round': MATCH_ROUND})
    save_config_file({k: int(v) if not isinstance(v, str) else v for k,v in params_dict.items()}, PARAMS_PATH)

    for t, (k, v) in enumerate(sample1_sub.items()):
        # get data
        signal = sample1_sub[k]['data']
        # init model
        model = PeltSegmentor(**init_params)
        # model = BinSegSegmentor(**init_params)
        # make pred 
        signal = signal.reshape(-1, 1)
        sample1_sub[k][f'pred'] = model.infer(signal, **pred_params)
        
        # visualize
        pic = model.plot(signal, sample1_sub[k]['peaks'])
        pic[0].savefig(PLOTS_PATH / f"pic{t}.png")
        plt.close()
        # metrics
        y_true, y_pred = match_change_points(sample1_sub[k]['peaks'], sample1_sub[k]['pred'], MATCH_ROUND)
        metrics['accuracy_score'].append(accuracy_score(y_true, y_pred))
        metrics['recall_score'].append(recall_score(y_true, y_pred))
        metrics['precision_score'].append(precision_score(y_true, y_pred))
        metrics['f1_score'].append(f1_score(y_true, y_pred))
    
    save_config_file({k:str([np.float64(sc).round(5).item() for sc in v]) for k,v in metrics.items()}, METRICS_PATH)

    logger = MLflowExperimentLogger(experiment_name=EXP_NAME, tags={'dataset':Path(DATA_PATH).stem})

    logger.log_experiment(
        pics_dir= PLOTS_PATH,
        params_path=PARAMS_PATH,
        metrics_path=METRICS_PATH
    )    

shutil.rmtree(EXP_PATH) 