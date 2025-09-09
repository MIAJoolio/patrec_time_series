### code тут код 
from pathlib import Path
import numpy as np
import pandas as pd

from aeon.clustering import (
    TimeSeriesKMeans
    , TimeSeriesKShape
    , TimeSeriesCLARA
    , KSpectralCentroid
    )

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score

from patrec_ts.utils import download_UCR_dataset
from patrec_ts.utils.files_helper import JSON_Handler
from patrec_ts.utils.loaders import StratifiedTSLoader


# def apply_aeon_kmeans(X_train, X_test, config):
#     """
#     В конфиге должны быть 
#         n_clusters
#     """
    
#     model = TimeSeriesKMeans(**config)
#     model.fit(X_train)
    
#     y_pred = model.predict(X_test)
    
#     visualize_clusters()
#     save_y_predults()
    
#     return model, y_pred
    
    
def visualize_clusters():
    return 'visualize done!'


def save_y_predults():
    return 'saving y_predults done!'


def evaluate_benchmark():
    
    ds_name = 'BasicMotions' 
    # ds_name = 'MelbournePedestrian'

    download_UCR_dataset(ds_name)

    ds_path = Path('data/UCR_benchmark/datasets') / ds_name
    JSON_Handler(
        filepath=ds_path / 'ds.json',
        data={
            'data': np.load(ds_path / 'raw_features.npy'),
            'labels': pd.read_csv(ds_path / 'labels.csv', header=None).iloc[:, 0].values
        },
        mode='write'
    )
    
    data = StratifiedTSLoader(
        filepath=ds_path / 'ds.json',
    )

    config =  {
        'n_clusters': 10, 
        'distance':'euclidean' # dtw
    }
    
    model_encoder = LabelEncoder().fit(data.splits['train']['labels'])
    labels = model_encoder.transform(data.splits['train']['labels'])

    cl_model, y_pred = apply_aeon_kmeans(data.splits['train']['data'], data.splits['train']['data'], config)

    print(
        'TRAIN: RI: ', rand_score(labels, y_pred),
        ' ARI: ', adjusted_rand_score(labels, y_pred),
        ' NMI: ', normalized_mutual_info_score(labels, y_pred),
    )

    labels_test = model_encoder.transform(data.splits['test']['labels'])
    y_pred_test = cl_model.predict(data.splits['test']['data'])

    print(
        'TEST: RI: ', rand_score(labels_test, y_pred_test),
        ' ARI: ', adjusted_rand_score(labels_test, y_pred_test),
        ' NMI: ', normalized_mutual_info_score(labels_test, y_pred_test),
    )

    print(11)

if __name__ == '__main__':
    
    evaluate_benchmark()