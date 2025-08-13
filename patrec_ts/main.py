from pathlib import Path

import numpy as np
import pandas as pd
from aeon.clustering import TimeSeriesCLARA
from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tslearn.clustering import TimeSeriesKMeans as TSLearnTimeSeriesKMeans

from patrec_ts.preprocessing.manual_methods.parametric.trend_linear_preprocessor import TrendLinearPreprocessor
from patrec_ts.preprocessing.manual_methods.season.dft_preprocessor import SeasonDFTPreprocessor
from patrec_ts.utils import download_UCR_dataset
from patrec_ts.utils.files_helper import JSON_Handler
from patrec_ts.utils.loaders import StratifiedTSLoader

ds_name = 'MelbournePedestrian'  # BasicMotions

download_UCR_dataset(ds_name, )

ds_path = Path('data/UCR_benchmark/datasets') / ds_name
JSON_Handler(
    filepath=ds_path / 'ds.json',
    data={
        'data': np.load(ds_path / 'raw_features.npy'),
        'labels': pd.read_csv(ds_path / 'labels.csv', header=None).iloc[:, 0].values
    },
    mode='write'
)

data: StratifiedTSLoader = StratifiedTSLoader(
    filepath=ds_path / 'ds.json',
)

pipeline = Pipeline(steps=[
    ('trend', TrendLinearPreprocessor()),
    # ('season', SeasonDFTPreprocessor()),
    ('cluster_method', TimeSeriesCLARA(n_clusters=11))
    # ('cluster_method', TSLearnTimeSeriesKMeans(n_clusters=4))
])

model_encoder = LabelEncoder().fit(data.splits['train']['labels'])
labels = model_encoder.transform(data.splits['train']['labels'])

pipeline.fit(data.splits['train']['data'], labels)
res = pipeline.predict(data.splits['train']['data'])

print(
    'TRAIN: RI: ', rand_score(labels, res),
    ' ARI: ', adjusted_rand_score(labels, res),
    ' NMI: ', normalized_mutual_info_score(labels, res),
)
labels_test = LabelEncoder().fit_transform(data.splits['test']['labels'])
res_test = pipeline.predict(data.splits['test']['data'])

print(
    'TEST: RI: ', rand_score(labels_test, res_test),
    ' ARI: ', adjusted_rand_score(labels_test, res_test),
    ' NMI: ', normalized_mutual_info_score(labels_test, res_test),
)

# DataSet MelbournePedestrian
# 1. Scores only with ClusterMethods (train + test)
# 2. Разные комбинация детрейна, десезенности, дешум и депики и тд
# 3. Это все в ml.flow (в jupyter ноутбуке)

print(1)