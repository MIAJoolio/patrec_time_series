import os
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


def split_train_test(X:np.ndarray, y:np.ndarray, split_ratio:float=0.7, random_state:int=None):
    
    # Инициализируем сплиттер с указанием доли train и random_state
    spliter = StratifiedShuffleSplit(n_splits=1, test_size=1 - split_ratio, random_state=random_state)

    # if y.dtype.kind in ['U', 'S', 'O']:  # U - Unicode, S - bytes, O - object
    # print("Data is not numeric, LabelEncoder is used")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y.ravel())

    print("Old Classes:", le.classes_)
    print("New Classes:", np.unique(y_encoded))

    # Получаем индексы train и test
    for train_index, test_index in spliter.split(X, y_encoded):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    
    return X_train, X_test, y_train, y_test


def prepare_to_ts2vec(dataset_name:str, split_ratio:int=0.7, random_state:int=None):
    """
    Функция загрузки и подготовки данных для обучения ts2vec. Данные сохраняются в data/UCR_benchmark/datasets
    """
    
    dir_path = Path('data/UCR_benchmark/datasets')
    dt_path = dir_path / dataset_name
    
    # TODO для неравных по длине датасетов
    
    if not dt_path.is_dir():
        raise ValueError("Incorrect dataset name! Not in UCR benchmark!")
    
    X, y = np.load(dt_path/'raw_features.npy'), pd.read_csv(dt_path/'labels.csv', header=None).values

    MAIN_PATH = Path('src/feature_extraction/ts2vec')
    os.makedirs(MAIN_PATH / f"datasets/UCR/{dataset_name}", exist_ok=True)

    # Преобразуем X из (n_samples, 1, n_timesteps) в (n_samples, n_timesteps)
    X_flat = X.squeeze(axis=1)  # shape: (n_samples, n_timesteps)

    X_train, X_test, y_train, y_test = split_train_test(X_flat, y, split_ratio, random_state)   

    # Соединяем метки и признаки
    # shape: (n_train, n_timesteps + 1)
    train_data = np.hstack((y_train.reshape(-1, 1), X_train))  
    # shape: (n_test, n_timesteps + 1)
    test_data = np.hstack((y_test.reshape(-1, 1), X_test))     
    
    # Сохраняем файлы
    pd.DataFrame(train_data).to_csv(MAIN_PATH/f"datasets/UCR/{dataset_name}/{dataset_name}_TRAIN.tsv", sep='\t', index=False, header=False)
    pd.DataFrame(test_data).to_csv(MAIN_PATH/f"datasets/UCR/{dataset_name}/{dataset_name}_TEST.tsv", sep='\t', index=False, header=False)
    


def main():
    # prepare_to_ts2vec('DucksAndGeese')
    return None

if __name__ == '__main__':
    main() 