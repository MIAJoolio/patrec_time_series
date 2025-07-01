import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("ts2vec/"))

from aeon.datasets import load_classification

from ts2vec import TS2Vec

from src.utils import *


def get_ts2vec_feat(dataset_name='ItalyPowerDemand', train_ratio=0.8, config_file='configs/fe_default/ts2vec_config.yaml', model_save_path='experiments/pipe2/ts2vec_model.pt'):

    # Load the dataset
    train_data, test_data, train_labels, test_labels, _ = load_UCR(dataset_name, train_ratio=train_ratio)
    # print(train_data.shape, test_data.shape, test_data[0])
    train_data = train_data.swapaxes(1, 2)
    test_data = test_data.swapaxes(1, 2)
    # print(train_data.shape, test_data.shape, test_data[0])
    print(pd.Series(train_labels.reshape(-1,)).value_counts())
    print(pd.Series(test_labels.reshape(-1,)).value_counts())
    
    # Train a TS2Vec model
    config = load_config_file(config_file)
    model_params = config['model_params'] 
    
    model = TS2Vec(
        input_dims=train_data.shape[-1],
        device=1,
        output_dims=model_params['output_dims'],
        hidden_dims=model_params['hidden_dims'],
        depth=model_params['depth'],
        lr=model_params['lr']
    )

    loss_log = model.fit(
        train_data,
        verbose=True,
        n_epochs = model_params['n_epochs'] 
    )

    encode_params = config['encode_params'] 
    if encode_params['encoding_window']:
        # Compute instance-level representations for test set
        test_repr = model.encode(test_data, encoding_window='full_series')  # n_instances x output_dims
    else:
        # Sliding inference for test set
        test_repr = model.encode(
            test_data,
            causal=True,
            sliding_length=encode_params['sliding_length'],
            sliding_padding=encode_params['sliding_padding']
        )  
        
    model.save(model_save_path) 
    
    return test_repr, test_labels

def main():

    data, lbls = get_ts2vec_feat()
    print(data.shape)

if __name__ == '__main__':
    main()