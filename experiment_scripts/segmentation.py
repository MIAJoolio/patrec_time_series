import os 
os.chdir('/root/patrec_time_series/')
from pathlib import Path

import numpy as np
from patrec.utils import plot_series, plot_series_grid
from patrec.utils.files_helper import Format1_loader



if __name__ == '__main__':
    
    DATASET_PATH = 'data/Synthetic_data/17_06_samples/lin_sin/sample1_10_1_0.json'
    
    loader = Format1_loader()
    sample1 = loader.load_data(DATASET_PATH)
    