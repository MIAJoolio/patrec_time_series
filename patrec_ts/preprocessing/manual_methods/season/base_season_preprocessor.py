from abc import ABC, abstractmethod

import numpy as np

from patrec_ts.feature_extraction.fe_classes import DecompositionResult
from patrec_ts.preprocessing.base_preprocessor import BasePreprocessor


class BaseSeasonPreprocessor(BasePreprocessor, ABC):
    def _check_data_input(self, data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray) or data.ndim != 1:
            raise ValueError(f"data must be a 1D numpy array. Got shape {data.shape}, type {type(data)}")

        if data.shape[0] < 2:
            raise ValueError(f"data must contain at least 2 points for linear regression. Got {data.shape[0]}")

    def fit(self, data: np.ndarray, y: np.ndarray | None = None) -> BasePreprocessor:
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        output = []
        for ds_index in range(data.shape[0]):
            features = []
            for feature_index in range(data[ds_index].shape[0]):
                res = self.decompose(data[ds_index][feature_index])
                features.append(res.component)

            output.append(np.array(features))

        return np.array(output)

    @abstractmethod
    def decompose(self, data: np.ndarray, **kwargs) -> DecompositionResult:
        raise NotImplementedError()
