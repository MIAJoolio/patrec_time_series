from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BasePreprocessor(ABC, BaseEstimator, TransformerMixin):
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def fit(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()