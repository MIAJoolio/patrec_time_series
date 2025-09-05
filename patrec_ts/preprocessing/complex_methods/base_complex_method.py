from abc import ABC, abstractmethod

import numpy as np

from patrec_ts.feature_extraction.fe_classes import DecompositionResult
from patrec_ts.preprocessing.base_preprocessor import BasePreprocessor


class BaseComplexPreprocessor(BasePreprocessor, ABC):
    def __init__(
        self,
        trend_cls: type[BasePreprocessor],
        season_cls: type[BasePreprocessor],
        peak_cls: type[BasePreprocessor],
        noise_cls: type[BasePreprocessor],
    ):
        self._trend_cls = trend_cls
        self._trend_models: list[list[BasePreprocessor]] = []

        self._season_cls = season_cls
        self._season_models: list[list[BasePreprocessor]] = []

        self._peak_cls = peak_cls
        self._peak_models: list[list[BasePreprocessor]] = []

        self._noise_cls = noise_cls
        self._noise_models: list[list[BasePreprocessor]] = []

    def fit(self, data: np.ndarray, y: np.ndarray | None = None) -> BasePreprocessor:
        self._trend_models = np.zeros((data.shape[0], data.shape[1])).tolist()
        self._season_models = np.zeros((data.shape[0], data.shape[1])).tolist()
        self._peak_models = np.zeros((data.shape[0], data.shape[1])).tolist()
        self._noise_models = np.zeros((data.shape[0], data.shape[1])).tolist()

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """(, , ) -> (, , )"""
        output = np.zeros_like(data)

        for ds_index in range(data.shape[0]):
            for feature_index in range(data[ds_index].shape[0]):
                out_ = self.decompose(data, ds_index, feature_index)  # (N, ) -> (N, 4)

                out_ = self.process(out_)

                out_ = self.merge(out_)

                output[ds_index, feature_index] = out_

        return np.array(output)

    def decompose(self, data: np.ndarray, ds_index: int, feature_index: int) -> np.ndarray:
        el = data[ds_index][feature_index]

        self._trend_models[ds_index][feature_index] = self._trend_cls().fit(el)
        trend, _ = self._trend_models[ds_index][feature_index].transform(el)

        self._season_models[ds_index][feature_index] = self._season_cls().fit(el)
        season, _ = self._season_models[ds_index][feature_index].transform(el)

        self._peak_models[ds_index][feature_index] = self._peak_cls().fit(el)
        peak, _ = self._peak_models[ds_index][feature_index].transform(el)

        self._noise_models[ds_index][feature_index] = self._noise_cls().fit(el)
        noise, _ = self._noise_models[ds_index][feature_index].transform(el)

        return np.concatenate([trend, season, peak, noise], axis=0).T

    def process(self, data):
        return data

    def merge(self, data):
        return data
