from typing import Any, Tuple

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Basic_encoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: Any) -> Tuple[Any, Any]:
        """
        Прямой проход через энкодер.
        Возвращает: (latent, hidden_state), где latent — латентное представление,
                    hidden_state — опционально, может быть использовано для некоторых архитектур.
        """
        pass

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Размерность латентного пространства."""
        pass


class Basic_decoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, latent: Any, *args, **kwargs) -> Any:
        """
        Прямой проход через декодер.
        Принимает: latent — латентное представление
        Возвращает: reconstructed — восстановленные данные
        """
        pass
    
class Basic_AE(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """
        Прямой проход через автоэнкодер.
        Возвращает восстановленные данные
        """
        pass

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Возвращает размерность латентного пространства."""
        pass
    
    @property
    @abstractmethod
    def encode(self, x: Any):
        """Возвращает латентное пространство для входных данных"""
        pass    

    def flatten_data(self, x: Any) -> Any:
        """
        Функция для выравнивания [batch_size, seq_len, input_size] в [batch_size, seq_len *  input_size] для автоэнкодеров, которые не могут работать с первоначальной формой ряда
        По умолчанию возвращает вход без изменений.
        """
        return x

