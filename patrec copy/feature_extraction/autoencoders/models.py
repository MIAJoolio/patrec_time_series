from typing import Literal, List, Dict, Any, Optional, Union, Tuple

import torch
import torch.nn as nn

from patrec.feature_extraction.autoencoders.base import Basic_AE
from patrec.feature_extraction.autoencoders.core import Linear_encoder, Linear_decoder, LSTM_encoder, LSTM_decoder


__all__ = [
    'Linear_AE',
    'LSTM_AE'
]


class Linear_AE(Basic_AE):
    def __init__(self, encoder: Linear_encoder, decoder: Linear_decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @property
    def latent_dim(self) -> int:
        return self.encoder.latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten_data(x)  # <-- здесь происходит выравнивание
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten_data(x)
        latent = self.encoder(x)
        return latent

    def flatten_data(self, x: torch.Tensor) -> torch.Tensor:
        """
        Переопределяем метод базового класса.
        Выравнивает [B, T, F] -> [B, T*F]
        """
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class LSTM_AE(Basic_AE):
    def __init__(self, encoder: LSTM_encoder, decoder: LSTM_decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._seq_len = decoder.seq_len

    @property
    def latent_dim(self) -> int:
        return self.encoder.latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, _ = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent, _ = self.encoder(x)
        return latent