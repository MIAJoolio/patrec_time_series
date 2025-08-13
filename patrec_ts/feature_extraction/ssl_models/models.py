from typing import Literal, List, Dict, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
from .base import SLD_Basic_AE
from .core import SLD_Linear_encoder, SLD_Linear_decoder, SLD_LSTM_encoder, SLD_LSTM_decoder

__all__ = [
    'SLD_Linear_AE',
    'SLD_LSTM_AE'
]

class SLD_Linear_AE(SLD_Basic_AE):
    def __init__(self, encoder: SLD_Linear_encoder, decoder: SLD_Linear_decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @property
    def latent_dim(self) -> int:
        return self.encoder.latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten_data(x)
        latent, _ = self.encoder(x)
        # Self-learning step
        improved_latent = self.decoder.self_learn(latent)
        reconstructed = self.decoder(improved_latent)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten_data(x)
        latent, _ = self.encoder(x)
        return latent

    def flatten_data(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class SLD_LSTM_AE(SLD_Basic_AE):
    def __init__(self, encoder: SLD_LSTM_encoder, decoder: SLD_LSTM_decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @property
    def latent_dim(self) -> int:
        return self.encoder.latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, _ = self.encoder(x)
        # Self-learning step
        improved_latent = self.decoder.self_learn(latent)
        reconstructed = self.decoder(improved_latent)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent, _ = self.encoder(x)
        return latent