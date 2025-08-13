from typing import Literal, List, Dict, Any, Optional, Union, Tuple

import torch
import torch.nn as nn

from patrec.feature_extraction.autoencoders.base import Basic_decoder, Basic_encoder


class Linear_encoder(Basic_encoder):
    def __init__(self, input_size: int, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.flatten = lambda x: x.view(x.shape[0], -1)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self._latent_dim = latent_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        x = self.flatten(x)
        latent = self.net(x)
        return latent


class Linear_decoder(Basic_decoder):
    def __init__(self, latent_dim: int, output_size: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )
        self.output_size = output_size

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        reconstructed = self.net(latent)
        # [B, output_size]
        return reconstructed.view(latent.shape[0], -1)  


class LSTM_encoder(Basic_encoder):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self._latent_dim = latent_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs, (h_n, c_n) = self.lstm(x)
        # Используем последний слой
        latent = self.to_latent(h_n[-1])
        # Возвращаем скрытое состояние как hidden_state  
        return latent, h_n  
    
    
class LSTM_decoder(Basic_decoder):
    def __init__(
        self, latent_dim: int, hidden_dim: int, output_dim: int, seq_len: int, num_layers: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.num_layers = num_layers

        # Проекция латента в начальное скрытое состояние
        self.from_latent = nn.Linear(latent_dim, hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,     # размерность входа на каждом шаге
            hidden_size=output_dim,   # размерность выхода на каждом шаге
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.shape[0]

        # Инициализируем начальное скрытое состояние
        hidden = self.from_latent(latent).unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = torch.zeros_like(hidden)

        # Создаем нулевой вход для декодера
        decoder_input = torch.zeros(batch_size, self.seq_len, self.lstm.input_size, device=latent.device)

        # Декодируем
        reconstructed, _ = self.lstm(decoder_input, (hidden, cell))

        return reconstructed