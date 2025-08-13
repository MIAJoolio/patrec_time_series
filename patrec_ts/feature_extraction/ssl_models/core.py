from typing import Literal, List, Dict, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
from .base import SLD_Basic_decoder, SLD_Basic_encoder

class SLD_Linear_encoder(SLD_Basic_encoder):
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
        return latent, None

class SLD_Linear_decoder(SLD_Basic_decoder):
    def __init__(self, latent_dim: int, output_size: int, hidden_dim: int = 128):
        super().__init__()
        # Main reconstruction network
        self.recon_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )
        
        # Self-learning network
        self.self_learn_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.output_size = output_size

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        reconstructed = self.recon_net(latent)
        return reconstructed.view(latent.shape[0], -1)
    
    def self_learn(self, latent: torch.Tensor) -> torch.Tensor:
        """Self-learning mechanism - improves the latent representation"""
        return self.self_learn_net(latent)

class SLD_LSTM_encoder(SLD_Basic_encoder):
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
        latent = self.to_latent(h_n[-1])
        return latent, h_n

    
class SLD_LSTM_decoder(SLD_Basic_decoder):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, seq_len: int, num_layers: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Reconstruction components (renamed to recon_net)
        self.recon_net = nn.ModuleDict({
            'from_latent': nn.Linear(latent_dim, hidden_dim),
            'lstm': nn.LSTM(
                input_size=hidden_dim,
                hidden_size=output_dim,
                num_layers=num_layers,
                batch_first=True
            )
        })
        
        # Self-learning components (renamed to self_learn_net)
        self.self_learn_net = nn.LSTM(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.shape[0]
        hidden = self.recon_net['from_latent'](latent).unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = torch.zeros_like(hidden)
        decoder_input = torch.zeros(batch_size, self.seq_len, self.recon_net['lstm'].input_size, device=latent.device)
        reconstructed, _ = self.recon_net['lstm'](decoder_input, (hidden, cell))
        return reconstructed
    
    def self_learn(self, latent: torch.Tensor) -> torch.Tensor:
        """Self-learning mechanism for LSTM decoder"""
        batch_size = latent.shape[0]
        latent = latent.unsqueeze(1)  # Add sequence dimension
        improved_latent, _ = self.self_learn_net(latent)
        return improved_latent.squeeze(1)