from typing import Any, Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class SLD_Basic_encoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: Any) -> Tuple[Any, Any]:
        """
        Forward pass through encoder.
        Returns: (latent, hidden_state)
        """
        pass

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Latent space dimension."""
        pass

class SLD_Basic_decoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, latent: Any, *args, **kwargs) -> Any:
        """
        Forward pass through decoder.
        Args:
            latent: latent representation
        Returns: reconstructed data
        """
        pass
    
    @abstractmethod
    def self_learn(self, *args, **kwargs) -> Any:
        """
        Self-learning mechanism for the decoder
        """
        pass

class SLD_Basic_AE(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward pass through autoencoder"""
        pass

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Returns latent space dimension"""
        pass
    
    @abstractmethod
    def encode(self, x: Any):
        """Returns latent representation for input data"""
        pass    

    def flatten_data(self, x: Any) -> Any:
        """Flatten time series data if needed"""
        return x