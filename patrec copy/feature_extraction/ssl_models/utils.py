from typing import Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from .base import SLD_Basic_AE


@dataclass
class SLD_Training_config:
    epochs_num: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    model_save_path: Union[str, Path] = "scripts/sld_experiment"
    device: str = "cuda:0"
    experiment_name: str = "sld_autoencoder_experiment"
    log_dir: str = ".logs"
    delta: float = 0.01
    decoder_learning_rate: float = 1e-4  # Separate learning rate for decoder


@dataclass
class SLD_Training_config:
    epochs_num: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    model_save_path: Union[str, Path] = "scripts/sld_experiment"
    device: str = "cuda:0"
    experiment_name: str = "sld_autoencoder_experiment"
    log_dir: str = ".logs"
    delta: float = 0.01
    decoder_learning_rate: float = 1e-4  # Separate learning rate for decoder

class SLD_AE_trainer:
    def __init__(self, model: SLD_Basic_AE,
                 model_config: SLD_Training_config,
                 train_data: DataLoader,
                 val_data: DataLoader,
                 test_data: DataLoader = None,
                 logger: Optional[Any] = None):
        
        self.model = model
        self.config = model_config
        self.train_loader = train_data
        self.val_loader = val_data
        self.test_data = test_data
        self.logger = logger
        
        # Setup device
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).float()
        
        # Create save directory
        self.model_save_path = Path(self.config.model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizers - separate for encoder and decoder
        self.encoder_optimizer = optim.Adam(
            self.model.encoder.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        self.decoder_optimizer = optim.Adam(
            list(self.model.decoder.recon_net.parameters()) + 
            list(self.model.decoder.self_learn_net.parameters()),
            lr=self.config.decoder_learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.encoder_optimizer, 
            mode='min', 
            patience=self.config.patience // 2
        )
        
        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {'epoch_num': [], 'train_loss': [], 'val_loss': [], 'lr': []}

    def train_loop(self, visualize_latent_space: bool = False):
        for epoch in range(self.config.epochs_num):
            train_loss = self._train_one_epoch()
            val_loss = self._validate_one_epoch()
            
            # Update history
            self.history['epoch_num'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(self.encoder_optimizer.param_groups[0]['lr'])
            
            # Step scheduler
            self.scheduler.step(val_loss)
            
            # Early stopping check
            early_stop = self._handle_early_stopping(val_loss)
            
            if self.logger:
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs_num} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {self.encoder_optimizer.param_groups[0]['lr']:.2e}"
                )
            
            if early_stop:
                if self.logger:
                    self.logger.info(f'Early stopping triggered at epoch {epoch + 1}')
                break
        
        self._save_results()
        
        if visualize_latent_space:
            self._visualize_latent_space()

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (x, _) in enumerate(self.train_loader):
            x = x.to(self.device).float()
            
            # Zero gradients
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            
            # Forward pass
            latent, _ = self.model.encoder(x)
            improved_latent = self.model.decoder.self_learn(latent)
            reconstructed = self.model.decoder(improved_latent)
            
            # Compute loss
            loss = self._compute_loss(reconstructed, x)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            
            total_loss += loss.item() * x.size(0)
        
        return total_loss / len(self.train_loader.dataset)

    def _validate_one_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(self.val_loader):
                x = x.to(self.device).float()
                latent, _ = self.model.encoder(x)
                improved_latent = self.model.decoder.self_learn(latent)
                reconstructed = self.model.decoder(improved_latent)
                loss = self._compute_loss(reconstructed, x)
                total_loss += loss.item() * x.size(0)
        
        return total_loss / len(self.val_loader.dataset)

    def _compute_loss(self, reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if reconstructed.shape == target.shape:
            return self.criterion(reconstructed.float(), target.float())

        batch_size = target.shape[0]

        if len(reconstructed.shape) == 2 and len(target.shape) == 3:
            seq_len, input_dim = target.shape[1], target.shape[2]
            reconstructed = reconstructed.view(batch_size, seq_len, input_dim)
        elif len(reconstructed.shape) == 3 and len(target.shape) == 2:
            seq_len, input_dim = reconstructed.shape[1], reconstructed.shape[2]
            target = target.view(batch_size, seq_len, input_dim)
        else:
            raise ValueError(f"Shapes not compatible: {reconstructed.shape}, {target.shape}")

        return self.criterion(reconstructed.float(), target.float())

    def _handle_early_stopping(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss - self.config.delta:
            self.best_val_loss = val_loss
            self.best_epoch = len(self.history['val_loss'])
            self.patience_counter = 0
            torch.save(self.model.state_dict(), self.model_save_path / 'best_model.pth')
            if self.logger:
                self.logger.info(f"New best model at epoch {self.best_epoch} with val loss {val_loss:.4f}")
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.config.patience

    def _save_results(self):
        # Save training history
        pd.DataFrame(self.history).to_csv(self.model_save_path / 'training_history.csv', index=False)
        
        # Save best model
        self.model.load_state_dict(torch.load(self.model_save_path / 'best_model.pth'))
        
        if self.logger:
            self.logger.info(
                f"Training completed. Best model at epoch {self.best_epoch} "
                f"with val loss {self.best_val_loss:.4f}"
            )

    def _visualize_latent_space(self):
        self.model.eval()
        latents = []
        labels = []
        
        with torch.no_grad():
            for x, y in self.train_loader:
                x = x.to(self.device).float()
                z = self.model.encode(x)
                latents.append(z.cpu().numpy())
                labels.append(y.numpy())
        
        latents = np.vstack(latents)
        labels = np.hstack(labels)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        latents_2d = tsne.fit_transform(latents)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='tab10')
        plt.colorbar(scatter)
        plt.title("Latent Space Visualization (t-SNE)")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True)
        plt.savefig(self.model_save_path / 'latent_space.png')
        plt.close()