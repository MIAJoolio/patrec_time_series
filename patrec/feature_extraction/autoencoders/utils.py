from typing import Literal, List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.utils import setup_logger, Logger, plot_series_grid, plot_series
from src.feature_extraction.autoencoders.base import Basic_AE


__all__ = [
    'Training_config',
    'AE_trainer',
    'extract_latent_features',
    'visualize_all_latent_points'
]


@dataclass
class Training_config:
    epochs_num: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    model_save_path: Union[str, Path] = "scripts/experiment1"
    device: str = "cuda:1"
    experiment_name: str = "autoencoder_experiment1"
    log_dir: str = ".logs"
    delta: float = 0.01


class AE_trainer():
    
    def __init__(self, model:Basic_AE,
                 model_config:Training_config,
                 train_data:DataLoader,
                 val_data:DataLoader,
                 test_data:DataLoader=None,
                 logger:Optional[Logger] = None):
        
        # настройка модели и окружения 
        self.model = model
        self.config = self._load_config(model_config)   
        self.logger = logger or setup_logger(
            name=self.model_config['experiment_name'],
            log_dir=self.model_config['log_dir'],
            level='debug'
        )
        self.train_loader = train_data
        self.val_loader = val_data
        self.test_data = test_data 
        
    def _load_config(self, model_config):
        """
        Функция по установке параметров модели и окружения
        """
        self.model_config = model_config.__dict__
        self.device = torch.device(self.model_config['device'] if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).float()

        self.model_save_path = Path(self.model_config['model_save_path'])
        self.model_save_path.mkdir(parents=True, exist_ok=True)

    
    def train_loop(self, visualize_latent_space:bool=False):
        """
        Функция объединяющая обучение, валидацию, визуализацию и инференс модели автоэнкодера 
        """                
        # инициализация компонент
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.model_config['lr'], weight_decay=self.model_config['weight_decay'])
        
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=self.model_config['patience'] // 2)
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
         
        self.history = {'epoch_num':[], 'train_loss': [], 'val_loss': [], 'lr': []}
        
        for epoch in range(self.model_config['epochs_num']):
            # обучение и валидация
            train_loss = self._train_one_epoch()
            val_loss = self._validate_one_epoch()
            # сохраняем историю
            self.history['epoch_num'].append(epoch+1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)
            self.scheduler.step(val_loss)
            early_stop = self._handle_early_stopping(val_loss)

            # логирование эпохи
            self.logger.log_metric("train_loss", train_loss, step=epoch + 1)
            self.logger.log_metric("val_loss", val_loss, step=epoch + 1)
            self.logger.log_metric("learning_rate", current_lr, step=epoch + 1)
            self.logger.info(
                f"Epoch {epoch + 1}/{self.model_config['epochs_num']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            
            # Early Stopping
            if early_stop:
                self.logger.info(f'Early stopping triggered at epoch {epoch + 1}')
                break
            
        self._create_result_df()
        self._plot_loss_graphs()
        self._plot_predictions()
        
        if visualize_latent_space:
            self._plot_latent_space()
    
    def compute_loss(self, reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Выравнивает размерности и считает MSE Loss.
        Поддерживает следующие случаи:
            - reconstructed: [B, T*F], target: [B, T, F]
            - reconstructed: [B, T, F], target: [B, T*F]
            - reconstructed: [B, T, F], target: [B, T, F]
        """
        # Если формы совпадают — просто возвращаем loss
        if reconstructed.shape == target.shape:
            return self.criterion(reconstructed.float(), target.float())

        batch_size = target.shape[0]

        # Случай 1: reconstructed плоский, target последовательность
        if len(reconstructed.shape) == 2 and len(target.shape) == 3:
            seq_len, input_dim = target.shape[1], target.shape[2]
            reconstructed = reconstructed.view(batch_size, seq_len, input_dim)

        # Случай 2: target плоский, reconstructed последовательность
        elif len(reconstructed.shape) == 3 and len(target.shape) == 2:
            seq_len, input_dim = reconstructed.shape[1], reconstructed.shape[2]
            target = target.view(batch_size, seq_len, input_dim)

        # Случай 3: одна из форм не определена
        else:
            raise ValueError(f"Shapes not compatible: {reconstructed.shape}, {target.shape}")

        return self.criterion(reconstructed.float(), target.float())

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch_idx, (x, _) in enumerate(self.train_loader):
            x = x.to(self.device).float()
            self.optimizer.zero_grad()
            reconstructed = self.model(x)
            # print(len(reconstructed),reconstructed[0].shape, reconstructed[1].shape)
            loss = self.compute_loss(reconstructed, x)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
            # if batch_idx % 50 == 0:
            #     self.logger.debug(f"Batch {batch_idx} | Loss: {loss.item():.4f}")
        return total_loss / len(self.train_loader.dataset)
    
    def _validate_one_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(self.val_loader):
                x = x.to(self.device).float()
                reconstructed = self.model(x)
                loss = self.compute_loss(reconstructed, x)
                total_loss += loss.item() * x.size(0)
        return total_loss / len(self.val_loader.dataset)
    
    def _handle_early_stopping(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss - self.model_config.get("delta", 0):
            self.best_val_loss = val_loss
            self.best_epoch = len(self.history['val_loss'])
            self.patience_counter = 0
            torch.save(self.model.state_dict(), self.model_save_path / 'best_model.pth')
            self.logger.info(f"New best model at epoch {self.best_epoch + 1} with val loss {val_loss:.4f}")
        else:
            self.patience_counter += 1
        return self.patience_counter >= self.model_config['patience']
    
    def _create_result_df(self):
        pd.DataFrame(self.history).to_csv(self.model_save_path / 'training_history.csv', index=False)
        self.logger.save_metrics()
        self.logger.save_params()
        self.model.load_state_dict(torch.load(self.model_save_path / 'best_model.pth'))
        self.logger.info(
            f"Training completed. Best model at epoch {self.best_epoch + 1} "
            f"with val loss {self.best_val_loss:.4f}"
        )

    def _plot_loss_graphs(self):
        plot_series_grid(
            series_list=[self.history["train_loss"], self.history["val_loss"]],
            labels=["Train Loss", "Validation Loss"],
            x_series=np.arange(1, len(self.history["train_loss"]) + 1),
            plot_title="Training and Validation Loss",
            ylabel="Loss (MSE)",
            xlabel="Epoch",
            figsize=(12, 6),
            grid=True,
            layout="vertical",
            save_path=str(self.model_save_path / 'training_loss_plot.png')
        )

    def _plot_predictions(self):
        
        if self.test_data is None:
            data = self.val_loader
        else:
            data = self.test_data

        self.model.eval()            
        with torch.no_grad():
            for batch_idx, (x, cls_id) in enumerate(data):
                x = x.to(self.device).float()
                recon = self.model(x)

                plot_series(series_list=[x.squeeze(0).cpu().numpy(), recon[0].squeeze(0).cpu().numpy()],
                            labels=['original', 'reconstructed'],
                            plot_title='Autoencoder reconstruction',
                            save_path= self.model_save_path / f'pred/class_id_{str(cls_id.item())}_{batch_idx}.png'
                            )

    def _extract_latent_space(self, data:DataLoader):
        """
        Функция по извлечению латентного пространтсва 
        """    
        self.model.eval()
        latents = []
        labels = []
        with torch.no_grad():
            for x, y in data:
                x = x.to(self.device).float()
                z = self.model.encode(x)
                latents.append(z.cpu().numpy())
                labels.append(y.numpy())
        return np.vstack(latents), np.hstack(labels)
         
    def _plot_latent_space(self):
        """
        Функция для визуализации полученного латентного пространства
        """
        train_latents, train_labels = self._extract_latent_space(self.train_loader)
        val_latents, val_labels = self._extract_latent_space(self.val_loader)
        
        # Обучение t-SNE только на train
        tsne = TSNE(n_components=2, random_state=42, init='random')
        tsne.fit(train_latents)  # обучаем только на train

        # Проекция train и val
        train_2d = tsne.transform(train_latents)
        val_2d = tsne.transform(val_latents)

        # Визуализация
        plt.figure(figsize=(12, 8))

        # Точки из train
        scatter_train = plt.scatter(train_2d[:, 0], train_2d[:, 1], c=train_labels, cmap='tab10', s=60, label='Train', edgecolor='k', alpha=0.7)

        # Точки из val
        scatter_val = plt.scatter(val_2d[:, 0], val_2d[:, 1], c=val_labels,cmap='tab10', marker='X', s=100, label='Val', edgecolor='k')

        # Легенда
        unique_labels = np.unique(np.concatenate([train_labels, val_labels]))
        handles, _ = scatter_train.legend_elements(prop="colors", num=len(unique_labels))
        plt.legend(handles, unique_labels, title="Classes")

        plt.title("Latent Space Visualization (t-SNE) - Train and Val")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.model_save_path / 'latent_space_tsne.png')
        plt.close()
        
    def load_best_model(self, path:str=None):
        if path is None:
            model_path = self.model_save_path / 'best_model.pth'
        else:
            model_path = path
            
        if model_path.exists():
            return self.model.load_state_dict(torch.load(model_path))
        return ValueError('Model file is not found')
    
    def load_history(self):
        history_path = self.model_save_path / 'training_history.csv'
        if history_path.exists():
            return pd.read_csv(history_path)
        return ValueError('File is not found')