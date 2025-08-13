from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataloader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


__all__ = [
    "Autoencoder_lightning",
    'train_with_lightning'
]


class Autoencoder_lightning(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        criterion: Optional[nn.Module] = None
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = criterion or nn.MSELoss()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, _ = batch
        reconstructed, _ = self.model(x)
        loss = self.criterion(reconstructed, x)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, _ = batch
        reconstructed, _ = self.model(x)
        loss = self.criterion(reconstructed, x)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=self.trainer.check_val_every_n_epoch // 2
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

            
@dataclass
class Training_config:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    model_save_path: Union[str, Path] = "scripts/experiment1"
    device: str = "cuda:1"
    experiment_name: str = "autoencoder_experiment1"
    log_dir: str = ".logs"
    
def train_with_lightning(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Training_config,
    logger: Optional[Any] = None
) -> tuple[pl.LightningModule, dict]:
    """
    Обучает модель с помощью PyTorch Lightning
    """

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.model_save_path,
        filename="best_model",
        monitor="val_loss",
        save_top_k=1,
        mode="min"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.05,
        patience=config.patience,
        verbose=False,
        mode="min"
    )

    # Lightning Module
    lightning_module = Autoencoder_lightning(
        model=model,
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="auto",
        devices="cuda: 1",
        log_every_n_steps=50,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        check_val_every_n_epoch=config.check_val_every_n_epoch or 1,
        enable_progress_bar=True
    )

    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    # Загружаем лучшую модель
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        lightning_module = lightning_module.load_from_checkpoint(best_model_path, model=model)

    # Собираем историю из логов
    history = {
        "train_loss": [x["train_loss"].item() for x in trainer.callback_metrics if "train_loss" in x],
        "val_loss": [x["val_loss"].item() for x in trainer.callback_metrics if "val_loss" in x],
        "lr": [pg["lr"] for pg in trainer.lr_scheduler_configs]
    }

    return lightning_module, history