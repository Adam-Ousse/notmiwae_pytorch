"""
Trainer class for not-MIWAE and MIWAE models with TensorBoard logging.

DataLoaders should return batches of (x_filled, mask) or (x_filled, mask, x_original)
where:
- x_filled: Data with missing values filled (e.g., with 0)
- mask: Binary mask (1=observed, 0=missing)
- x_original: Original complete data (optional, for evaluation)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from typing import Optional, Dict
from pathlib import Path
import datetime


class Trainer:
    """
    Trainer for not-MIWAE and MIWAE models.
    
    Args:
        model: NotMIWAE or MIWAE model
        lr: Learning rate
        device: Device ('cuda' or 'cpu')
        log_dir: Directory for TensorBoard logs
        checkpoint_dir: Directory for model checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: Optional[str] = None,
        log_dir: str = './runs',
        checkpoint_dir: str = './checkpoints'
    ):
        self.model = model
        self.lr = lr
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
            
        # Create directories
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = model.__class__.__name__
        
        self.log_dir = Path(log_dir) / f"{model_name}_{timestamp}"
        self.checkpoint_dir = Path(checkpoint_dir)
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self._graph_logged = False
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
    
    def log_model_graph(self, x: torch.Tensor, s: torch.Tensor):
        """Log model graph to TensorBoard."""
        if self._graph_logged:
            return
            
        try:
            class ModelWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                def forward(self, inputs):
                    return self.model(inputs[0], inputs[1])['loss']
            
            wrapper = ModelWrapper(self.model)
            wrapper.eval()
            self.writer.add_graph(wrapper, [(x[:2].to(self.device), s[:2].to(self.device))])
            self._graph_logged = True
            print(f"Model graph logged to TensorBoard: {self.log_dir}")
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")
            self._graph_logged = True
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        metrics = {'loss': 0.0, 'elbo': 0.0}
        has_missing_model = hasattr(self.model, 'missing_model')
        if has_missing_model:
            metrics['log_p_s_given_x'] = 0.0
        
        for batch in train_loader:
            x, s = batch[0].to(self.device), batch[1].to(self.device)
            
            if not self._graph_logged:
                self.log_model_graph(x, s)
            
            self.optimizer.zero_grad()
            output = self.model(x, s)
            loss = output['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            metrics['loss'] += loss.item()
            metrics['elbo'] += output['elbo'].item()
            if has_missing_model:
                metrics['log_p_s_given_x'] += output['log_p_s_given_x'].item()
            
            self.global_step += 1
            
        for key in metrics:
            metrics[key] /= len(train_loader)
        return metrics
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        metrics = {'loss': 0.0, 'elbo': 0.0}
        has_missing_model = hasattr(self.model, 'missing_model')
        if has_missing_model:
            metrics['log_p_s_given_x'] = 0.0
        
        for batch in val_loader:
            x, s = batch[0].to(self.device), batch[1].to(self.device)
            output = self.model(x, s)
            
            metrics['loss'] += output['loss'].item()
            metrics['elbo'] += output['elbo'].item()
            if has_missing_model:
                metrics['log_p_s_given_x'] += output['log_p_s_given_x'].item()
        
        for key in metrics:
            metrics[key] /= len(val_loader)
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        n_epochs: int = 100,
        log_interval: int = 10,
        save_best: bool = True,
        early_stopping_patience: int = 20,
        checkpoint_name: str = 'best_model.pt'
    ) -> Dict[str, list]:
        """
        Full training loop.
        
        Args:
            train_loader: DataLoader returning (x_filled, mask) or (x_filled, mask, x_original)
            val_loader: Optional validation DataLoader
            n_epochs: Number of epochs
            log_interval: Print interval
            save_best: Save best model
            early_stopping_patience: Epochs before early stopping
            checkpoint_name: Checkpoint filename
        """
        history = {'train_loss': [], 'train_elbo': [], 'val_loss': [], 'val_elbo': []}
        start_time = time.time()
        
        for epoch in range(n_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_elbo'].append(train_metrics['elbo'])
            
            self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Train/ELBO', train_metrics['elbo'], epoch)
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                history['val_elbo'].append(val_metrics['elbo'])
                
                self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('Val/ELBO', val_metrics['elbo'], epoch)
                
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.epochs_without_improvement = 0
                    if save_best:
                        self.save_checkpoint(checkpoint_name)
                else:
                    self.epochs_without_improvement += 1
                    
                if self.epochs_without_improvement >= early_stopping_patience:
                    print(f"\nEarly stopping after {epoch + 1} epochs")
                    break
            
            if (epoch + 1) % log_interval == 0:
                msg = f"Epoch {epoch + 1}/{n_epochs} - Loss: {train_metrics['loss']:.4f}, ELBO: {train_metrics['elbo']:.4f}"
                if val_loader:
                    msg += f" | Val Loss: {val_metrics['loss']:.4f}"
                print(msg)
        
        print(f"\nTraining completed in {time.time() - start_time:.2f}s")
        self.writer.close()
        return history
    
    def save_checkpoint(self, name: str = 'checkpoint.pt'):
        """Save model checkpoint."""
        path = self.checkpoint_dir / name
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }, path)
        
    def load_checkpoint(self, name: str = 'checkpoint.pt'):
        """Load model checkpoint."""
        path = self.checkpoint_dir / name
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        return self
