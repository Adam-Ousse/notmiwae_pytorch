"""
PyTorch Implementation of not-MIWAE

not-MIWAE: Deep Generative Modelling with Missing not at Random Data
Paper: https://arxiv.org/abs/2006.12871
Authors: Niels Bruun Ipsen, Pierre-Alexandre Mattei, Jes Frellsen

This package provides:
- NotMIWAE: The full not-MIWAE model with explicit missing process modeling
- MIWAE: Standard MIWAE for comparison (without missing process)
- Trainer with TensorBoard logging
- Utility functions for evaluation and visualization

Example usage:
    
    from notmiwae_pytorch import NotMIWAE, MIWAE, Trainer
    from notmiwae_pytorch.utils import imputation_rmse, set_seed
    
    # Set seed
    set_seed(42)
    
    # Create model
    model = NotMIWAE(
        input_dim=10,
        latent_dim=5,
        hidden_dim=128,
        n_samples=20,
        missing_process='selfmasking_known'
    )
    
    # Train (expects DataLoader returning (x_filled, mask, x_original))
    trainer = Trainer(model, lr=1e-3)
    history = trainer.train(train_loader, val_loader, n_epochs=100)
    
    # Impute
    x_imputed = model.impute(x_filled, mask, n_samples=1000)
"""

from .models import NotMIWAE, MIWAE
from .trainer import Trainer
from . import utils

__version__ = "1.0.0"

__all__ = [
    'NotMIWAE',
    'MIWAE',
    'Trainer',
    'utils'
]
