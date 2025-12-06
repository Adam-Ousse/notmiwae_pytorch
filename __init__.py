"""
PyTorch Implementation of not-MIWAE

not-MIWAE: Deep Generative Modelling with Missing not at Random Data
Paper: https://arxiv.org/abs/2006.12871
Authors: Niels Bruun Ipsen, Pierre-Alexandre Mattei, Jes Frellsen

This package provides:
- NotMIWAE: The full not-MIWAE model with explicit missing process modeling
- MIWAE: Standard MIWAE for comparison (without missing process)
- Custom missing process support via BaseMissingProcess
- Trainer with TensorBoard logging
- Utility functions for evaluation and visualization

Example usage:
    
    from notmiwae_pytorch import NotMIWAE, MIWAE, Trainer, BaseMissingProcess
    from notmiwae_pytorch.utils import imputation_rmse, set_seed
    
    # Set seed
    set_seed(42)
    
    # Create model with built-in missing process
    model = NotMIWAE(
        input_dim=10,
        latent_dim=5,
        hidden_dim=128,
        n_samples=20,
        missing_process='selfmasking_known_signs'
    )
    
    # Or create model with custom missing process
    class MyMissingProcess(BaseMissingProcess):
        def __init__(self, input_dim, **kwargs):
            super().__init__(input_dim, **kwargs)
            self.threshold = nn.Parameter(torch.zeros(input_dim))
            
        def forward(self, x):
            return x - self.threshold
            
        def interpret(self, verbose=True):
            return {'thresholds': self.threshold.detach().cpu().numpy()}
    
    model = NotMIWAE(
        input_dim=10,
        missing_process=MyMissingProcess(10)
    )
    
    # Train (expects DataLoader returning (x_filled, mask, x_original))
    trainer = Trainer(model, lr=1e-3)
    history = trainer.train(train_loader, val_loader, n_epochs=100)
    
    # Impute
    x_imputed = model.impute(x_filled, mask, n_samples=1000)
"""

from .models import NotMIWAE, MIWAE, BaseMissingProcess
from .trainer import Trainer
from . import utils

__version__ = "1.0.0"

__all__ = [
    'NotMIWAE',
    'MIWAE',
    'BaseMissingProcess',
    'Trainer',
    'utils'
]
