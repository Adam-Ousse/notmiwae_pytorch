# PyTorch Implementation of not-MIWAE

This is a PyTorch implementation of the **not-MIWAE** model from the paper:

> **not-MIWAE: Deep Generative Modelling with Missing not at Random Data**  
> Niels Bruun Ipsen, Pierre-Alexandre Mattei, Jes Frellsen  
> ICLR 2021 | [Paper](https://arxiv.org/abs/2006.12871)
## Overview

The not-MIWAE extends the Missing Data Importance Weighted Autoencoder (MIWAE) by explicitly modeling the missing data mechanism. This allows it to handle **Missing Not At Random (MNAR)** data, where the probability of a value being missing depends on the value itself.

### Key Features

- **NotMIWAE Model**: Full implementation with encoder, decoder, and missing process networks
- **MIWAE Model**: Standard MIWAE for comparison (assumes MCAR)
- **Missing Process Interpretation**: Built-in tools to interpret learned missing mechanisms
- **Trainer**: Complete training loop with TensorBoard logging, early stopping, and checkpointing
- **Utilities**: Functions for evaluation and data preprocessing

## Installation

```bash
pip install notmiwae-pytorch
```

Or install from source:

```bash
git clone https://github.com/Adam-Ousse/notmiwae_pytorch.git
cd notmiwae_pytorch
pip install -e .
```

## Quick Start

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from notmiwae_pytorch import NotMIWAE, Trainer
from notmiwae_pytorch.utils import set_seed, imputation_rmse

# Set seed
set_seed(42)

# Prepare your data (x_filled has 0s for missing, mask is 1=observed, 0=missing)
# DataLoader should return (x_filled, mask, x_original) tuples
train_dataset = TensorDataset(x_filled, mask, x_original)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create model with feature names for interpretation
model = NotMIWAE(
    input_dim=10,
    latent_dim=5,
    hidden_dim=128,
    n_samples=20,
    missing_process='selfmasking_known_signs',
    feature_names=['feat1', 'feat2', ...]  # Optional
)

# Train
trainer = Trainer(model, lr=1e-3)
history = trainer.train(train_loader, val_loader, n_epochs=100)

# Interpret the learned missing process
model.interpret_missing_process()

# Evaluate imputation
rmse, x_imputed = imputation_rmse(model, x_original, x_filled, mask)
```

## Model Architecture

### not-MIWAE Objective

The not-MIWAE maximizes a lower bound on the joint log-likelihood:

$$\log p(x_o, s) \geq \mathbb{E}_{q(z|x_o)}\left[\log \frac{1}{K}\sum_{k=1}^{K} \frac{p(x_o|z_k) \cdot p(s|x_k) \cdot p(z_k)}{q(z_k|x_o)}\right]$$

where:
- $x_o$: observed values
- $s$: missingness indicator (1=observed, 0=missing)
- $z$: latent variables
- $K$: number of importance samples

### Missing Process Models

The model supports several missing mechanisms through `p(s|x)`. **The more prior knowledge you have about the missing mechanism in your data, the more accurate the imputations will be.** Choose the model that best matches your assumptions:

1. **`selfmasking`**: $\text{logit}(p(s_d=1|x)) = -W_d(x_d - b_d)$
2. **`selfmasking_known_signs`**: Same as above but with $W_d > 0$ (known direction)
   - Supports directional control via `signs` parameter:
     - `+1.0`: High values more likely to be missing
     - `-1.0`: Low values more likely to be missing
3. **`linear`**: Linear mapping from $x$ to logits
4. **`nonlinear`**: MLP mapping from $x$ to logits

### Custom Missing Process

You can create your own missing process by inheriting from `BaseMissingProcess`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from notmiwae_pytorch import NotMIWAE, BaseMissingProcess

class SinusoidalMissingProcess(BaseMissingProcess):
    """
    Custom missing process with sinusoidal pattern.
    Captures non-monotonic missingness where certain value ranges
    are more/less likely to be observed.
    """
    
    def __init__(self, input_dim, init_frequency=1.0, **kwargs):
        super().__init__(input_dim, **kwargs)
        
        # Learnable parameters
        self.amplitude = nn.Parameter(torch.ones(1, 1, input_dim))
        self.frequency = nn.Parameter(torch.full((1, 1, input_dim), init_frequency))
        self.phase = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, input_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for p(s=1|x)."""
        amp = F.softplus(self.amplitude)
        freq = F.softplus(self.frequency)
        return amp * torch.sin(freq * x + self.phase) + self.bias
    
    def interpret(self, verbose=True):
        """Interpret learned parameters."""
        results = {
            'amplitude': F.softplus(self.amplitude).detach().cpu().numpy(),
            'frequency': F.softplus(self.frequency).detach().cpu().numpy(),
        }
        if verbose:
            print(f"Learned frequency: {results['frequency'].mean():.3f}")
        return results

# Use custom missing process
custom_missing = SinusoidalMissingProcess(input_dim=10)
model = NotMIWAE(
    input_dim=10,
    missing_process=custom_missing  # Pass instance instead of string
)
```

The `BaseMissingProcess` requires implementing:
- `forward(x)`: Returns logits for $p(s=1|x)$
- `interpret(verbose)` (optional): Returns dict with learned parameters

See `notebooks/demo_notmiwae_sinusoidal.ipynb` for a complete example.

#### Directional Missingness Control 

For `selfmasking_known_signs`, you can specify the direction of missingness per feature:

```python
import torch

# Define directional patterns for 4 features
signs = torch.tensor([
    +1.0,  # Feature 0: high values → missing (e.g., sensor saturation)
    +1.0,  # Feature 1: high values → missing
    -1.0,  # Feature 2: low values → missing (e.g., below detection threshold)
    -1.0   # Feature 3: low values → missing
])

model = NotMIWAE(
    input_dim=4,
    latent_dim=10,
    missing_process='selfmasking_known_signs',
    signs=signs  # Optional: defaults to all +1.0 (high→missing)
)
```

See `demo_signs.py` for a complete demonstration.

## Files

```
notmiwae_pytorch/
├── __init__.py          # Package initialization
├── models/
│   ├── __init__.py
│   ├── base.py          # Encoder, Decoders, MissingProcess
│   ├── notmiwae.py      # NotMIWAE model
│   └── miwae.py         # MIWAE model (baseline)
├── trainer.py           # Training loop with TensorBoard logging
├── utils.py             # Utility functions
├── example.py           # Complete example script
├── requirements.txt     # Dependencies
└── notebooks/
    └── demo_notmiwae.ipynb  # Interactive demo notebook
```

**Note:** DataLoaders should return `(x_filled, mask, x_original)` tuples where:
- `x_filled`: Data with missing values filled (e.g., with 0)
- `mask`: Binary mask (1=observed, 0=missing)
- `x_original`: Original complete data (for evaluation)

## Running the Example

```bash
cd notmiwae_pytorch
python example.py
```

This will:
1. Load the UCI Wine Quality dataset
2. Introduce MNAR missing values
3. Train both not-MIWAE and MIWAE models
4. Compare imputation performance

## TensorBoard

To view training logs:

```bash
tensorboard --logdir=./runs
```

Then open http://localhost:6006 in your browser.

## Notebook Demo

See `notebooks/demo_notmiwae.ipynb` for an interactive demonstration with visualizations.

## Interpreting the Missing Process

After training, you can interpret what the model learned about the missing mechanism:

```python
# For selfmasking models: shows W (strength) and b (threshold) per feature
model.interpret_missing_process()
# Output: "feature_0: Higher values (>0.25) more likely MISSING (W=1.234)"

# For linear/nonlinear models: compute sensitivity matrix
sensitivity = model.compute_missing_sensitivity(x_sample)
```

## Differences from Original TensorFlow Implementation

This PyTorch implementation:
- Uses modern PyTorch conventions (nn.Module, DataLoader, etc.)
- Includes TensorBoard integration via `torch.utils.tensorboard`
- Provides cleaner separation of concerns (models, trainer)
- Adds type hints and comprehensive docstrings
- Includes missing process interpretation tools

## Citation

```bibtex
@inproceedings{ipsen2021notmiwae,
  title={not-MIWAE: Deep Generative Modelling with Missing not at Random Data},
  author={Ipsen, Niels Bruun and Mattei, Pierre-Alexandre and Frellsen, Jes},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

## License

This implementation follows the license of the original repository.
