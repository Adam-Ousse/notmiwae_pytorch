# not-MIWAE: Deep Generative Modelling with Missing Not at Random Data

**Course Project**: Probabilistic Graphical Models and Deep Generative Models  

## Team

**Adam Gassem**  
adam.gassem@ensta.fr · adam.gassem@ip-paris.fr  
ENSTA Paris · ENS Paris-Saclay, France

**Amine Maazizi**  
amine.maazizi@ensta.fr · amine.maazizi@ip-paris.fr  
ENSTA Paris · ENS Paris-Saclay, France

**Ewerthon Melzani**  
ewerthon.melzani@ensta.fr · ewerthon.melzani@ip-paris.fr  
ENSTA Paris · ENS Paris-Saclay, France

---

## Overview

This project implements and extends the **not-MIWAE** model from the paper:

> **not-MIWAE: Deep Generative Modelling with Missing not at Random Data**  
> Niels Bruun Ipsen, Pierre-Alexandre Mattei, Jes Frellsen  
> ICLR 2021 | [Paper](https://arxiv.org/abs/2006.12871)

The not-MIWAE extends the Missing Data Importance Weighted Autoencoder (MIWAE) by explicitly modeling the missing data mechanism. This allows it to handle **Missing Not At Random (MNAR)** data, where the probability of a value being missing depends on the value itself.

## Contributions

Beyond reviewing and reproducing the not-MIWAE framework, this work makes the following contributions:

1. **Unified PyTorch Implementation**: We provide unified PyTorch implementations of MIWAE, not-MIWAE, supMIWAE, and sup-not-MIWAE, released as open-source code on [GitHub](https://github.com/Adam-Ousse/notmiwae_pytorch) and distributed via [PyPI](https://pypi.org/project/notmiwae-pytorch/).

2. **Supervised MNAR Extension (sup-not-MIWAE)**: We propose a supervised extension obtained by faithful probabilistic integration of not-MIWAE and supMIWAE within the same modeling framework, providing a transparent supervised MNAR baseline.

3. **Extended Experiments**: We reproduce and extend the original experimental study by evaluating not-MIWAE on high-dimensional image data (CelebA dataset) with MNAR clipping experiments.

4. **Optimal Transport Interpretation**: We introduce an optimal-transport interpretation of imputation under MNAR, recasting classical loss-based point estimators as Wasserstein projections onto Dirac measures.





## Project Structure

```
notmiwae_pytorch/
├── notmiwae_pytorch/          # Main package
│   ├── __init__.py            # Package initialization
│   ├── models/                # Model implementations
│   │   ├── __init__.py
│   │   ├── base.py            # Encoder, Decoders, Missing Process classes
│   │   ├── miwae.py           # MIWAE (baseline, assumes MCAR)
│   │   ├── notmiwae.py        # not-MIWAE (handles MNAR)
│   │   ├── supmiwae.py        # Supervised MIWAE
│   │   └── supnotmiwae.py     # Supervised not-MIWAE (our contribution)
│   ├── trainer.py             # Training loop with logging
│   └── utils.py               # Utility functions
├── notebooks/                 # Jupyter notebooks
│   ├── demo_notmiwae.ipynb                   # Basic demo
│   ├── demo_supnotmiwae.ipynb                # Supervised learning demo
│   ├── demo_notmiwae_directional.ipynb       # Directional missingness
│   ├── notmiwae_CelebA.ipynb                 # Image imputation (CelebA)
│   ├── MNAR_simple_concrete.ipynb            # Concrete strength dataset
│   ├── MNAR_simple_banknote.ipynb            # Banknote authentication dataset
│   ├── MNAR_simple_white.ipynb               # White wine quality dataset
│   ├── MNAR_simple_breastUCI_dataset.ipynb   # Breast UCI dataset
│   └── evaluate_imputation_performance.ipynb # Performance evaluation
├── tests/                     # Unit tests
├── requirements.txt           # Dependencies
├── pyproject.toml            # Package configuration
└── README.md                 # This file
```

### Key Features

- **NotMIWAE Model**: Full implementation with encoder, decoder, and missing process networks
- **MIWAE Model**: Standard MIWAE for comparison (assumes MCAR)
- **Supervised Extensions**: SupMIWAE and Sup-not-MIWAE for classification/regression with MNAR data
- **Missing Process Interpretation**: Built-in tools to interpret learned missing mechanisms
- **Custom Missing Processes**: Extensible framework for domain-specific missingness patterns
- **Multiple Output Distributions**: Gaussian, Bernoulli, Student-t Complete training loop with TensorBoard logging, early stopping, and checkpointing
- **Comprehensive Demos**: 8+ notebooks demonstrating various use cases

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

### Basic Imputation with not-MIWAE

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from notmiwae_pytorch import NotMIWAE, Trainer
from notmiwae_pytorch.utils import set_seed, impute

# Set seed for reproducibility
set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare your data
# x_filled: data with missing values filled (e.g., with 0)
# mask: binary mask (1=observed, 0=missing)
# x_original: original complete data (for evaluation)
train_dataset = TensorDataset(x_filled, mask, x_original)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create model
model = NotMIWAE(
    input_dim=10,
    latent_dim=5,
    hidden_dim=128,
    n_samples=20,
    missing_process='selfmasking',  # Choose missing mechanism
    out_dist='gauss'  # 'gauss', 'bern', 'student_t'
).to(device)

# Train
trainer = Trainer(
    model,
    device=device,
    lr=1e-3,
    n_epochs=100,
    original_data_available=True  # Compute RMSE during training
)
history = trainer.train(train_loader)

# Impute missing values
X_imputed = impute(model, x_filled, mask, n_samples=1000)

# Interpret the learned missing mechanism
model.interpret_missing_process()
```

### Supervised Learning with sup-not-MIWAE

```python
from notmiwae_pytorch import SupNotMIWAE

# Prepare supervised data
# y: target labels (classification) or values (regression)
train_dataset = TensorDataset(x_filled, mask, y)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create supervised model
model = SupNotMIWAE(
    input_dim=10,
    latent_dim=5,
    hidden_dim=128,
    n_samples=20,
    missing_process='selfmasking',
    y_dim=2,  # Number of classes or output dimension
    task='classification'  # or 'regression'
).to(device)

# Train (includes both reconstruction and prediction objectives)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(100):
    for x, s, y in train_loader:
        x, s, y = x.to(device), s.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x, s, y)
        loss = out['loss']
        loss.backward()
        optimizer.step()

# Predict on test data (properly marginalizes over missing values)
y_pred = model.predict(x_test, mask_test, n_samples=1000)
```

### Custom Missing Process Example

```python
from notmiwae_pytorch import BaseMissingProcess
import torch.nn as nn

class TemperatureSensorClipping(BaseMissingProcess):
    """
    Sensor fails at high temperatures (e.g., > 750°C).
    Models P(missing | temperature) = sigmoid(W * (temp - threshold))
    """
    def __init__(self, input_dim, threshold=750.0, **kwargs):
        super().__init__(input_dim, **kwargs)
        self.W = nn.Parameter(torch.ones(1, 1, input_dim) * 5.0)  # Positive = high→missing
        self.threshold = nn.Parameter(torch.full((1, 1, input_dim), threshold))
    
    def forward(self, x):
        return self.W * (x - self.threshold)
    
    def interpret(self, verbose=True):
        if verbose:
            print(f"Learned threshold: {self.threshold.mean().item():.1f}°C")
        return {'threshold': self.threshold.detach().cpu().numpy()}

# Use custom missing process
model = NotMIWAE(
    input_dim=5,
    missing_process=TemperatureSensorClipping(input_dim=5, threshold=750.0)
)
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

### sup-not-MIWAE Objective (Our Contribution)

The supervised extension adds a prediction term $p(y|x)$ to the joint likelihood:

$$\log p(x_o, y, s) \geq \mathbb{E}_{q(z|x_o)}\left[\log \frac{1}{K}\sum_{k=1}^{K} \frac{p(x_o|z_k) \cdot p(y|x_k) \cdot p(s|x_k) \cdot p(z_k)}{q(z_k|x_o)}\right]$$

This properly marginalizes over missing values during prediction, avoiding the approximation error of two-stage approaches (impute then predict).

### Missing Process Models

The model supports several missing mechanisms through `p(s|x)`. **The more prior knowledge you have about the missing mechanism in your data, the more accurate the imputations will be.** Choose the model that best matches your assumptions:

1. **`selfmasking`**: $\text{logit}(p(s_d=1|x)) = -W_d(x_d - b_d)$
   - Each feature's missingness depends only on its own value
   - Suitable when you don't know the direction (high vs. low values missing)

2. **`selfmasking_known_signs`**: Same as above but with constrained $W_d > 0$ or $W_d < 0$
   - Use when you know the direction of missingness (e.g., sensor saturation at high values)
   - Supports directional control via `signs` parameter:
     - `+1.0`: High values more likely to be missing (e.g., sensor clipping)
     - `-1.0`: Low values more likely to be missing (e.g., below detection limit)

3. **`linear`**: Linear mapping from all features $x$ to logits
   - Missingness in one feature can depend on other features
   - More flexible but requires more data

4. **`nonlinear`**: MLP mapping from $x$ to logits
   - Most flexible, captures complex missingness patterns
   - Requires sufficient data to avoid overfitting

### Output Distributions

The decoder $p(x|z)$ supports multiple distributions:

- **`gauss`**: Gaussian with learned mean and variance (default)
- **`bern`**: Bernoulli for binary data
- **`student_t`**: Student-t with learned degrees of freedom (robust to outliers)

## Advanced Usage


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

See [demo_notmiwae_directional.ipynb](notebooks/demo_notmiwae_directional.ipynb) for a complete demonstration.

### Interpreting the Missing Process

After training, you can interpret what the model learned about the missing mechanism:

```python
# For selfmasking models: shows W (strength) and b (threshold) per feature
model.interpret_missing_process()
# Output: "feature_0: Higher values (>0.25) more likely MISSING (W=1.234)"

# For linear/nonlinear models: compute sensitivity matrix
sensitivity = model.compute_missing_sensitivity(x_sample)
```

### Using Different Output Distributions

```python
# Student-t for robust imputation with outliers
model = NotMIWAE(
    input_dim=10,
    out_dist='student_t',  # Learns degrees of freedom
    missing_process='selfmasking'
)

```

## Files

**Note:** This section has been replaced by the more detailed **Project Structure** section above.

## Demos and Notebooks

We provide comprehensive Jupyter notebooks demonstrating various aspects of the framework:

1. **[demo_notmiwae.ipynb](notebooks/demo_notmiwae.ipynb)**: Basic not-MIWAE usage and comparison with MIWAE
2. **[demo_supnotmiwae.ipynb](notebooks/demo_supnotmiwae.ipynb)**: Supervised learning with MNAR data (classification & regression)
3. **[motor_temperature_demo.ipynb](notebooks/motor_temperature_demo.ipynb)**: 1D sensor data with high-temperature failures
4. **[notmiwae_CelebA.ipynb](notebooks/notmiwae_CelebA.ipynb)**: Image imputation with clipping (overexposed pixels)
5. **[demo_notmiwae_sinusoidal.ipynb](notebooks/demo_notmiwae_sinusoidal.ipynb)**: Custom sinusoidal missing process
6. **[demo_notmiwae_directional.ipynb](notebooks/demo_notmiwae_directional.ipynb)**: Directional missingness control
7. **[MNAR_simple_concrete.ipynb](notebooks/MNAR_simple_concrete.ipynb)**: Concrete strength prediction dataset
8. **[evaluate_imputation_performance.ipynb](notebooks/evaluate_imputation_performance.ipynb)**: Performance benchmarking

To run the notebooks:

```bash
git clone https://github.com/Adam-Ousse/notmiwae_pytorch.git
cd notmiwae_pytorch
pip install -e .
jupyter notebook notebooks/
```

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

## Implementation Notes

### Differences from Original TensorFlow Implementation

This PyTorch implementation:
- Uses modern PyTorch conventions (nn.Module, DataLoader, etc.)
- Includes TensorBoard integration via `torch.utils.tensorboard`
- Provides cleaner separation of concerns (models, trainer)
- Adds type hints and comprehensive docstrings
- Includes missing process interpretation tools
- Extends with supervised learning capabilities (sup-not-MIWAE)
- Supports multiple output distributions (Gaussian, Bernoulli, Student-t)

### Data Format

DataLoaders should return `(x_filled, mask, x_original)` tuples where:
- `x_filled`: Data with missing values filled (e.g., with 0)
- `mask`: Binary mask (1=observed, 0=missing)
- `x_original`: Original complete data (for evaluation, optional)

For supervised learning, return `(x_filled, mask, y)` where `y` are the targets.

## References

[1] Ipsen, N. B., Mattei, P. A., & Frellsen, J. (2021). not-MIWAE: Deep Generative Modelling with Missing not at Random Data. *International Conference on Learning Representations (ICLR)*.

[2] Ipsen, N. B., Mattei, P. A., & Frellsen, J. (2022). How to deal with missing data in supervised deep learning? *International Conference on Learning Representations (ICLR)*.

## Citation

If you use this code in your research, please cite the original papers:

```bibtex
@inproceedings{ipsen2021notmiwae,
  title={not-MIWAE: Deep Generative Modelling with Missing not at Random Data},
  author={Ipsen, Niels Bruun and Mattei, Pierre-Alexandre and Frellsen, Jes},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

@inproceedings{ipsen2022supmiwae,
  title={How to deal with missing data in supervised deep learning?},
  author={Ipsen, Niels Bruun and Mattei, Pierre-Alexandre and Frellsen, Jes},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

## License

This implementation follows the license of the original repository.

## Acknowledgments

This project was developed as part of the **Probabilistic Graphical Models and Deep Generative Models** course. We thank the course instructors and the original authors of not-MIWAE and supMIWAE for their foundational work.

---

**Contact**: For questions or issues, please open an issue on [GitHub](https://github.com/Adam-Ousse/notmiwae_pytorch/issues).
