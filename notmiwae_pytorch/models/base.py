"""
Base components for MIWAE models: Encoder, Decoders, Missing Process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
import numpy as np
from typing import Optional, Tuple, Literal


class Encoder(nn.Module):
    """Encoder network q(z|x) that maps input data to latent space parameters."""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)
        return mu, logvar


class Encoder_CNN(nn.Module):
    """CNN Encoder for 32x32 grayscale images following SVHN architecture."""
    
    def __init__(self, latent_dim: int = 20):
        super().__init__()
        # Input: (batch, 1, 32, 32)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),   # -> (batch, 64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (batch, 128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> (batch, 256, 4, 4)
            nn.ReLU()
        )
        
        # Flatten: 256 * 4 * 4 = 4096
        self.fc_mu = nn.Linear(4096, latent_dim)
        self.fc_logvar = nn.Linear(4096, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Expects (batch, 1, 32, 32) input."""
        h = self.conv_layers(x)
        h = h.flatten(start_dim=1)  # Flatten to (batch, 4096)
        
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)
        return mu, logvar


class GaussianDecoder(nn.Module):
    """Gaussian decoder p(x|z) for continuous data."""
    
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_std = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(z)
        mu = self.fc_mu(h)
        std = F.softplus(self.fc_std(h)) + 1e-6
        return mu, std


class BernoulliDecoder(nn.Module):
    """Bernoulli decoder p(x|z) for binary data."""
    
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class GaussianDecoder_CNN(nn.Module):
    """CNN Gaussian decoder for 32x32 grayscale images following SVHN architecture."""
    
    def __init__(self, latent_dim: int = 20):
        super().__init__()
        # Dense layer to expand latent to 4096
        self.fc = nn.Linear(latent_dim, 4096)
        
        # Mu path
        self.deconv_mu = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # -> (batch, 256, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> (batch, 128, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # -> (batch, 64, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),     # -> (batch, 1, 32, 32)
            nn.Sigmoid()
        )
        
        # Std path
        self.deconv_std = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Returns flattened mu and std."""
        # Expand latent
        h = self.fc(z)
        h = h.view(-1, 256, 4, 4)  # Reshape to (batch, 256, 4, 4)
        
        # Generate mu and std
        mu = self.deconv_mu(h)  # (batch, 1, 32, 32)
        log_std = self.deconv_std(h)  # (batch, 1, 32, 32)
        std = F.softplus(log_std) + 1e-6
        
        # Flatten to (batch, 1024) for compatibility with MLP version
        mu_flat = mu.flatten(start_dim=1)
        std_flat = std.flatten(start_dim=1)
        
        return mu_flat, std_flat


class BaseMissingProcess(nn.Module):
    """
    Abstract base class for missing process models p(s|x).
    
    To create a custom missing process:
    1. Inherit from BaseMissingProcess
    2. Implement the forward() method to compute logits
    3. Optionally override interpret() for custom interpretation
    
    Example:
        class MyCustomMissing(BaseMissingProcess):
            def __init__(self, input_dim: int, **kwargs):
                super().__init__(input_dim, **kwargs)
                self.linear = nn.Linear(input_dim, input_dim)
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)
                
            def interpret(self, verbose: bool = True) -> dict:
                # Custom interpretation logic
                return {'weights': self.linear.weight.detach()}
    """
    
    def __init__(
        self, 
        input_dim: int,
        feature_names: Optional[list] = None,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_names = feature_names or [f"feature_{i}" for i in range(input_dim)]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for p(s=1|x).
        
        Args:
            x: Input data, shape can be:
               - (batch_size, input_dim) for single samples
               - (batch_size, n_samples, input_dim) for multiple importance samples
               
        Returns:
            logits: Same shape as input, logit(p(s=1|x))
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def interpret(self, verbose: bool = True) -> dict:
        """
        Interpret the learned missing process parameters.
        
        Override this method to provide custom interpretation for your missing process.
        
        Args:
            verbose: If True, print human-readable interpretations
            
        Returns:
            Dictionary with interpretation results
        """
        results = {
            'process_type': self.__class__.__name__,
            'feature_names': self.feature_names,
            'note': 'No custom interpretation implemented. Override interpret() method.'
        }
        
        if verbose:
            print(f"Missing process: {self.__class__.__name__}")
            print("No custom interpretation available.")
            
        return results
    
    def compute_sensitivity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient-based sensitivity: how much each input feature
        affects each output's missingness probability.
        
        Args:
            x: Input data (batch_size, input_dim)
            
        Returns:
            sensitivity: (input_dim, input_dim) matrix where [i,j] = 
                         how much feature j affects feature i's missingness
        """
        x = x.clone().requires_grad_(True)
        
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            logits = self.forward(x.unsqueeze(1)).squeeze(1)
        else:
            logits = self.forward(x)
            
        sensitivity = torch.zeros(self.input_dim, self.input_dim)
        
        for i in range(self.input_dim):
            grad = torch.autograd.grad(
                logits[:, i].sum(), x, retain_graph=True
            )[0]
            sensitivity[i] = grad.abs().mean(dim=0)
            
        return sensitivity.detach()


class SelfMaskingProcess(BaseMissingProcess):
    """
    Self-masking missing process: each feature's missingness depends only on itself.
    
    logit(p(s=1|x)) = -W * (x - b)
    
    Args:
        input_dim: Number of features
        feature_names: Optional list of feature names
    """
    
    def __init__(self, input_dim: int, feature_names: Optional[list] = None, **kwargs):
        super().__init__(input_dim, feature_names)
        self.W = nn.Parameter(torch.ones(1, 1, input_dim))
        self.b = nn.Parameter(torch.zeros(1, 1, input_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -self.W * (x - self.b)
    
    def interpret(self, verbose: bool = True) -> dict:
        W = self.W.detach().squeeze().cpu().numpy()
        b = self.b.detach().squeeze().cpu().numpy()
        
        results = {
            'process_type': 'selfmasking',
            'feature_names': self.feature_names,
            'W': W,
            'b': b,
            'interpretations': []
        }
        
        for i in range(self.input_dim):
            w_i, b_i = W[i], b[i]
            name = self.feature_names[i]
            effective_w = -w_i
            
            if abs(effective_w) < 0.1:
                direction = "no strong dependency"
                interp = f"{name}: Nearly random missingness (W≈0)"
            elif effective_w < 0:
                direction = "high values missing"
                interp = f"{name}: Higher values (>{b_i:.2f}) more likely MISSING (W={w_i:.3f})"
            else:
                direction = "low values missing"
                interp = f"{name}: Lower values (<{b_i:.2f}) more likely MISSING (W={w_i:.3f})"
            
            results['interpretations'].append({
                'feature': name, 'W': float(w_i), 'b': float(b_i),
                'direction': direction, 'threshold': float(b_i)
            })
            
            if verbose:
                print(interp)
                
        return results


class SelfMaskingKnownSignsProcess(BaseMissingProcess):
    """
    Self-masking with known direction: W is constrained positive, direction set by signs.
    
    logit(p(s=1|x)) = -signs * softplus(W) * (x - b)
    
    Args:
        input_dim: Number of features
        signs: Tensor of shape (input_dim,) with +1.0 for high-values-missing,
               -1.0 for low-values-missing. Default: all +1.0 (high values missing)
        feature_names: Optional list of feature names
    """
    
    def __init__(
        self, 
        input_dim: int, 
        signs: Optional[torch.Tensor] = None,
        feature_names: Optional[list] = None,
        **kwargs
    ):
        super().__init__(input_dim, feature_names)
        self.W = nn.Parameter(torch.ones(1, 1, input_dim))
        self.b = nn.Parameter(torch.zeros(1, 1, input_dim))
        
        if signs is None:
            signs = torch.ones(1, 1, input_dim)
        else:
            if signs.dim() == 1:
                signs = signs.view(1, 1, -1)
            elif signs.dim() == 2:
                signs = signs.view(1, 1, -1)
        self.register_buffer('signs', signs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_positive = F.softplus(self.W)
        slope = -self.signs * W_positive
        return slope * (x - self.b)
    
    def interpret(self, verbose: bool = True) -> dict:
        W = F.softplus(self.W).detach().squeeze().cpu().numpy()
        b = self.b.detach().squeeze().cpu().numpy()
        signs = self.signs.detach().squeeze().cpu().numpy()
        
        results = {
            'process_type': 'selfmasking_known_signs',
            'feature_names': self.feature_names,
            'W': W, 'b': b, 'signs': signs,
            'interpretations': []
        }
        
        for i in range(self.input_dim):
            w_i, b_i, sign_i = W[i], b[i], signs[i]
            name = self.feature_names[i]
            effective_w = -sign_i * w_i
            
            if abs(effective_w) < 0.1:
                direction = "no strong dependency"
                interp = f"{name}: Nearly random missingness (W≈0)"
            elif effective_w < 0:
                direction = "high values missing"
                interp = f"{name}: Higher values (>{b_i:.2f}) more likely MISSING (W={w_i:.3f}, sign={sign_i:+.0f})"
            else:
                direction = "low values missing"
                interp = f"{name}: Lower values (<{b_i:.2f}) more likely MISSING (W={w_i:.3f}, sign={sign_i:+.0f})"
            
            results['interpretations'].append({
                'feature': name, 'W': float(w_i), 'b': float(b_i),
                'direction': direction, 'threshold': float(b_i)
            })
            
            if verbose:
                print(interp)
                
        return results


class LinearMissingProcess(BaseMissingProcess):
    """
    Linear missing process: missingness can depend on all features.
    
    logit(p(s=1|x)) = Ax + b
    
    Args:
        input_dim: Number of features
        feature_names: Optional list of feature names
    """
    
    def __init__(self, input_dim: int, feature_names: Optional[list] = None, **kwargs):
        super().__init__(input_dim, feature_names)
        self.linear = nn.Linear(input_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    def interpret(self, verbose: bool = True) -> dict:
        A = self.linear.weight.detach().cpu().numpy()
        bias = self.linear.bias.detach().cpu().numpy()
        
        results = {
            'process_type': 'linear',
            'feature_names': self.feature_names,
            'A': A, 'bias': bias,
            'interpretations': []
        }
        
        if verbose:
            print("Linear missing process: logit(p(s|x)) = Ax + b\n")
        
        for i in range(self.input_dim):
            name_i = self.feature_names[i]
            weights = A[i, :]
            sorted_idx = np.argsort(np.abs(weights))[::-1]
            top_influences = []
            
            for j in sorted_idx[:3]:
                if abs(weights[j]) > 0.1:
                    name_j = self.feature_names[j]
                    direction = "↑" if weights[j] > 0 else "↓"
                    top_influences.append(f"{name_j}({direction}{abs(weights[j]):.2f})")
            
            results['interpretations'].append({
                'feature': name_i, 'top_influences': top_influences,
                'bias': float(bias[i]), 'self_weight': float(weights[i])
            })
            
            if verbose:
                if top_influences:
                    print(f"{name_i} missingness influenced by: {', '.join(top_influences)}")
                else:
                    print(f"{name_i}: Weak dependencies (mostly random)")
                    
        return results


class NonlinearMissingProcess(BaseMissingProcess):
    """
    Nonlinear missing process using MLP.
    
    logit(p(s=1|x)) = MLP(x)
    
    Args:
        input_dim: Number of features
        hidden_dim: Hidden layer dimension (default: 64)
        feature_names: Optional list of feature names
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 64,
        feature_names: Optional[list] = None,
        **kwargs
    ):
        super().__init__(input_dim, feature_names)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
    
    def interpret(self, verbose: bool = True) -> dict:
        results = {
            'process_type': 'nonlinear',
            'feature_names': self.feature_names,
            'note': 'Nonlinear model - use compute_sensitivity() for interpretation',
            'layers': []
        }
        
        if verbose:
            print("Nonlinear missing process (MLP)")
            print("Use compute_sensitivity(x) for gradient-based interpretation")
            
        for name, param in self.mlp.named_parameters():
            results['layers'].append({
                'name': name,
                'shape': list(param.shape),
                'norm': float(param.norm().item())
            })
            
        return results


# Factory function for backward compatibility
def MissingProcess(
    input_dim: int,
    missing_process: Literal['selfmasking', 'selfmasking_known_signs', 'linear', 'nonlinear'] = 'selfmasking',
    hidden_dim: int = 64,
    feature_names: Optional[list] = None,
    signs: Optional[torch.Tensor] = None
) -> BaseMissingProcess:
    """
    Factory function to create a missing process model.
    
    For custom missing processes, inherit from BaseMissingProcess directly.
    
    Args:
        input_dim: Number of features
        missing_process: Type of missing mechanism
        hidden_dim: Hidden dimension for nonlinear model
        feature_names: Optional list of feature names
        signs: Direction signs for selfmasking_known_signs
        
    Returns:
        BaseMissingProcess instance
    """
    if missing_process == 'selfmasking':
        return SelfMaskingProcess(input_dim, feature_names=feature_names)
    elif missing_process == 'selfmasking_known_signs':
        return SelfMaskingKnownSignsProcess(input_dim, signs=signs, feature_names=feature_names)
    elif missing_process == 'linear':
        return LinearMissingProcess(input_dim, feature_names=feature_names)
    elif missing_process == 'nonlinear':
        return NonlinearMissingProcess(input_dim, hidden_dim=hidden_dim, feature_names=feature_names)
    else:
        raise ValueError(f"Unknown missing_process: {missing_process}. "
                        f"Use 'selfmasking', 'selfmasking_known_signs', 'linear', 'nonlinear', "
                        f"or pass a custom BaseMissingProcess instance.")
