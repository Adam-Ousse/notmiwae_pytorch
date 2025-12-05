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
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)
        return mu, logvar


class GaussianDecoder(nn.Module):
    """Gaussian decoder p(x|z) for continuous data."""
    
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_std = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.fc1(z))
        h = torch.tanh(self.fc2(h))
        mu = self.fc_mu(h)
        std = F.softplus(self.fc_std(h)) + 1e-6
        return mu, std


class BernoulliDecoder(nn.Module):
    """Bernoulli decoder p(x|z) for binary data."""
    
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_logits = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(z))
        h = torch.tanh(self.fc2(h))
        return self.fc_logits(h)


class MissingProcess(nn.Module):
    """
    Missing process model p(s|x) for MNAR data.
    
    Types:
    - 'selfmasking': logit(p(s=1|x)) = -W*(x-b), each feature's missingness depends on itself
    - 'selfmasking_known': Same but W > 0 (higher values more likely missing)
    - 'linear': logit(p(s=1|x)) = Ax + b, missingness can depend on all features
    - 'nonlinear': logit(p(s=1|x)) = MLP(x), flexible nonlinear dependencies
    
    Args:
        input_dim: Number of features
        missing_process: Type of missing mechanism
        hidden_dim: Hidden dimension for nonlinear model (default: 64)
        feature_names: Optional list of feature names for interpretation
    """
    
    def __init__(
        self, 
        input_dim: int, 
        missing_process: Literal['selfmasking', 'selfmasking_known', 'linear', 'nonlinear'] = 'selfmasking',
        hidden_dim: int = 64,
        feature_names: Optional[list] = None,
        signs: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.missing_process = missing_process
        self.input_dim = input_dim
        self.feature_names = feature_names or [f"feature_{i}" for i in range(input_dim)]
        
        if missing_process in ['selfmasking', 'selfmasking_known']:
            # Diagonal: each feature's missingness depends only on itself
            self.W = nn.Parameter(torch.ones(1, 1, input_dim))
            self.b = nn.Parameter(torch.zeros(1, 1, input_dim))
            
            # For selfmasking_known: register known signs
            # signs: +1.0 for "High values missing" (negative slope required)
            #        -1.0 for "Low values missing" (positive slope required)
            if missing_process == 'selfmasking_known':
                if signs is None:
                    # Default: all features have high values missing (paper's setup)
                    signs = torch.ones(1, 1, input_dim)
                else:
                    # Ensure correct shape (1, 1, input_dim)
                    if signs.dim() == 1:
                        signs = signs.view(1, 1, -1)
                    elif signs.dim() == 2:
                        signs = signs.view(1, 1, -1)
                self.register_buffer('signs', signs)
            
        elif missing_process == 'linear':
            # Full matrix: missingness can depend on all features
            self.linear = nn.Linear(input_dim, input_dim)
            
        elif missing_process == 'nonlinear':
            # MLP for complex dependencies
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
        else:
            raise ValueError(f"Unknown missing_process: {missing_process}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for p(s=1|x)."""
        if self.missing_process == 'selfmasking':
            return -self.W * (x - self.b)
        elif self.missing_process == 'selfmasking_known':
            # Constrain magnitude to be positive
            W_positive = F.softplus(self.W)
            
            # Apply direction based on signs:
            # If sign is +1 (High->Missing), we want Negative Slope: -1 * W * (x - b)
            # If sign is -1 (Low->Missing),  we want Positive Slope: +1 * W * (x - b)
            # Therefore: slope = -signs * W_positive
            slope = -self.signs * W_positive
            
            return slope * (x - self.b)
        elif self.missing_process == 'linear':
            return self.linear(x)
        else:  # nonlinear
            return self.mlp(x)
    
    def interpret(self, verbose: bool = True) -> dict:
        """
        Interpret the learned missing process parameters.
        
        Returns a dictionary with interpretable information about which features
        are likely to be missing and under what conditions.
        """
        results = {
            'process_type': self.missing_process,
            'feature_names': self.feature_names,
            'interpretations': []
        }
        
        if self.missing_process in ['selfmasking', 'selfmasking_known']:
            # Get W and b values
            if self.missing_process == 'selfmasking_known':
                W = F.softplus(self.W).detach().squeeze().cpu().numpy()
                signs = self.signs.detach().squeeze().cpu().numpy()
            else:
                W = self.W.detach().squeeze().cpu().numpy()
                signs = None
            b = self.b.detach().squeeze().cpu().numpy()
            
            results['W'] = W
            results['b'] = b
            if signs is not None:
                results['signs'] = signs
            
            for i in range(self.input_dim):
                w_i, b_i = W[i], b[i]
                name = self.feature_names[i]
                
                # Interpret the parameters
                # For selfmasking: logit(p(s=1|x)) = -W*(x-b)
                # For selfmasking_known: logit(p(s=1|x)) = -sign*W_positive*(x-b)
                #   If sign=+1: slope is -W (high values -> missing)
                #   If sign=-1: slope is +W (low values -> missing)
                
                if self.missing_process == 'selfmasking_known':
                    sign_i = signs[i]
                    effective_w = -sign_i * w_i  # This is the actual slope
                else:
                    sign_i = None
                    effective_w = -w_i
                
                if abs(effective_w) < 0.1:
                    direction = "no strong dependency"
                    interp = f"{name}: Nearly random missingness (W≈0)"
                elif effective_w < 0:  # Negative slope: high values -> negative logit -> missing
                    direction = "high values missing"
                    if self.missing_process == 'selfmasking_known':
                        interp = f"{name}: Higher values (>{b_i:.2f}) more likely MISSING (W={w_i:.3f}, sign={sign_i:+.0f})"
                    else:
                        interp = f"{name}: Higher values (>{b_i:.2f}) more likely MISSING (W={w_i:.3f})"
                else:  # Positive slope: low values -> negative logit -> missing
                    direction = "low values missing"
                    if self.missing_process == 'selfmasking_known':
                        interp = f"{name}: Lower values (<{b_i:.2f}) more likely MISSING (W={w_i:.3f}, sign={sign_i:+.0f})"
                    else:
                        interp = f"{name}: Lower values (<{b_i:.2f}) more likely MISSING (W={w_i:.3f})"
                
                results['interpretations'].append({
                    'feature': name,
                    'W': float(w_i),
                    'b': float(b_i),
                    'direction': direction,
                    'threshold': float(b_i)
                })
                
                if verbose:
                    print(interp)
                    
        elif self.missing_process == 'linear':
            # Analyze the weight matrix
            A = self.linear.weight.detach().cpu().numpy()  # (output, input)
            bias = self.linear.bias.detach().cpu().numpy()
            
            results['A'] = A
            results['bias'] = bias
            
            if verbose:
                print("Linear missing process: logit(p(s|x)) = Ax + b\n")
            
            for i in range(self.input_dim):
                name_i = self.feature_names[i]
                weights = A[i, :]
                
                # Find which features most influence this feature's missingness
                sorted_idx = np.argsort(np.abs(weights))[::-1]
                top_influences = []
                
                for j in sorted_idx[:3]:  # Top 3 influences
                    if abs(weights[j]) > 0.1:
                        name_j = self.feature_names[j]
                        direction = "↑" if weights[j] > 0 else "↓"
                        top_influences.append(f"{name_j}({direction}{abs(weights[j]):.2f})")
                
                results['interpretations'].append({
                    'feature': name_i,
                    'top_influences': top_influences,
                    'bias': float(bias[i]),
                    'self_weight': float(weights[i])
                })
                
                if verbose:
                    if top_influences:
                        print(f"{name_i} missingness influenced by: {', '.join(top_influences)}")
                    else:
                        print(f"{name_i}: Weak dependencies (mostly random)")
                        
        elif self.missing_process == 'nonlinear':
            # For nonlinear, we can only provide gradient-based sensitivity
            results['note'] = "Nonlinear model - use gradient analysis for interpretation"
            
            if verbose:
                print("Nonlinear missing process (MLP)")
                print("For detailed interpretation, use gradient-based sensitivity analysis:")
                print("  1. Compute gradients of logits w.r.t. inputs")
                print("  2. Average over dataset to find important features")
                
            # Provide layer info
            results['layers'] = []
            for name, param in self.mlp.named_parameters():
                results['layers'].append({
                    'name': name,
                    'shape': list(param.shape),
                    'norm': float(param.norm().item())
                })
                
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
        x = x.requires_grad_(True)
        logits = self.forward(x.unsqueeze(1)).squeeze(1)  # (batch, input_dim)
        
        sensitivity = torch.zeros(self.input_dim, self.input_dim)
        
        for i in range(self.input_dim):
            # Gradient of feature i's missingness logit w.r.t. all inputs
            grad = torch.autograd.grad(
                logits[:, i].sum(), x, retain_graph=True
            )[0]
            sensitivity[i] = grad.abs().mean(dim=0)
            
        return sensitivity.detach()
