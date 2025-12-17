import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
import numpy as np
from typing import Optional, Literal

from .supnotmiwae import PredictionHead


class SupMIWAE(nn.Module):
    """
    SupMIWAE: Supervised MIWAE without the missing process model.
    
    Maximizes: log p(x_obs, y) >= E_q[log p(x_obs|z) + log p(y|x) + log p(z) - log q(z|x)]
    
    This baseline ignores the missingness mechanism p(s|x) entirely, treating missing values
    as if they're missing completely at random (MCAR).
    
    Args:
        input_dim: Dimension of the input data
        latent_dim: Dimension of the latent space
        hidden_dim: Dimension of hidden layers
        n_samples: Number of importance samples (K)
        out_dist: Output distribution ('gauss' or 'bern')
        y_dim: Dimension of target variable (1 for regression, n_classes for classification)
        task: 'regression' or 'classification'
        feature_names: Optional list of feature names
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 50,
        hidden_dim: int = 128,
        n_samples: int = 20,
        out_dist: Literal['gauss', 'bern'] = 'gauss',
        y_dim: int = 1,
        task: Literal['regression', 'classification'] = 'regression',
        feature_names: Optional[list] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_samples = n_samples
        self.out_dist = out_dist
        self.y_dim = y_dim
        self.task = task
        self.feature_names = feature_names or [f'x_{i}' for i in range(input_dim)]
        
        # Encoder: q(z|x)
        from .base import Encoder
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        
        # Decoder: p(x|z)
        if out_dist == 'gauss':
            from .base import GaussianDecoder
            self.decoder = GaussianDecoder(latent_dim, hidden_dim, input_dim)
        else:
            from .base import BernoulliDecoder
            self.decoder = BernoulliDecoder(latent_dim, hidden_dim, input_dim)
        
        # Prediction head p(y|x)
        self.predictor = PredictionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=y_dim,
            task=task,
            predict_variance=(task == 'regression')
        )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(mu.size(0), n_samples, mu.size(1), device=mu.device)
        z = mu.unsqueeze(1) + eps * std.unsqueeze(1)
        return z
    
    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        y: torch.Tensor,
        n_samples: Optional[int] = None
    ) -> dict:
        """
        Forward pass for supervised training.
        
        Args:
            x: Data with missing values filled (batch_size, input_dim)
            s: Observation mask (batch_size, input_dim) - ignored for SupMIWAE
            y: Target labels/values (batch_size,) or (batch_size, y_dim)
            n_samples: Number of importance samples
            
        Returns:
            Dictionary with loss and diagnostic info
        """
        if n_samples is None:
            n_samples = self.n_samples
        
        # Encode
        q_mu, q_logvar = self.encoder(x)
        z = self.reparameterize(q_mu, q_logvar, n_samples)  # (batch_size, n_samples, latent_dim)

        # Decode p(x|z)
        if self.out_dist == 'gauss':
            x_mu, x_std = self.decoder(z)  # (batch_size, n_samples, input_dim)
            p_x_given_z = Normal(x_mu, x_std)
            eps_x = torch.randn_like(x_mu)
            x_sample = x_mu + x_std * eps_x
        else:
            logits_x = self.decoder(z)
            p_x_given_z = Bernoulli(logits=logits_x)
            x_mu = torch.sigmoid(logits_x)
            x_sample = p_x_given_z.sample().float()

        # Log p(x|z) treating all dims as observed (SupMIWAE ignores missingness mechanism)
        x_expanded = x.unsqueeze(1).expand(-1, n_samples, -1)
        log_p_x_given_z = (p_x_given_z.log_prob(x_expanded) * s.unsqueeze(1)).sum(dim=-1)
        
        # Log p(z) with broadcast over samples
        prior_mu = torch.zeros_like(q_mu).unsqueeze(1)       # (batch_size, 1, latent_dim)
        prior_std = torch.ones_like(q_mu).unsqueeze(1)       # (batch_size, 1, latent_dim)
        p_z = Normal(prior_mu, prior_std)
        log_p_z = p_z.log_prob(z).sum(dim=-1)  # (batch_size, n_samples)
        
        # Log q(z|x) with broadcast over samples
        q_mu_exp = q_mu.unsqueeze(1).expand(-1, n_samples, -1)
        q_std_exp = torch.exp(0.5 * q_logvar).unsqueeze(1).expand(-1, n_samples, -1)
        q_z_given_x = Normal(q_mu_exp, q_std_exp)
        log_q_z_given_x = q_z_given_x.log_prob(z).sum(dim=-1)  # (batch_size, n_samples)
        
        # Compute log p(y|x) - use x directly (ignoring missingness)
        y_logits = self.predictor(x_expanded)  # (batch_size, n_samples, y_dim or n_classes)
        
        if self.task == 'regression':
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            
            if y_logits.shape[-1] >= 2 * self.y_dim:
                y_mean = y_logits[..., :self.y_dim]
                y_logvar = y_logits[..., self.y_dim:2*self.y_dim]
                y_var = torch.nn.functional.softplus(y_logvar) + 1e-6
            else:
                y_mean = y_logits
                y_var = torch.ones_like(y_logits) * 0.1
            
            p_y_given_x = Normal(y_mean, torch.sqrt(y_var))
            y_expanded = y.unsqueeze(1).expand(-1, n_samples, -1)
            log_p_y_given_x = p_y_given_x.log_prob(y_expanded).sum(dim=-1)  # (batch_size, n_samples)
        
        elif self.task == 'classification':
            if y.dim() == 1:
                log_probs = F.log_softmax(y_logits, dim=-1)
                y_expanded = y.unsqueeze(1).expand(-1, n_samples)
                log_p_y_given_x = log_probs.gather(-1, y_expanded.unsqueeze(-1)).squeeze(-1)
            else:
                log_probs = F.log_softmax(y_logits, dim=-1)
                y_expanded = y.unsqueeze(1).expand(-1, n_samples, -1)
                log_p_y_given_x = (log_probs * y_expanded).sum(dim=-1)
        
        # MIWAE importance weights (without missing process)
        log_w = log_p_x_given_z + log_p_y_given_x + log_p_z - log_q_z_given_x
        
        elbo = (torch.logsumexp(log_w, dim=1) - np.log(n_samples)).mean()
        
        return {
            'loss': -elbo,
            'elbo': elbo,
            'log_p_x_given_z': log_p_x_given_z.mean(),
            'log_p_y_given_x': log_p_y_given_x.mean(),
            'log_p_z': log_p_z.mean(),
            'log_q_z_given_x': log_q_z_given_x.mean(),
            'x_recon': x_mu,
        }
    
    def predict(
        self,
        x: torch.Tensor,
        s: torch.Tensor = None,
        n_samples: int = 1000
    ) -> torch.Tensor:
        """
        Predict target y. SupMIWAE uses x directly (ignoring missingness).
        
        Args:
            x: Data with missing values filled (batch_size, input_dim)
            s: Observation mask - ignored for SupMIWAE
            n_samples: Number of samples for Monte Carlo averaging
            
        Returns:
            y_pred: Predicted targets
        """
        self.eval()
        with torch.no_grad():
            x_expanded = x.unsqueeze(1).expand(-1, n_samples, -1)
            y_logits = self.predictor(x_expanded)  # (batch_size, n_samples, y_dim or n_classes)
            
            if self.task == 'regression':
                y_pred = y_logits[..., :self.y_dim].mean(dim=1)  # Average samples, extract mean
                return y_pred
            
            elif self.task == 'classification':
                probs = F.softmax(y_logits, dim=-1)  # (batch_size, n_samples, n_classes)
                y_pred = probs.mean(dim=1)  # (batch_size, n_classes)
                return y_pred
            
            else:
                raise ValueError(f"Unknown task: {self.task}")
