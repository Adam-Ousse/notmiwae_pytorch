"""
not-MIWAE: Deep Generative Modelling with Missing Not At Random Data

Paper: "not-MIWAE: Deep Generative Modelling with Missing not at Random Data"
Authors: Niels Bruun Ipsen, Pierre-Alexandre Mattei, Jes Frellsen (ICLR 2021)

Extends MIWAE by explicitly modeling the missing data mechanism p(s|x).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
import numpy as np
from typing import Optional, Literal

from .base import Encoder, GaussianDecoder, BernoulliDecoder, MissingProcess


class NotMIWAE(nn.Module):
    """
    not-MIWAE: Handles Missing Not At Random (MNAR) data.
    
    Maximizes: log p(x_obs, s) >= E_q[log p(x_obs|z) + log p(s|x) + log p(z) - log q(z|x)]
    
    Args:
        input_dim: Dimension of the input data
        latent_dim: Dimension of the latent space  
        hidden_dim: Dimension of hidden layers
        n_samples: Number of importance samples (K)
        out_dist: Output distribution ('gauss' or 'bern')
        missing_process: Type of missing mechanism 
            ('selfmasking', 'selfmasking_known', 'linear', 'nonlinear')
        feature_names: Optional list of feature names for interpretation
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 50,
        hidden_dim: int = 128,
        n_samples: int = 20,
        out_dist: Literal['gauss', 'bern'] = 'gauss',
        missing_process: Literal['selfmasking', 'selfmasking_known', 'linear', 'nonlinear'] = 'selfmasking',
        feature_names: Optional[list] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_samples = n_samples
        self.out_dist = out_dist
        self.missing_process_type = missing_process
        self.feature_names = feature_names
        
        # Encoder q(z|x)
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        
        # Decoder p(x|z)
        if out_dist == 'gauss':
            self.decoder = GaussianDecoder(latent_dim, hidden_dim, input_dim)
        else:
            self.decoder = BernoulliDecoder(latent_dim, hidden_dim, input_dim)
            
        # Missing process p(s|x)
        self.missing_model = MissingProcess(
            input_dim, 
            missing_process, 
            hidden_dim=hidden_dim // 2,
            feature_names=feature_names
        )
        
        # Prior p(z)
        self.register_buffer('prior_mu', torch.zeros(1))
        self.register_buffer('prior_std', torch.ones(1))
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Reparameterization trick for sampling z ~ q(z|x)."""
        std = torch.exp(0.5 * logvar)
        mu = mu.unsqueeze(1).expand(-1, n_samples, -1)
        std = std.unsqueeze(1).expand(-1, n_samples, -1)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def forward(self, x: torch.Tensor, s: torch.Tensor, n_samples: Optional[int] = None) -> dict:
        """
        Forward pass.
        
        Args:
            x: Data with missing values filled (batch_size, input_dim)
            s: Observation mask, 1=observed, 0=missing (batch_size, input_dim)
            n_samples: Number of importance samples
        """
        if n_samples is None:
            n_samples = self.n_samples
            
        # Encode
        q_mu, q_logvar = self.encoder(x)
        z = self.reparameterize(q_mu, q_logvar, n_samples)
        
        # Decode
        if self.out_dist == 'gauss':
            x_mu, x_std = self.decoder(z)
            p_x_given_z = Normal(x_mu, x_std)
            # Reparameterization: x ~ N(mu, std) = mu + std * eps
            eps = torch.randn_like(x_mu)
            x_sample = x_mu + x_std * eps
        else:
            logits = self.decoder(z)
            p_x_given_z = Bernoulli(logits=logits)
            x_mu = torch.sigmoid(logits)
            x_sample = p_x_given_z.sample().float()
        
        # Expand x and s for K samples
        x_expanded = x.unsqueeze(1).expand(-1, n_samples, -1)
        s_expanded = s.unsqueeze(1).expand(-1, n_samples, -1)
        
        # log p(x_obs|z) - only observed dimensions
        log_p_x_given_z = (s_expanded * p_x_given_z.log_prob(x_expanded)).sum(dim=-1)
        
        # Mix observed x with sampled x for missing values
        x_mixed = x_sample * (1 - s_expanded) + x_expanded * s_expanded
        
        # log p(s|x) - missing process
        miss_logits = self.missing_model(x_mixed)
        log_p_s_given_x = Bernoulli(logits=miss_logits).log_prob(s_expanded).sum(dim=-1)
        
        # log q(z|x)
        q_mu_exp = q_mu.unsqueeze(1).expand(-1, n_samples, -1)
        q_std_exp = torch.exp(0.5 * q_logvar).unsqueeze(1).expand(-1, n_samples, -1)
        log_q_z_given_x = Normal(q_mu_exp, q_std_exp).log_prob(z).sum(dim=-1)
        
        # log p(z)
        prior = Normal(self.prior_mu, self.prior_std)
        log_p_z = prior.log_prob(z).sum(dim=-1)
        
        # not-MIWAE ELBO (includes missing process)
        log_w = log_p_x_given_z + log_p_s_given_x + log_p_z - log_q_z_given_x
        elbo = (torch.logsumexp(log_w, dim=1) - np.log(n_samples)).mean()
        
        # Also compute standard MIWAE ELBO for comparison
        log_w_miwae = log_p_x_given_z + log_p_z - log_q_z_given_x
        miwae_elbo = (torch.logsumexp(log_w_miwae, dim=1) - np.log(n_samples)).mean()
        
        return {
            'loss': -elbo,
            'elbo': elbo,
            'miwae_elbo': miwae_elbo,
            'log_p_x_given_z': log_p_x_given_z.mean(),
            'log_p_s_given_x': log_p_s_given_x.mean(),
            'log_p_z': log_p_z.mean(),
            'log_q_z_given_x': log_q_z_given_x.mean(),
            'x_recon': x_mu,
        }
    
    def impute(self, x: torch.Tensor, s: torch.Tensor, n_samples: int = 1000) -> torch.Tensor:
        """Impute missing values using importance-weighted averaging."""
        self.eval()
        with torch.no_grad():
            q_mu, q_logvar = self.encoder(x)
            z = self.reparameterize(q_mu, q_logvar, n_samples)
            
            if self.out_dist == 'gauss':
                x_mu, x_std = self.decoder(z)
                p_x_given_z = Normal(x_mu, x_std)
                x_sample = p_x_given_z.rsample()
            else:
                logits = self.decoder(z)
                p_x_given_z = Bernoulli(logits=logits)
                x_mu = torch.sigmoid(logits)
                x_sample = p_x_given_z.sample().float()
            
            x_expanded = x.unsqueeze(1).expand(-1, n_samples, -1)
            s_expanded = s.unsqueeze(1).expand(-1, n_samples, -1)
            
            # Mix observed and sampled
            x_mixed = x_sample * (1 - s_expanded) + x_expanded * s_expanded
            
            # Compute importance weights
            log_p_x_given_z = (s_expanded * p_x_given_z.log_prob(x_expanded)).sum(dim=-1)
            
            miss_logits = self.missing_model(x_mixed)
            log_p_s_given_x = Bernoulli(logits=miss_logits).log_prob(s_expanded).sum(dim=-1)
            
            q_mu_exp = q_mu.unsqueeze(1).expand(-1, n_samples, -1)
            q_std_exp = torch.exp(0.5 * q_logvar).unsqueeze(1).expand(-1, n_samples, -1)
            log_q_z_given_x = Normal(q_mu_exp, q_std_exp).log_prob(z).sum(dim=-1)
            
            prior = Normal(self.prior_mu, self.prior_std)
            log_p_z = prior.log_prob(z).sum(dim=-1)
            
            log_w = log_p_x_given_z + log_p_s_given_x + log_p_z - log_q_z_given_x
            w = F.softmax(log_w, dim=1)
            
            x_imputed = (w.unsqueeze(-1) * x_mu).sum(dim=1)
            return x * s + x_imputed * (1 - s)
    
    def interpret_missing_process(self, verbose: bool = True) -> dict:
        """
        Interpret the learned missing process parameters.
        
        Provides insights into which features are likely to be missing
        and under what conditions (e.g., high values, dependencies on other features).
        
        Args:
            verbose: If True, print human-readable interpretations
            
        Returns:
            Dictionary with detailed interpretation results
        """
        return self.missing_model.interpret(verbose=verbose)
    
    def compute_missing_sensitivity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute how much each input feature affects each feature's missingness.
        
        Useful for understanding complex (linear/nonlinear) missing mechanisms.
        
        Args:
            x: Sample data to compute sensitivity on (batch_size, input_dim)
            
        Returns:
            Sensitivity matrix (input_dim, input_dim) where [i,j] shows
            how much feature j affects feature i's probability of being missing
        """
        return self.missing_model.compute_sensitivity(x)
