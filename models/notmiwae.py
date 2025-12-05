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
        signs: Optional tensor of shape (input_dim,) with +1.0 for high-values-missing,
               -1.0 for low-values-missing. Only used with 'selfmasking_known'.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 50,
        hidden_dim: int = 128,
        n_samples: int = 20,
        out_dist: Literal['gauss', 'bern'] = 'gauss',
        missing_process: Literal['selfmasking', 'selfmasking_known', 'linear', 'nonlinear'] = 'selfmasking',
        feature_names: Optional[list] = None,
        signs: Optional[torch.Tensor] = None
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
            feature_names=feature_names,
            signs=signs
        )
        
        # Prior p(z)
        self.register_buffer('prior_mu', torch.zeros(1))
        self.register_buffer('prior_std', torch.ones(1))
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Reparameterization trick for sampling z ~ q(z|x). using mu and logvar calculate n_samples"""
        std = torch.exp(0.5 * logvar)
        mu = mu.unsqueeze(1).expand(-1, n_samples, -1)
        std = std.unsqueeze(1).expand(-1, n_samples, -1)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def _compute_importance_weights(
        self, 
        x: torch.Tensor, 
        s: torch.Tensor, 
        z: torch.Tensor,
        q_mu: torch.Tensor,
        q_logvar: torch.Tensor,
        n_samples: int,
        return_reconstructions: bool = False
    ) -> dict:
        """
        Compute importance weights and related quantities for ELBO/imputation.
        
        Args:
            x: Data with missing values filled (batch_size, input_dim)
            s: Observation mask (batch_size, input_dim)
            z: Sampled latent variables (batch_size, n_samples, latent_dim)
            q_mu: Encoder mean (batch_size, latent_dim)
            q_logvar: Encoder log variance (batch_size, latent_dim)
            n_samples: Number of importance samples
            return_reconstructions: If True, return x_mu and x_sample
            
        Returns:
            Dictionary with log_w (importance weights), and optionally x_mu, x_sample, p_x_given_z
        """
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
        
        # not-MIWAE importance weights (includes missing process)
        log_w = log_p_x_given_z + log_p_s_given_x + log_p_z - log_q_z_given_x
        
        # Standard MIWAE weights (for comparison)
        log_w_miwae = log_p_x_given_z + log_p_z - log_q_z_given_x
        
        result = {
            'log_w': log_w,
            'log_w_miwae': log_w_miwae,
            'log_p_x_given_z': log_p_x_given_z,
            'log_p_s_given_x': log_p_s_given_x,
            'log_p_z': log_p_z,
            'log_q_z_given_x': log_q_z_given_x,
        }
        
        if return_reconstructions:
            result['x_mu'] = x_mu
            result['x_sample'] = x_sample
            result['p_x_given_z'] = p_x_given_z
        
        return result
    
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
        
        # Compute importance weights and related quantities
        weights_info = self._compute_importance_weights(
            x, s, z, q_mu, q_logvar, n_samples, return_reconstructions=True
        )
        
        # not-MIWAE ELBO (includes missing process)
        log_w = weights_info['log_w']
        elbo = (torch.logsumexp(log_w, dim=1) - np.log(n_samples)).mean()
        
        # Standard MIWAE ELBO for comparison
        log_w_miwae = weights_info['log_w_miwae']
        miwae_elbo = (torch.logsumexp(log_w_miwae, dim=1) - np.log(n_samples)).mean()
        
        return {
            'loss': -elbo,
            'elbo': elbo,
            'miwae_elbo': miwae_elbo,
            'log_p_x_given_z': weights_info['log_p_x_given_z'].mean(),
            'log_p_s_given_x': weights_info['log_p_s_given_x'].mean(),
            'log_p_z': weights_info['log_p_z'].mean(),
            'log_q_z_given_x': weights_info['log_q_z_given_x'].mean(),
            'x_recon': weights_info['x_mu'],
        }
    
    def impute(
        self, 
        x: torch.Tensor, 
        s: torch.Tensor, 
        n_samples: int = 1000,
        solver: Literal['l2', 'l1'] = 'l2'
    ) -> torch.Tensor:
        """
        Impute missing values using importance-weighted averaging.
        
        Args:
            x: Data with missing values filled (batch_size, input_dim)
            s: Observation mask (batch_size, input_dim)
            n_samples: Number of importance samples
            solver: Loss function for imputation
                - 'l2': Squared loss, returns conditional mean (default)
                - 'l1': Absolute loss, returns conditional median
        
        Returns:
            x_imputed: Data with missing values imputed
            
        Notes:
            L2 (squared loss): Optimal imputation is the conditional mean E[x_m|x_o, s]
            L1 (absolute loss): Optimal imputation is the conditional median, estimated
            by solving F_j(x_j) = 0.5 where F_j is the CDF of each missing feature.
        """
        self.eval()
        with torch.no_grad():
            # Encode
            q_mu, q_logvar = self.encoder(x)
            z = self.reparameterize(q_mu, q_logvar, n_samples)
            
            # Compute importance weights
            weights_info = self._compute_importance_weights(
                x, s, z, q_mu, q_logvar, n_samples, return_reconstructions=True
            )
            
            # Normalized importance weights: α_k = w_k / Σw_k
            log_w = weights_info['log_w']
            alpha = F.softmax(log_w, dim=1)  # (batch_size, n_samples)
            
            x_mu = weights_info['x_mu']  # (batch_size, n_samples, input_dim)
            
            if solver == 'l2':
                # L2 loss: Conditional mean E[x_m|x_o, s] ≈ Σ α_k * E[x_m|x_o, s, z_k]
                x_imputed = (alpha.unsqueeze(-1) * x_mu).sum(dim=1)
                
            elif solver == 'l1':
                # L1 loss: Conditional median via CDF estimation
                # For Gaussian: compute weighted quantile at 0.5
                if self.out_dist == 'gauss':
                    x_imputed = self._compute_conditional_median_gaussian(
                        x_mu, weights_info['p_x_given_z'], alpha
                    )
                else:
                    # For Bernoulli, median is less meaningful, fall back to mean
                    x_imputed = (alpha.unsqueeze(-1) * x_mu).sum(dim=1)
            else:
                raise ValueError(f"Unknown solver: {solver}. Choose 'l2' or 'l1'.")
            
            # Keep observed values, impute missing
            return x * s + x_imputed * (1 - s)
    
    def _compute_conditional_median_gaussian(
        self,
        x_mu: torch.Tensor,
        p_x_given_z: Normal,
        alpha: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute conditional median for Gaussian observation model.
        
        Uses CDF estimation: F_j(x_j) = Σ alpha_k * F_{x_j|x_o,s,z_k}(x_j)
        Solves F_j(x_j) = 0.5 for each feature j.
        
        Args:
            x_mu: Means from decoder (batch_size, n_samples, input_dim)
            p_x_given_z: Normal distribution p(x|z)
            alpha: Normalized importance weights (batch_size, n_samples)
            
        Returns:
            x_median: Conditional median (batch_size, input_dim)
        """
        batch_size, n_samples, input_dim = x_mu.shape
        
        # For Gaussian mixture, approximate median by weighted median
        # Sort samples by value for each feature
        x_sorted, sort_idx = torch.sort(x_mu, dim=1)  # (batch, n_samples, input_dim)
        
        # Rearrange weights according to sorted order
        # Expand alpha for broadcasting
        alpha_expanded = alpha.unsqueeze(-1).expand(-1, -1, input_dim)  # (batch, n_samples, input_dim)
        alpha_sorted = torch.gather(alpha_expanded, 1, sort_idx)
        
        # Compute cumulative sum of weights
        alpha_cumsum = torch.cumsum(alpha_sorted, dim=1)  # (batch, n_samples, input_dim)
        
        # Find index where cumsum >= 0.5 (median position)
        median_idx = (alpha_cumsum >= 0.5).to(torch.float).argmax(dim=1)  # (batch, input_dim)
        
        # Gather median values
        median_idx_expanded = median_idx.unsqueeze(1)  # (batch, 1, input_dim)
        x_median = torch.gather(x_sorted, 1, median_idx_expanded).squeeze(1)  # (batch, input_dim)
        
        return x_median
    
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
