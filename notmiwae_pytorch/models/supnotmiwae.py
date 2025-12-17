"""
sup-not-MIWAE: Supervised extension of not-MIWAE for supervised learning with MNAR data.

Paper: "not-MIWAE: Deep Generative Modelling with Missing not at Random Data"
Authors: Niels Bruun Ipsen, Pierre-Alexandre Mattei, Jes Frellsen (ICLR 2021)

Paper: "How to deal with missing data in supervised deep learning"
Authors: Niels Bruun Ipsen, Pierre-Alexandre Mattei, Jes Frellsen (ICLR 2022)

Extends not-MIWAE by adding a supervised component p(y|x) for classification/regression
with missing data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli
import numpy as np
from typing import Optional, Literal

from .notmiwae import NotMIWAE


class PredictionHead(nn.Module):
    """
    Low-capacity prediction head:
      - Classification: logistic/softmax regression (single Linear)
      - Regression: linear regression (single Linear), optionally predicts log-variance too

    Args:
        input_dim: input feature dimension
        hidden_dim: kept for API compatibility (unused)
        output_dim: 1 for binary/regression scalar, n_classes for multiclass
        task: 'regression' or 'classification'
        predict_variance: for regression, output (mean, logvar) if True
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,  # unused, kept to avoid changing your call sites
        output_dim: int,
        task: Literal['regression', 'classification'] = 'regression',
        predict_variance: bool = True
    ):
        super().__init__()
        self.task = task
        self.output_dim = output_dim
        self.predict_variance = bool(predict_variance) and (task == "regression")

        final_output_dim = output_dim * 2 if self.predict_variance else output_dim
        self.net = nn.Linear(input_dim, final_output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_dim) or (batch, n_samples, input_dim)
        returns:
          - regression + variance: (.., 2*output_dim) = [mean, logvar]
          - regression no variance: (.., output_dim) = mean
          - classification: (.., output_dim) logits
        """
        return self.net(x)


class SupNotMIWAE(NotMIWAE):
    """
    sup-not-MIWAE: Supervised not-MIWAE for prediction with MNAR data.
    
    Maximizes: log p(x_obs, y, s) >= E_q[log p(x_obs|z) + log p(y|x) + log p(s|x) + log p(z) - log q(z|x)]
    
    Conceptual difference: Adds p(y|x) term to the likelihood for supervised learning.
    
    Args:
        input_dim: Dimension of the input data
        latent_dim: Dimension of the latent space
        hidden_dim: Dimension of hidden layers
        n_samples: Number of importance samples (K)
        out_dist: Output distribution ('gauss' or 'bern')
        missing_process: Type of missing mechanism
        y_dim: Dimension of target variable (1 for regression, n_classes for classification)
        task: 'regression' or 'classification'
        feature_names: Optional list of feature names
        signs: Optional tensor for selfmasking_known
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 50,
        hidden_dim: int = 128,
        n_samples: int = 20,
        out_dist: Literal['gauss', 'bern'] = 'gauss',
        missing_process: Literal['selfmasking', 'selfmasking_known', 'linear', 'nonlinear'] = 'selfmasking',
        y_dim: int = 1,
        task: Literal['regression', 'classification'] = 'regression',
        feature_names: Optional[list] = None,
        signs: Optional[torch.Tensor] = None
    ):
        # Initialize parent not-MIWAE
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_samples=n_samples,
            out_dist=out_dist,
            missing_process=missing_process,
            feature_names=feature_names,
            signs=signs
        )
        
        # Add supervised components
        self.y_dim = y_dim
        self.task = task
        
        # Prediction head p(y|x)
        self.predictor = PredictionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=y_dim,
            task=task
        )
        
    def _compute_importance_weights(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        q_mu: torch.Tensor,
        q_logvar: torch.Tensor,
        n_samples: int,
        y: Optional[torch.Tensor] = None,
        return_reconstructions: bool = False
    ) -> dict:
        """
        Compute importance weights with supervised component.
        
        Args:
            x: Data with missing values filled (batch_size, input_dim)
            s: Observation mask (batch_size, input_dim)
            z: Sampled latent variables (batch_size, n_samples, latent_dim)
            q_mu: Encoder mean (batch_size, latent_dim)
            q_logvar: Encoder log variance (batch_size, latent_dim)
            n_samples: Number of importance samples
            y: Target labels/values (batch_size,) for classification or (batch_size, y_dim) for regression
            return_reconstructions: If True, return additional outputs
            
        Returns:
            Dictionary with importance weights and related quantities
        """
        # Get base importance weights from parent class
        base_weights = super()._compute_importance_weights(
            x, s, z, q_mu, q_logvar, n_samples, return_reconstructions=True
        )
        
        # If no labels provided (e.g., at test time without y), return base weights
        if y is None:
            return base_weights
        
        # Compute log p(y|x) for supervised component
        x_mixed = base_weights['x_sample'] * (1 - s.unsqueeze(1).expand_as(base_weights['x_sample'])) + \
                  x.unsqueeze(1).expand_as(base_weights['x_sample']) * s.unsqueeze(1).expand_as(base_weights['x_sample'])
        
        # Predict y from completed x
        y_logits = self.predictor(x_mixed)  # (batch_size, n_samples, y_dim)
        
        if self.task == 'regression':
            # Gaussian likelihood for regression with learned variance
            if y.dim() == 1:
                y = y.unsqueeze(-1)  # (batch_size, 1)
            
            # Use softplus(y_logits_var) for variance to ensure positivity
            # Split output for mean and variance if output_dim > y_dim
            if y_logits.shape[-1] >= 2 * self.y_dim:
                y_mean = y_logits[..., :self.y_dim]
                y_logvar = y_logits[..., self.y_dim:2*self.y_dim]
                y_var = torch.nn.functional.softplus(y_logvar) + 1e-6
            else:
                y_mean = y_logits
                y_var = torch.ones_like(y_logits) * 0.1  # Default variance
            
            p_y_given_x = Normal(y_mean, torch.sqrt(y_var))
            y_expanded = y.unsqueeze(1).expand(-1, n_samples, -1)  # (batch_size, n_samples, y_dim)
            log_p_y_given_x = p_y_given_x.log_prob(y_expanded).sum(dim=-1)  # (batch_size, n_samples)
            
        elif self.task == 'classification':
            # Categorical likelihood for classification
            if y.dim() == 1:
                # Class indices (batch_size,)
                log_probs = F.log_softmax(y_logits, dim=-1)  # (batch_size, n_samples, n_classes)
                y_expanded = y.unsqueeze(1).expand(-1, n_samples)  # (batch_size, n_samples)
                log_p_y_given_x = log_probs.gather(-1, y_expanded.unsqueeze(-1)).squeeze(-1)  # (batch_size, n_samples)
            else:
                # One-hot encoded (batch_size, n_classes)
                log_probs = F.log_softmax(y_logits, dim=-1)
                y_expanded = y.unsqueeze(1).expand(-1, n_samples, -1)
                log_p_y_given_x = (log_probs * y_expanded).sum(dim=-1)  # (batch_size, n_samples)
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        # Supervised importance weights: add log p(y|x) term
        log_w_sup = (
            base_weights['log_p_x_given_z'] +
            log_p_y_given_x +  # NEW: supervised term
            base_weights['log_p_s_given_x'] +
            base_weights['log_p_z'] -
            base_weights['log_q_z_given_x']
        )
        
        # Build result dictionary
        result = {
            'log_w': log_w_sup,
            'log_w_notmiwae': base_weights['log_w'],  # Original not-MIWAE weights (without y)
            'log_w_miwae': base_weights['log_w_miwae'],  # MIWAE weights (no y, no s)
            'log_p_x_given_z': base_weights['log_p_x_given_z'],
            'log_p_y_given_x': log_p_y_given_x,
            'log_p_s_given_x': base_weights['log_p_s_given_x'],
            'log_p_z': base_weights['log_p_z'],
            'log_q_z_given_x': base_weights['log_q_z_given_x'],
        }
        
        if return_reconstructions:
            result['x_mu'] = base_weights['x_mu']
            result['x_sample'] = base_weights['x_sample']
            result['x_mixed'] = x_mixed
            result['y_logits'] = y_logits
            result['p_x_given_z'] = base_weights['p_x_given_z']
        
        return result
    
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
            s: Observation mask (batch_size, input_dim)
            y: Target labels/values (batch_size,) or (batch_size, y_dim)
            n_samples: Number of importance samples
            
        Returns:
            Dictionary with loss and diagnostic info
        """
        if n_samples is None:
            n_samples = self.n_samples
        
        # Encode
        q_mu, q_logvar = self.encoder(x)
        z = self.reparameterize(q_mu, q_logvar, n_samples)
        
        # Compute supervised importance weights
        weights_info = self._compute_importance_weights(
            x, s, z, q_mu, q_logvar, n_samples, y=y, return_reconstructions=True
        )
        
        # sup-not-MIWAE ELBO (includes y term)
        log_w = weights_info['log_w']
        elbo = (torch.logsumexp(log_w, dim=1) - np.log(n_samples)).mean()
        
        # Comparison ELBOs
        log_w_notmiwae = weights_info['log_w_notmiwae']
        notmiwae_elbo = (torch.logsumexp(log_w_notmiwae, dim=1) - np.log(n_samples)).mean()
        
        log_w_miwae = weights_info['log_w_miwae']
        miwae_elbo = (torch.logsumexp(log_w_miwae, dim=1) - np.log(n_samples)).mean()
        
        return {
            'loss': -elbo,
            'elbo': elbo,
            'notmiwae_elbo': notmiwae_elbo,
            'miwae_elbo': miwae_elbo,
            'log_p_x_given_z': weights_info['log_p_x_given_z'].mean(),
            'log_p_y_given_x': weights_info['log_p_y_given_x'].mean(),
            'log_p_s_given_x': weights_info['log_p_s_given_x'].mean(),
            'log_p_z': weights_info['log_p_z'].mean(),
            'log_q_z_given_x': weights_info['log_q_z_given_x'].mean(),
            'x_recon': weights_info['x_mu'],
        }
    
    def predict(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        n_samples: int = 1000,
        return_variance: bool = False
    ) -> torch.Tensor:
        """
        Predict target y from incomplete data using importance-weighted averaging.
        
        Computes: p(y|x_obs, s) ≈ Σ_k α_k p(y|x_k)
        
        This is the key benefit of sup-not-MIWAE: proper prediction under MNAR missingness.
        
        Args:
            x: Data with missing values filled (batch_size, input_dim)
            s: Observation mask (batch_size, input_dim)
            n_samples: Number of importance samples
            return_variance: If True, also return predictive variance (regression only)
            
        Returns:
            y_pred: Predicted targets
                - Regression: (batch_size, y_dim)
                - Classification: (batch_size, n_classes) probabilities
            (Optional) y_var: Predictive variance (batch_size, y_dim) for regression
        """
        self.eval()
        with torch.no_grad():
            # Encode
            q_mu, q_logvar = self.encoder(x)
            z = self.reparameterize(q_mu, q_logvar, n_samples)
            
            # Compute importance weights (without y)
            weights_info = self._compute_importance_weights(
                x, s, z, q_mu, q_logvar, n_samples, y=None, return_reconstructions=True
            )
            
            # Use not-MIWAE weights (without y term) for prediction
            # When y=None, base weights are returned with 'log_w' key (not-MIWAE)
            log_w = weights_info['log_w']
            alpha = F.softmax(log_w, dim=1)  # (batch_size, n_samples)
            
            # Compute x_mixed: observed values + sampled missing values
            x_sample = weights_info['x_sample']  # (batch_size, n_samples, input_dim)
            s_expanded = s.unsqueeze(1).expand_as(x_sample)
            x_expanded = x.unsqueeze(1).expand_as(x_sample)
            x_mixed = x_sample * (1 - s_expanded) + x_expanded * s_expanded
            
            # Get predictions from each completed sample
            y_logits = self.predictor(x_mixed)  # (batch_size, n_samples, y_dim or n_classes)
            
            if self.task == 'regression':
                # Extract mean predictions (first y_dim elements)
                if y_logits.shape[-1] >= 2 * self.y_dim:
                    y_mean = y_logits[..., :self.y_dim]  # (batch_size, n_samples, y_dim)
                else:
                    y_mean = y_logits
                
                # Importance-weighted mean prediction
                y_pred = (alpha.unsqueeze(-1) * y_mean).sum(dim=1)  # (batch_size, y_dim)
                
                if return_variance:
                    # Importance-weighted variance
                    y_var = (alpha.unsqueeze(-1) * (y_mean - y_pred.unsqueeze(1)) ** 2).sum(dim=1)
                    return y_pred, y_var
                
                return y_pred
                
            elif self.task == 'classification':
                # Importance-weighted probabilities
                probs = F.softmax(y_logits, dim=-1)  # (batch_size, n_samples, n_classes)
                y_pred = (alpha.unsqueeze(-1) * probs).sum(dim=1)  # (batch_size, n_classes)
                
                return y_pred
            
            else:
                raise ValueError(f"Unknown task: {self.task}")
    
    def predict_with_imputation(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        n_samples: int = 1000,
        solver: Literal['l2', 'l1'] = 'l2'
    ) -> tuple:
        """
        First impute missing values, then predict target.
        
        This is a two-stage approach (impute then predict) as a baseline comparison.
        The `predict()` method is preferred as it marginalizes properly.
        
        Args:
            x: Data with missing values filled (batch_size, input_dim)
            s: Observation mask (batch_size, input_dim)
            n_samples: Number of importance samples
            solver: Imputation solver ('l2' or 'l1')
            
        Returns:
            y_pred: Predicted targets
            x_imputed: Imputed data
        """
        # First impute
        x_imputed = self.impute(x, s, n_samples=n_samples, solver=solver)
        
        # Then predict using complete data
        self.eval()
        with torch.no_grad():
            y_logits = self.predictor(x_imputed)
            
            if self.task == 'regression':
                y_pred = y_logits[..., :self.y_dim]  # Extract mean only
            else:  # classification
                y_pred = F.softmax(y_logits, dim=-1)
        
        return y_pred, x_imputed

