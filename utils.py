"""
Utility functions for not-MIWAE experiments.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt


def imputation_rmse(
    model,
    x_original: Union[np.ndarray, torch.Tensor],
    x_filled: Union[np.ndarray, torch.Tensor],
    mask: Union[np.ndarray, torch.Tensor],
    n_samples: int = 1000,
    batch_size: int = 100,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    solver= None # l1 or l2
) -> Tuple[float, np.ndarray]:
    """
    Compute imputation RMSE for missing values.
    
    Uses importance-weighted averaging to impute missing values
    and computes RMSE compared to the original data.
    
    Args:
        model: Trained NotMIWAE or MIWAE model
        x_original: Original complete data
        x_filled: Data with missing values filled (usually with 0)
        mask: Binary mask (1=observed, 0=missing)
        n_samples: Number of importance samples for imputation
        batch_size: Batch size for processing
        device: Device to use for computation
        verbose: Whether to print progress
        
    Returns:
        rmse: Root mean squared error of imputation
        x_imputed: The imputed data
    """
    if device is None:
        device = next(model.parameters()).device
        
    # Convert to tensors if needed
    if isinstance(x_original, np.ndarray):
        x_original = torch.tensor(x_original, dtype=torch.float32)
    if isinstance(x_filled, np.ndarray):
        x_filled = torch.tensor(x_filled, dtype=torch.float32)
    if isinstance(mask, np.ndarray):
        mask = torch.tensor(mask, dtype=torch.float32)
    
    model.eval()
    n = x_original.shape[0]
    x_imputed_list = []
    
    with torch.no_grad():
        for i in range(0, n, batch_size):
            end_idx = min(i + batch_size, n)
            x_batch = x_filled[i:end_idx].to(device)
            s_batch = mask[i:end_idx].to(device)
            if solver:
                x_imp = model.impute(x_batch, s_batch, n_samples=n_samples, solver=solver)
            else :
                x_imp = model.impute(x_batch, s_batch, n_samples=n_samples)
            x_imputed_list.append(x_imp.cpu())
            
            if verbose and i % 500 == 0:
                print(f"Imputing: {i}/{n}")
    
    x_imputed = torch.cat(x_imputed_list, dim=0)
    
    # Compute RMSE only for missing values
    missing_mask = (1 - mask).bool()
    
    if missing_mask.sum() == 0:
        return 0.0, x_imputed.numpy()
    
    squared_errors = (x_original - x_imputed) ** 2
    mse = squared_errors[missing_mask].mean()
    rmse = torch.sqrt(mse).item()
    
    return rmse, x_imputed.numpy()



def standardize(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize data to zero mean and unit variance.
    
    Args:
        X: Data to standardize
        mean: Pre-computed mean (if None, computed from X)
        std: Pre-computed std (if None, computed from X)
        
    Returns:
        X_std: Standardized data
        mean: Mean used for standardization
        std: Std used for standardization
    """
    if mean is None:
        mean = np.nanmean(X, axis=0)
    if std is None:
        std = np.nanstd(X, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        
    X_std = (X - mean) / std
    
    return X_std, mean, std


def destandardize(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """
    Reverse standardization.
    
    Args:
        X: Standardized data
        mean: Mean used for standardization
        std: Std used for standardization
        
    Returns:
        X_orig: Data in original scale
    """
    return X * std + mean


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def impute(
    model,
    dataset,
    n_samples: int = 1000,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    solver = None # l1 or l2
) -> np.ndarray:
    """
    Impute missing values using a trained model.
    
    Takes a TensorDataset and returns the data with missing values imputed
    by the model while preserving original observed values.
    
    Args:
        model: Trained NotMIWAE or MIWAE model
        dataset: TensorDataset containing (x_filled, mask) tensors
        n_samples: Number of importance samples for imputation
        batch_size: Batch size for processing (for GPU/CPU efficiency)
        device: Device to use for computation
        verbose: Whether to print progress
        
    Returns:
        x_imputed: Data with missing values imputed (observed values preserved)
        
    Example:
        >>> from notmiwae_pytorch.utils import impute
        >>> from torch.utils.data import TensorDataset
        >>> 
        >>> # Create Dataset
        >>> dataset = TensorDataset(x_filled, mask)
        >>> 
        >>> # After training your model
        >>> x_imputed = impute(model, dataset, n_samples=1000, batch_size=64)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Extract tensors from dataset
    x_filled = dataset.tensors[0]
    mask = dataset.tensors[1]
    
    n = x_filled.shape[0]
    n_batches = (n + batch_size - 1) // batch_size
    
    model.eval()
    x_imputed_list = []
    
    with torch.no_grad():
        for i in range(0, n, batch_size):
            end_idx = min(i + batch_size, n)
            x_batch = x_filled[i:end_idx].to(device)
            s_batch = mask[i:end_idx].to(device)
            
            # Model's impute method already preserves observed values
            if solver:
                x_imp = model.impute(x_batch, s_batch, n_samples=n_samples, solver=solver)
            else:
                x_imp = model.impute(x_batch, s_batch, n_samples=n_samples)
            x_imputed_list.append(x_imp.cpu())
            
            if verbose:
                batch_idx = i // batch_size + 1
                print(f"Imputing batch {batch_idx}/{n_batches}")
    
    x_imputed = torch.cat(x_imputed_list, dim=0).numpy()
    
    return x_imputed
