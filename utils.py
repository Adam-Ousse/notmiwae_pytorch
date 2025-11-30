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
    verbose: bool = False
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


def introduce_mnar_missing(
    X: np.ndarray,
    missing_rate: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Introduce MNAR (Missing Not At Random) missing values.
    
    Following the experimental setup in the not-MIWAE paper:
    Values above the mean in the first D/2 dimensions are missing.
    
    Args:
        X: Complete data of shape (N, D)
        missing_rate: Not used directly, the mechanism is deterministic
        
    Returns:
        X_nan: Data with NaN for missing values
        X_filled: Data with 0 for missing values  
        mask: Binary mask (1=observed, 0=missing)
    """
    N, D = X.shape
    X_nan = X.copy()
    
    # MNAR in first D/2 dimensions
    n_miss_cols = D // 2
    mean = np.mean(X_nan[:, :n_miss_cols], axis=0)
    
    # Values above mean are missing
    mask_missing = X_nan[:, :n_miss_cols] > mean
    X_nan[:, :n_miss_cols][mask_missing] = np.nan
    
    # Create filled version and mask
    X_filled = X_nan.copy()
    X_filled[np.isnan(X_nan)] = 0
    mask = (~np.isnan(X_nan)).astype(np.float32)
    
    return X_nan, X_filled, mask


def introduce_mcar_missing(
    X: np.ndarray,
    missing_rate: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Introduce MCAR (Missing Completely At Random) missing values.
    
    Args:
        X: Complete data of shape (N, D)
        missing_rate: Proportion of values to be missing
        
    Returns:
        X_nan: Data with NaN for missing values
        X_filled: Data with 0 for missing values
        mask: Binary mask (1=observed, 0=missing)
    """
    N, D = X.shape
    
    # Create random mask
    mask = np.random.binomial(1, 1 - missing_rate, size=(N, D)).astype(np.float32)
    
    X_nan = X.copy()
    X_nan[mask == 0] = np.nan
    
    X_filled = X.copy()
    X_filled[mask == 0] = 0
    
    return X_nan, X_filled, mask


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
