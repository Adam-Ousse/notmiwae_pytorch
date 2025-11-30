"""
Example script demonstrating the not-MIWAE PyTorch implementation.

This script:
1. Loads the UCI Wine Quality dataset
2. Introduces MNAR missing values
3. Trains both not-MIWAE and MIWAE models
4. Compares imputation performance

Usage:
    python example.py
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import our implementation
from models import NotMIWAE, MIWAE
from trainer import Trainer, compute_imputation_rmse
from utils import set_seed, introduce_mnar_missing, standardize

# Configuration
SEED = 42
BATCH_SIZE = 64
HIDDEN_DIM = 128
N_SAMPLES = 20
N_EPOCHS = 100
LEARNING_RATE = 1e-3


def main():
    # Set seed for reproducibility
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\n" + "="*60)
    print("Loading UCI Wine Quality dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    data = pd.read_csv(url, sep=';')
    X = data.drop('quality', axis=1).values.astype(np.float32)
    
    N, D = X.shape
    print(f"Dataset: {N} samples, {D} features")
    
    # Standardize
    X_std, mean, std = standardize(X)
    
    # Shuffle
    perm = np.random.permutation(N)
    X_std = X_std[perm]
    
    # Introduce MNAR missing values
    print("\nIntroducing MNAR missing values...")
    X_nan, X_filled, mask = introduce_mnar_missing(X_std)
    print(f"Missing rate: {(1 - mask.mean()):.2%}")
    
    # Train/val split
    train_ratio = 0.8
    n_train = int(N * train_ratio)
    
    X_train_filled = torch.tensor(X_filled[:n_train], dtype=torch.float32)
    mask_train = torch.tensor(mask[:n_train], dtype=torch.float32)
    X_train_orig = torch.tensor(X_std[:n_train], dtype=torch.float32)
    
    X_val_filled = torch.tensor(X_filled[n_train:], dtype=torch.float32)
    mask_val = torch.tensor(mask[n_train:], dtype=torch.float32)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train_filled, mask_train, X_train_orig)
    val_dataset = TensorDataset(X_val_filled, mask_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Latent dimension
    latent_dim = D - 1
    
    # =====================================
    # Train not-MIWAE
    # =====================================
    print("\n" + "="*60)
    print("Training not-MIWAE...")
    print("="*60)
    
    notmiwae = NotMIWAE(
        input_dim=D,
        latent_dim=latent_dim,
        hidden_dim=HIDDEN_DIM,
        n_samples=N_SAMPLES,
        out_dist='gauss',
        missing_process='selfmasking_known'
    )
    
    trainer_notmiwae = Trainer(
        model=notmiwae,
        lr=LEARNING_RATE,
        device=device,
        log_dir='./runs',
        checkpoint_dir='./checkpoints'
    )
    
    history_notmiwae = trainer_notmiwae.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=N_EPOCHS,
        log_interval=20,
        save_best=True,
        early_stopping_patience=20,
        checkpoint_name='notmiwae_best.pt'
    )
    
    # =====================================
    # Train MIWAE (for comparison)
    # =====================================
    print("\n" + "="*60)
    print("Training MIWAE (baseline)...")
    print("="*60)
    
    miwae = MIWAE(
        input_dim=D,
        latent_dim=latent_dim,
        hidden_dim=HIDDEN_DIM,
        n_samples=N_SAMPLES,
        out_dist='gauss'
    )
    
    trainer_miwae = Trainer(
        model=miwae,
        lr=LEARNING_RATE,
        device=device,
        log_dir='./runs',
        checkpoint_dir='./checkpoints'
    )
    
    history_miwae = trainer_miwae.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=N_EPOCHS,
        log_interval=20,
        save_best=True,
        early_stopping_patience=20,
        checkpoint_name='miwae_best.pt'
    )
    
    # =====================================
    # Evaluate Imputation
    # =====================================
    print("\n" + "="*60)
    print("Evaluating Imputation Performance...")
    print("="*60)
    
    # Load best models
    trainer_notmiwae.load_checkpoint('notmiwae_best.pt')
    trainer_miwae.load_checkpoint('miwae_best.pt')
    
    # Compute RMSE
    rmse_notmiwae = compute_imputation_rmse(
        notmiwae, X_train_orig, X_train_filled, mask_train,
        n_samples=1000, device=device
    )
    
    rmse_miwae = compute_imputation_rmse(
        miwae, X_train_orig, X_train_filled, mask_train,
        n_samples=1000, device=device
    )
    
    # Mean imputation baseline
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed_mean = imputer.fit_transform(X_nan[:n_train])
    missing_mask = (1 - mask[:n_train]).astype(bool)
    rmse_mean = np.sqrt(np.mean((X_std[:n_train][missing_mask] - X_imputed_mean[missing_mask])**2))
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Method':<25} | {'RMSE':<10}")
    print("-"*40)
    print(f"{'Mean Imputation':<25} | {rmse_mean:.5f}")
    print(f"{'MIWAE':<25} | {rmse_miwae:.5f}")
    print(f"{'not-MIWAE':<25} | {rmse_notmiwae:.5f}")
    print("-"*40)
    
    improvement = (rmse_miwae - rmse_notmiwae) / rmse_miwae * 100
    print(f"\nnot-MIWAE improvement over MIWAE: {improvement:.2f}%")
    
    print("\nDone! Check ./runs for TensorBoard logs.")
    print("Run: tensorboard --logdir=./runs")


if __name__ == "__main__":
    main()
