"""
Test script to verify NotMIWAE works correctly with the refactored code.
"""

import torch
from models.notmiwae import NotMIWAE
from models.miwae import MIWAE

# Test parameters
batch_size = 32
input_dim = 10
latent_dim = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*60)
print(f"Testing NotMIWAE on {device}")
print("="*60)

# Create model
model = NotMIWAE(
    input_dim=input_dim,
    latent_dim=latent_dim,
    hidden_dim=64,
    n_samples=20,
    out_dist='gauss',
    missing_process='selfmasking'
).to(device)

print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Create synthetic data
x = torch.randn(batch_size, input_dim).to(device)
s = torch.ones(batch_size, input_dim).to(device)

# Introduce missing values (30% missing)
missing_mask = torch.rand(batch_size, input_dim).to(device) < 0.3
s[missing_mask] = 0
x[missing_mask] = 0

print(f"\nData shape: {x.shape}")
print(f"Missing rate: {(1 - s.mean()):.2%}")

# Test forward pass
print("\n" + "="*60)
print("Testing forward pass...")
print("="*60)

output = model(x, s)
print(f"✓ Forward pass successful")
print(f"  Loss: {output['loss'].item():.4f}")
print(f"  ELBO: {output['elbo'].item():.4f}")
print(f"  MIWAE ELBO: {output['miwae_elbo'].item():.4f}")

# Test imputation with L2 (mean)
print("\n" + "="*60)
print("Testing imputation with L2 solver (mean)...")
print("="*60)

x_imputed_l2 = model.impute(x, s, n_samples=100, solver='l2')
print(f"✓ L2 imputation successful")
print(f"  Imputed shape: {x_imputed_l2.shape}")
print(f"  Imputation range: [{x_imputed_l2.min():.3f}, {x_imputed_l2.max():.3f}]")

# Test imputation with L1 (median)
print("\n" + "="*60)
print("Testing imputation with L1 solver (median)...")
print("="*60)

x_imputed_l1 = model.impute(x, s, n_samples=100, solver='l1')
print(f"✓ L1 imputation successful")
print(f"  Imputed shape: {x_imputed_l1.shape}")
print(f"  Imputation range: [{x_imputed_l1.min():.3f}, {x_imputed_l1.max():.3f}]")

# Compare L1 vs L2
diff = (x_imputed_l1 - x_imputed_l2).abs().mean()
print(f"\nMean absolute difference between L1 and L2: {diff:.4f}")

# Test missing process interpretation
print("\n" + "="*60)
print("Testing missing process interpretation...")
print("="*60)

interpretation = model.interpret_missing_process(verbose=False)
print(f"✓ Interpretation successful")
print(f"  Process type: {interpretation['process_type']}")
print(f"  Number of features: {len(interpretation['interpretations'])}")

print("\n" + "="*60)
print("All tests passed! ✓")
print("="*60)
