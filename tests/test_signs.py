"""
Test signs parameter for selfmasking_known_signs missing process.
"""

import torch
import numpy as np
from models.notmiwae import NotMIWAE

def test_signs_parameter():
    """Test that signs parameter correctly controls directional missingness."""
    
    # Setup
    input_dim = 4
    n_samples = 100
    
    # Create test data with clear pattern:
    # Features 0, 1: values in [0, 1]
    # Features 2, 3: values in [0, 1]
    torch.manual_seed(42)
    x = torch.rand(n_samples, input_dim)
    
    # Define signs:
    # Feature 0: +1 (high values missing)
    # Feature 1: +1 (high values missing)
    # Feature 2: -1 (low values missing)
    # Feature 3: -1 (low values missing)
    signs = torch.tensor([1.0, 1.0, -1.0, -1.0])
    
    # Create model with custom signs
    model = NotMIWAE(
        input_dim=input_dim,
        latent_dim=10,
        hidden_dim=64,
        n_samples=20,
        missing_process='selfmasking_known_signs',
        feature_names=[f'Feature_{i}' for i in range(input_dim)],
        signs=signs
    )
    
    # Manually set W and b for predictable behavior
    # Set W to make effects strong (after softplus, W ≈ 2)
    model.missing_model.W.data.fill_(2.0)
    model.missing_model.b.data.fill_(0.5)  # threshold at 0.5
    
    # Test 1: Check that signs are stored correctly
    print("Shape of signs buffer:", model.missing_model.signs.shape)
    print("Shape of W:", model.missing_model.W.shape)
    print("Shape of b:", model.missing_model.b.shape)
    
    stored_signs = model.missing_model.signs.squeeze()
    assert torch.allclose(stored_signs, signs), "Signs not stored correctly"
    print("✓ Signs stored correctly:", stored_signs.tolist())
    
    # Test 2: Check forward pass with high values
    x_high = torch.ones(1, input_dim) * 0.9  # All values high (> 0.5)
    logits_high = model.missing_model(x_high)
    
    print("\n✓ Forward pass test:")
    print(f"  Input shape: {x_high.shape}")
    print(f"  Output shape: {logits_high.shape}")
    print(f"  All logits: {logits_high.squeeze().tolist()}")
    
    # Flatten to [4] for easier indexing
    logits_flat = logits_high.view(-1)
    print(f"  Features 0,1 (+1 sign): {logits_flat[:2].tolist()}")
    print(f"  Features 2,3 (-1 sign): {logits_flat[2:].tolist()}")
    
    # For features with sign=+1: slope = -1 * W_positive
    # logit = -W * (0.9 - 0.5) = negative (more likely missing)
    assert (logits_flat[:2] < 0).all(), "Features 0,1 should have negative logits (high -> missing)"
    
    # For features with sign=-1: slope = +1 * W_positive  
    # logit = +W * (0.9 - 0.5) = positive (less likely missing)
    assert (logits_flat[2:] > 0).all(), "Features 2,3 should have positive logits (high -> observed)"
    
    # Test 3: Check forward pass with low values
    x_low = torch.ones(1, input_dim) * 0.1  # All values low (< 0.5)
    logits_low = model.missing_model(x_low)
    
    logits_flat_low = logits_low.view(-1)
    print("\n✓ Logits for low values (0.1):")
    print(f"  Features 0,1 (+1 sign): {logits_flat_low[:2].tolist()}")
    print(f"  Features 2,3 (-1 sign): {logits_flat_low[2:].tolist()}")
    
    # For features with sign=+1: slope = -1 * W_positive
    # logit = -W * (0.1 - 0.5) = positive (less likely missing)
    assert (logits_flat_low[:2] > 0).all(), "Features 0,1 should have positive logits (low -> observed)"
    
    # For features with sign=-1: slope = +1 * W_positive
    # logit = +W * (0.1 - 0.5) = negative (more likely missing)
    assert (logits_flat_low[2:] < 0).all(), "Features 2,3 should have negative logits (low -> missing)"
    
    # Test 4: Check interpret method
    print("\n✓ Interpretation:")
    interp = model.missing_model.interpret(verbose=False)
    
    assert 'signs' in interp, "Signs should be in interpretation"
    print(f"  Signs in results: {interp['signs'].tolist()}")
    
    # Verify directions are correctly interpreted
    for i, item in enumerate(interp['interpretations']):
        print(f"  {item['feature']}: {item['direction']}")
        
        # Features 0,1 should be "high values missing"
        # Features 2,3 should be "low values missing"
        if i < 2:
            assert item['direction'] == 'high values missing', f"Feature {i} should be high->missing"
        else:
            assert item['direction'] == 'low values missing', f"Feature {i} should be low->missing"
    
    print("\n✅ All tests passed!")

if __name__ == '__main__':
    test_signs_parameter()
