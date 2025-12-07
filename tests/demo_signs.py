"""
Demo: Using the signs parameter for directional missingness control.

This demonstrates how to specify which direction (high or low values) 
should cause missingness for each feature using the selfmasking_known_signs mechanism.
"""

import torch
import numpy as np
from models.notmiwae import NotMIWAE

def demo_directional_missingness():
    """
    Demonstrate directional missingness control with signs parameter.
    """
    
    print("=" * 70)
    print("Directional Missingness Control Demo")
    print("=" * 70)
    
    # Setup scenario: 4 features with different missingness patterns
    input_dim = 4
    feature_names = [
        'Temperature',      # Feature 0: High temp -> missing (sensor fails)
        'Pressure',         # Feature 1: High pressure -> missing (sensor saturates)
        'Humidity',         # Feature 2: Low humidity -> missing (sensor threshold)
        'Wind_Speed'        # Feature 3: Low wind -> missing (anemometer stalls)
    ]
    
    # Define directional patterns:
    # +1.0 = High values more likely to be missing
    # -1.0 = Low values more likely to be missing
    signs = torch.tensor([
        +1.0,  # Temperature: high -> missing
        +1.0,  # Pressure: high -> missing
        -1.0,  # Humidity: low -> missing
        -1.0   # Wind_Speed: low -> missing
    ])
    
    print("\n📋 Scenario Setup:")
    print("-" * 70)
    for i, (name, sign) in enumerate(zip(feature_names, signs)):
        direction = "High→Missing" if sign > 0 else "Low→Missing"
        print(f"  {name:15s}: {direction:14s} (sign={sign:+.1f})")
    
    # Create model with custom signs
    model = NotMIWAE(
        input_dim=input_dim,
        latent_dim=10,
        hidden_dim=64,
        n_samples=20,
        missing_process='selfmasking_known_signs',
        feature_names=feature_names,
        signs=signs
    )
    
    # Manually set parameters for demonstration
    # Use strong W values (after softplus ≈ 3.0) and threshold at 0.5
    model.missing_model.W.data.fill_(2.5)
    model.missing_model.b.data.fill_(0.5)
    
    print("\n🔬 Model Parameters:")
    print("-" * 70)
    print(f"  W (magnitude): ~{torch.nn.functional.softplus(model.missing_model.W[0,0,0]).item():.2f}")
    print(f"  b (threshold): {model.missing_model.b[0,0,0].item():.2f}")
    
    # Test Case 1: Extreme high values (0.95)
    print("\n📊 Test Case 1: HIGH VALUES (0.95)")
    print("-" * 70)
    x_high = torch.ones(1, input_dim) * 0.95
    logits_high = model.missing_model(x_high).view(-1)
    probs_high = torch.sigmoid(logits_high)
    
    for i, (name, logit, prob) in enumerate(zip(feature_names, logits_high, probs_high)):
        expected = "Missing" if signs[i] > 0 else "Observed"
        print(f"  {name:15s}: logit={logit:+.3f}, P(observed)={prob:.3f} → {expected}")
    
    # Test Case 2: Extreme low values (0.05)
    print("\n📊 Test Case 2: LOW VALUES (0.05)")
    print("-" * 70)
    x_low = torch.ones(1, input_dim) * 0.05
    logits_low = model.missing_model(x_low).view(-1)
    probs_low = torch.sigmoid(logits_low)
    
    for i, (name, logit, prob) in enumerate(zip(feature_names, logits_low, probs_low)):
        expected = "Observed" if signs[i] > 0 else "Missing"
        print(f"  {name:15s}: logit={logit:+.3f}, P(observed)={prob:.3f} → {expected}")
    
    # Test Case 3: Mixed values
    print("\n📊 Test Case 3: MIXED VALUES")
    print("-" * 70)
    x_mixed = torch.tensor([[0.9, 0.1, 0.9, 0.1]])  # High, Low, High, Low
    logits_mixed = model.missing_model(x_mixed).view(-1)
    probs_mixed = torch.sigmoid(logits_mixed)
    
    for i, (name, val, logit, prob) in enumerate(zip(feature_names, x_mixed[0], logits_mixed, probs_mixed)):
        val_type = "High" if val > 0.5 else "Low"
        if (signs[i] > 0 and val > 0.5) or (signs[i] < 0 and val < 0.5):
            expected = "Missing"
        else:
            expected = "Observed"
        print(f"  {name:15s}: value={val:.2f} ({val_type:4s}), P(obs)={prob:.3f} → {expected}")
    
    # Interpretation
    print("\n🔍 Model Interpretation:")
    print("-" * 70)
    interp = model.missing_model.interpret(verbose=True)
    
    print("\n✅ Summary:")
    print("-" * 70)
    print("  The signs parameter allows precise control over directional missingness.")
    print("  Each feature can independently have high-values-missing (+1) or")
    print("  low-values-missing (-1) patterns, matching real-world sensor behaviors.")
    print("=" * 70)

if __name__ == '__main__':
    demo_directional_missingness()
