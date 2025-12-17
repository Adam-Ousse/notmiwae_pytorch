"""Test custom missing process functionality."""
import torch
import torch.nn as nn
from notmiwae_pytorch import BaseMissingProcess, MissingProcess, SelfMaskingKnownSignsProcess
from notmiwae_pytorch import NotMIWAE


# Test 1: Built-in missing process (backward compatible)
print("=== Test 1: Built-in missing process ===")
model1 = NotMIWAE(input_dim=10, missing_process='selfmasking_known_signs')
print(f"Missing process type: {model1.missing_process_type}")
print(f"Missing model class: {model1.missing_model.__class__.__name__}")

x = torch.rand(4, 10)
s = torch.ones(4, 10)
out = model1(x, s)
print(f"Loss: {out['loss'].item():.4f}")


# Test 2: Custom missing process
print("\n=== Test 2: Custom missing process ===")

class ClippingMissingProcess(BaseMissingProcess):
    """Custom: bright pixels more likely missing (for images)."""
    
    def __init__(self, input_dim, threshold=0.75, strength=50.0, **kwargs):
        super().__init__(input_dim, **kwargs)
        self.threshold = nn.Parameter(torch.full((1, 1, input_dim), threshold))
        self.strength = nn.Parameter(torch.full((1, 1, input_dim), strength))
        
    def forward(self, x):
        # Higher values -> lower logit -> more likely missing
        return -self.strength * (x - self.threshold)
    
    def interpret(self, verbose=True):
        thresh = self.threshold.detach().squeeze().cpu().numpy()
        strength = self.strength.detach().squeeze().cpu().numpy()
        results = {
            'process_type': 'clipping',
            'threshold_mean': float(thresh.mean()),
            'strength_mean': float(strength.mean())
        }
        if verbose:
            print(f"Clipping threshold: {results['threshold_mean']:.3f}")
            print(f"Clipping strength: {results['strength_mean']:.3f}")
        return results


custom_missing = ClippingMissingProcess(input_dim=1024, threshold=0.75, strength=50.0)
model2 = NotMIWAE(input_dim=1024, missing_process=custom_missing, architecture='CNN')
print(f"Missing process type: {model2.missing_process_type}")
print(f"Missing model class: {model2.missing_model.__class__.__name__}")

# Test forward pass
x = torch.rand(4, 1024)
s = torch.ones(4, 1024)
output = model2(x, s)
print(f"Forward pass: loss={output['loss'].item():.4f}")

# Test interpret
print("\nInterpretation:")
model2.interpret_missing_process()

# Test impute
x_imputed = model2.impute(x, s, n_samples=10)
print(f"\nImputation shape: {x_imputed.shape}")


# Test 3: Verify compute_sensitivity works
print("\n=== Test 3: Compute sensitivity ===")
sensitivity = model2.compute_missing_sensitivity(x[:2])
print(f"Sensitivity shape: {sensitivity.shape}")


print("\n=== All tests passed! ===")
