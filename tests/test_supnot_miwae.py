from notmiwae_pytorch import SupNotMIWAE, SupMIWAE, PredictionHead

# Supervised Not-MIWAE (accounts for MNAR missingness)
model = SupNotMIWAE(
    input_dim=10,
    latent_dim=5,
    missing_process='selfmasking',
    y_dim=1,  # or n_classes
    task='regression'  # or 'classification'
)

# Supervised MIWAE baseline (ignores missingness mechanism)
baseline = SupMIWAE(
    input_dim=10,
    y_dim=1,
    task='regression'
)