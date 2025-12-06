"""
Models module for not-MIWAE PyTorch implementation.
"""

from .notmiwae import NotMIWAE
from .miwae import MIWAE
from .base import (
    Encoder, Encoder_CNN, 
    GaussianDecoder, GaussianDecoder_CNN, BernoulliDecoder,
    # Missing process classes
    BaseMissingProcess,
    SelfMaskingProcess,
    SelfMaskingKnownSignsProcess,
    LinearMissingProcess,
    NonlinearMissingProcess,
    MissingProcess,  # Factory function for backward compatibility
)

__all__ = [
    'NotMIWAE',
    'MIWAE', 
    'Encoder',
    'Encoder_CNN',
    'GaussianDecoder',
    'GaussianDecoder_CNN',
    'BernoulliDecoder',
    # Missing process
    'BaseMissingProcess',
    'SelfMaskingProcess',
    'SelfMaskingKnownSignsProcess',
    'LinearMissingProcess',
    'NonlinearMissingProcess',
    'MissingProcess',
]
