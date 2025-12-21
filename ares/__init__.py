"""
ARES - Acoustic Resonance & Entity-Specific Voice Verification System v2.0

A multi-layered hybrid system for voice authentication and deepfake detection.
"""

__version__ = '2.0.0'
__author__ = 'ARES Team'

from .core.config import ARESConfig, get_config
from .core.processor import AudioProcessor
from .core.engine import ARESEngine

# Models
from .models.spectrotemporal.cnn_model import SpectroTemporalModel
from .models.acoustic.rnn_model import AcousticBiometricModel
from .models.linguistic.transformer_model import LinguisticProsodyModel
from .models.fusion.gradient_boosting import FusionModel

__all__ = [
    'ARESConfig',
    'get_config',
    'AudioProcessor',
    'ARESEngine',
    'SpectroTemporalModel',
    'AcousticBiometricModel',
    'LinguisticProsodyModel',
    'FusionModel'
]