"""
ARES - Acoustic Resonance & Entity-Specific Voice Verification System

A multi-layered hybrid system for voice authentication and deepfake detection.
"""

__version__ = '0.1.0'
__author__ = 'ARES Team'

from .core.processor import AudioProcessor
from .core.engine import ARESEngine
from .models.spectrotemporal.cnn_model import SpectroTemporalModel
from .models.acoustic.rnn_model import AcousticBiometricModel
from .models.linguistic.transformer_model import LinguisticProsodyModel
from .models.fusion.gradient_boosting import FusionModel 