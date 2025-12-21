"""
ARES Configuration Module.

Centralized configuration for the ARES voice verification system.
Supports both default values and environment variable overrides.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    n_mfcc: int = 40
    max_duration: float = 30.0  # Maximum audio duration in seconds
    min_duration: float = 0.5   # Minimum audio duration in seconds


@dataclass
class AugmentationConfig:
    """Audio augmentation configuration for robustness."""
    enabled: bool = True
    noise_prob: float = 0.3
    noise_min_snr_db: float = 3.0
    noise_max_snr_db: float = 30.0
    pitch_shift_prob: float = 0.2
    pitch_shift_semitones: float = 2.0
    time_stretch_prob: float = 0.2
    time_stretch_min_rate: float = 0.9
    time_stretch_max_rate: float = 1.1
    codec_prob: float = 0.2  # Simulate MP3/AAC compression


@dataclass
class Wav2Vec2Config:
    """Wav2Vec2/HuBERT model configuration."""
    model_name: str = "facebook/wav2vec2-base"  # Options: wav2vec2-base, wav2vec2-large, hubert-base-ls960
    freeze_encoder: bool = True  # Freeze encoder for CPU efficiency
    use_weighted_layer_sum: bool = True
    hidden_size: int = 768
    num_hidden_layers: int = 12


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Stream A: Spectro-Temporal (Wav2Vec2)
    wav2vec2: Wav2Vec2Config = field(default_factory=Wav2Vec2Config)
    
    # Stream B: Acoustic-Biometric
    acoustic_hidden_size: int = 256
    acoustic_num_layers: int = 2
    acoustic_dropout: float = 0.3
    acoustic_bidirectional: bool = True
    
    # Stream C: Linguistic-Prosodic
    linguistic_hidden_size: int = 256
    use_whisper: bool = True
    whisper_model: str = "tiny"  # tiny, base, small for CPU
    
    # Fusion
    fusion_attention_heads: int = 4
    fusion_hidden_size: int = 128
    fusion_dropout: float = 0.2
    use_attention_fusion: bool = True


@dataclass
class TrainingConfig:
    """Training configuration (for fine-tuning)."""
    batch_size: int = 8  # Small batch for CPU
    learning_rate: float = 1e-4
    epochs: int = 50
    early_stopping_patience: int = 10
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4  # Effective batch size = 32
    device: str = "cpu"
    mixed_precision: bool = False  # Disable for CPU
    num_workers: int = 0  # Single-threaded for CPU stability


@dataclass
class InferenceConfig:
    """Inference configuration."""
    confidence_threshold: float = 0.5
    use_calibration: bool = True
    calibration_temperature: float = 1.0
    return_embeddings: bool = False
    verbose: bool = False


@dataclass
class ARESConfig:
    """Main ARES configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Paths
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "models" / "weights")
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "ares")
    
    # Version
    version: str = "2.0.0"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "ARESConfig":
        """Create config with environment variable overrides."""
        config = cls()
        
        # Override from environment variables
        if os.getenv("ARES_SAMPLE_RATE"):
            config.audio.sample_rate = int(os.getenv("ARES_SAMPLE_RATE"))
        if os.getenv("ARES_DEVICE"):
            config.training.device = os.getenv("ARES_DEVICE")
        if os.getenv("ARES_WAV2VEC_MODEL"):
            config.model.wav2vec2.model_name = os.getenv("ARES_WAV2VEC_MODEL")
        if os.getenv("ARES_WHISPER_MODEL"):
            config.model.whisper_model = os.getenv("ARES_WHISPER_MODEL")
            
        return config


# Global default configuration
default_config = ARESConfig()


def get_config() -> ARESConfig:
    """Get the default ARES configuration."""
    return default_config
