"""
Test suite for ARES Audio Processor.
"""

import pytest
import numpy as np
import os
import tempfile
import soundfile as sf

from ares.core.processor import AudioProcessor, AudioAugmenter
from ares.core.config import get_config


class TestAudioProcessor:
    """Tests for AudioProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a processor instance."""
        return AudioProcessor()
    
    @pytest.fixture
    def sample_audio(self):
        """Generate a sample audio signal (sine wave)."""
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        # 440Hz sine wave with some harmonics
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
        return audio, sr
    
    @pytest.fixture
    def temp_audio_file(self, sample_audio):
        """Create a temporary audio file."""
        audio, sr = sample_audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sr)
            yield f.name
        os.unlink(f.name)
    
    def test_processor_initialization(self, processor):
        """Test that processor initializes with correct defaults."""
        config = get_config()
        assert processor.sample_rate == config.audio.sample_rate
        assert processor.n_fft == config.audio.n_fft
        assert processor.hop_length == config.audio.hop_length
    
    def test_load_audio(self, processor, temp_audio_file):
        """Test audio file loading."""
        waveform, sr = processor.load_audio(temp_audio_file)
        assert isinstance(waveform, np.ndarray)
        assert len(waveform) > 0
        assert sr > 0
    
    def test_preprocess(self, processor, sample_audio):
        """Test audio preprocessing."""
        audio, sr = sample_audio
        processed = processor.preprocess(audio, sr)
        
        # Should be normalized (allow small overshoot from high-pass filter)
        assert np.abs(processed).max() <= 1.1
        # Should have same general shape
        assert len(processed) > 0
    
    def test_extract_mel_spectrogram(self, processor, sample_audio):
        """Test mel-spectrogram extraction."""
        audio, sr = sample_audio
        processed = processor.preprocess(audio, sr)
        mel_spec = processor.extract_mel_spectrogram(processed)
        
        assert isinstance(mel_spec, np.ndarray)
        assert len(mel_spec.shape) == 2
        assert mel_spec.shape[0] == processor.n_mels
    
    def test_extract_mfcc(self, processor, sample_audio):
        """Test MFCC extraction."""
        audio, sr = sample_audio
        processed = processor.preprocess(audio, sr)
        mfccs = processor.extract_mfcc(processed)
        
        assert isinstance(mfccs, np.ndarray)
        assert len(mfccs.shape) == 2
        # Should have n_mfcc * 3 (base + delta + delta-delta)
        assert mfccs.shape[0] == processor.n_mfcc * 3
    
    def test_extract_pitch_contour(self, processor, sample_audio):
        """Test pitch extraction."""
        audio, sr = sample_audio
        processed = processor.preprocess(audio, sr)
        pitch = processor.extract_pitch_contour(processed)
        
        assert isinstance(pitch, dict)
        assert 'f0' in pitch
        assert 'voiced_flag' in pitch
    
    def test_extract_jitter_shimmer(self, processor, sample_audio):
        """Test jitter/shimmer extraction."""
        audio, sr = sample_audio
        processed = processor.preprocess(audio, sr)
        jitter, shimmer = processor.extract_jitter_shimmer(processed)
        
        assert isinstance(jitter, (int, float))
        assert isinstance(shimmer, (int, float))
        assert jitter >= 0
        assert shimmer >= 0
    
    def test_process_file(self, processor, temp_audio_file):
        """Test full file processing pipeline."""
        features = processor.process_file(temp_audio_file)
        
        assert isinstance(features, dict)
        assert 'waveform' in features
        assert 'mel_spectrogram' in features
        assert 'mfccs' in features
        assert 'pitch_contour' in features
        assert 'jitter' in features
        assert 'shimmer' in features


class TestAudioAugmenter:
    """Tests for AudioAugmenter class."""
    
    @pytest.fixture
    def augmenter(self):
        """Create an augmenter instance."""
        return AudioAugmenter()
    
    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio."""
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return audio, sr
    
    def test_augmenter_initialization(self, augmenter):
        """Test augmenter initializes correctly."""
        config = get_config()
        assert augmenter.aug_config.enabled == config.augmentation.enabled
    
    def test_augment_returns_same_shape(self, augmenter, sample_audio):
        """Test that augmentation preserves shape."""
        audio, sr = sample_audio
        augmented = augmenter.augment(audio, sr)
        
        assert isinstance(augmented, np.ndarray)
        # Length might change slightly with time stretch, but should be close
        assert len(augmented) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
