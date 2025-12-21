"""
Test suite for ARES Engine end-to-end pipeline.
"""

import pytest
import numpy as np
import os
import tempfile
import soundfile as sf

from ares.core.engine import ARESEngine
from ares.core.config import get_config


class TestARESEngine:
    """Tests for ARESEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create an engine instance."""
        return ARESEngine()
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create a sample audio file for testing."""
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create a more realistic audio signal (multiple frequencies + noise)
        audio = (
            0.3 * np.sin(2 * np.pi * 220 * t) +  # F0
            0.2 * np.sin(2 * np.pi * 440 * t) +  # Harmonic
            0.1 * np.sin(2 * np.pi * 660 * t) +  # Harmonic
            0.05 * np.random.randn(len(t))       # Noise
        )
        
        audio = audio / np.abs(audio).max() * 0.8  # Normalize
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio.astype(np.float32), sr)
            yield f.name
        os.unlink(f.name)
    
    def test_engine_initialization(self, engine):
        """Test that engine initializes correctly."""
        assert engine.audio_processor is not None
        assert engine.spectro_model is not None
        assert engine.acoustic_model is not None
        assert engine.linguistic_model is not None
        assert engine.fusion_model is not None
    
    def test_analyze_file_basic(self, engine, sample_audio_file):
        """Test basic file analysis."""
        result = engine.analyze_file(sample_audio_file)
        
        assert isinstance(result, dict)
        assert 'classification' in result
        assert result['classification'] in ['Authentic', 'Cloned (AI-Generated)', 'Error']
    
    def test_analyze_file_structure(self, engine, sample_audio_file):
        """Test that analysis result has correct structure."""
        result = engine.analyze_file(sample_audio_file)
        
        # Check required fields
        assert 'classification' in result
        assert 'confidence_score' in result
        assert 'evidence' in result
        
        # Check evidence structure
        evidence = result['evidence']
        assert 'spectro_temporal' in evidence
        assert 'acoustic_biometric' in evidence
        assert 'linguistic_prosodic' in evidence
        assert 'fusion' in evidence
        
        # Check each stream has score and findings
        for stream in ['spectro_temporal', 'acoustic_biometric', 'linguistic_prosodic']:
            assert 'score' in evidence[stream]
            assert 'findings' in evidence[stream]
            assert 0.0 <= evidence[stream]['score'] <= 1.0
    
    def test_analyze_file_scores_in_range(self, engine, sample_audio_file):
        """Test that all scores are in valid range."""
        result = engine.analyze_file(sample_audio_file)
        
        assert 0.0 <= result['confidence_score'] <= 100.0
        assert 0.0 <= result['raw_score'] <= 1.0
        assert result['analysis_time'] > 0
    
    def test_save_report(self, engine, sample_audio_file):
        """Test report saving."""
        result = engine.analyze_file(sample_audio_file)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            report_path = engine.save_report(result, f.name)
        
        assert os.path.exists(report_path)
        
        # Verify it's valid JSON
        import json
        with open(report_path) as f:
            loaded = json.load(f)
        
        assert loaded['classification'] == result['classification']
        
        os.unlink(report_path)
    
    def test_error_handling_invalid_file(self, engine):
        """Test error handling for invalid file."""
        result = engine.analyze_file('/nonexistent/file.wav')
        
        assert result['classification'] == 'Error'
        assert 'error' in result['evidence']


class TestModels:
    """Tests for individual model components."""
    
    def test_spectro_model_analyze(self):
        """Test SpectroTemporalModel analysis."""
        from ares.models.spectrotemporal.cnn_model import SpectroTemporalModel
        
        model = SpectroTemporalModel()
        
        # Create dummy inputs
        mel_spec = np.random.randn(128, 100)
        cqt = np.random.randn(84, 100)
        
        result = model.analyze(mel_spectrogram=mel_spec, constant_q_transform=cqt)
        
        assert 'score' in result
        assert 'findings' in result
        assert 0.0 <= result['score'] <= 1.0
    
    def test_acoustic_model_analyze(self):
        """Test AcousticBiometricModel analysis."""
        from ares.models.acoustic.rnn_model import AcousticBiometricModel
        
        model = AcousticBiometricModel()
        
        # Create dummy features
        features = {
            'mfccs': np.random.randn(120, 100),  # 40 MFCCs + delta + delta-delta
            'pitch_contour': {
                'f0': np.random.uniform(100, 300, 100),
                'voiced_flag': np.ones(100),
                'voiced_probs': np.ones(100)
            },
            'jitter': 0.01,
            'shimmer': 0.03
        }
        
        result = model.analyze(features)
        
        assert 'score' in result
        assert 'findings' in result
        assert 0.0 <= result['score'] <= 1.0
    
    def test_linguistic_model_analyze(self):
        """Test LinguisticProsodyModel analysis."""
        from ares.models.linguistic.transformer_model import LinguisticProsodyModel
        
        model = LinguisticProsodyModel()
        
        # Create dummy waveform
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        waveform = np.sin(2 * np.pi * 440 * t)
        
        result = model.analyze(waveform, sr)
        
        assert 'score' in result
        assert 'findings' in result
        assert 0.0 <= result['score'] <= 1.0
    
    def test_fusion_model_analyze(self):
        """Test FusionModel analysis."""
        from ares.models.fusion.gradient_boosting import FusionModel
        
        model = FusionModel()
        
        result = model.analyze(
            stream_a_score=0.3,
            stream_b_score=0.5,
            stream_c_score=0.4
        )
        
        assert 'score' in result
        assert 'decision_factors' in result
        assert 0.0 <= result['score'] <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
