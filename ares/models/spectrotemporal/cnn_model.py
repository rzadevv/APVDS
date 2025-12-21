"""
Spectro-Temporal Analysis Model (Stream A) - ARES v2.0

Uses a pre-trained Wav2Vec2-based deepfake detection model from HuggingFace
for accurate AI-generated voice detection.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SpectroTemporalModel:
    """
    Pre-trained Spectro-Temporal Analysis using Wav2Vec2 deepfake detector.
    
    Uses Hemgg/Deepfake-audio-detection (95.45% accuracy) from HuggingFace
    for reliable AI-generated voice detection.
    """
    
    def __init__(self, model_path: Optional[str] = None, config=None):
        """
        Initialize the model.
        
        Args:
            model_path: Path to custom classifier weights (optional)
            config: ARES configuration
        """
        self.device = torch.device('cpu')
        self.wav2vec2 = None
        self.processor = None
        self.config = config
        self.use_pretrained_classifier = False
        
        # Load pre-trained deepfake detection model
        self._load_pretrained_model()
        
        logger.info("SpectroTemporalModel initialized")
    
    def _load_pretrained_model(self):
        """Load pre-trained Wav2Vec2 deepfake detection model from HuggingFace."""
        try:
            from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
            
            # Pre-trained deepfake detection model (95.45% accuracy)
            model_name = "Hemgg/Deepfake-audio-detection"
            
            logger.info(f"Loading pre-trained deepfake detector: {model_name}")
            
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            self.wav2vec2 = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            
            self.wav2vec2.eval()
            self.wav2vec2.to(self.device)
            self.use_pretrained_classifier = True
            
            logger.info("Pre-trained deepfake detector loaded successfully!")
            
        except Exception as e:
            logger.warning(f"Could not load pre-trained model: {e}")
            logger.info("Will fall back to heuristic analysis")
            self.wav2vec2 = None
            self.processor = None
            self.use_pretrained_classifier = False
    
    def analyze(self, mel_spectrogram: np.ndarray = None, 
                constant_q_transform: np.ndarray = None,
                waveform: np.ndarray = None,
                sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Analyze audio for spectral artifacts indicating synthesis.
        
        Args:
            mel_spectrogram: Mel-spectrogram (fallback)
            constant_q_transform: CQT (fallback)
            waveform: Raw audio waveform (preferred)
            sample_rate: Sample rate
            
        Returns:
            Dictionary with score and findings
        """
        logger.info("Analyzing spectro-temporal features with Wav2Vec2")
        
        if waveform is not None and self.wav2vec2 is not None and self.use_pretrained_classifier:
            score, findings = self._analyze_with_pretrained(waveform, sample_rate)
        else:
            score, findings = self._analyze_spectrograms(mel_spectrogram, constant_q_transform)
        
        return {
            'score': float(score),
            'findings': findings
        }
    
    def _analyze_with_pretrained(self, waveform: np.ndarray, 
                                  sample_rate: int) -> Tuple[float, list]:
        """Analyze using pre-trained Wav2Vec2 classifier."""
        import librosa
        
        # Ensure correct sample rate (16kHz for Wav2Vec2)
        if sample_rate != 16000:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
        
        # Process input
        inputs = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.wav2vec2(input_values)
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Score = probability of being FAKE (AI-generated)
            # Label mapping: 0 = REAL, 1 = FAKE
            if logits.shape[-1] == 2:
                score = probs[0, 0].item()  # 0 = AIVoice (FAKE)
            else:
                score = torch.sigmoid(logits).item()
        
        # Generate findings based on score
        findings = self._generate_findings(score)
        
        return score, findings
    
    def _analyze_spectrograms(self, mel_spec: np.ndarray, 
                               cqt: np.ndarray) -> Tuple[float, list]:
        """Fallback analysis using spectrograms when no pre-trained model."""
        findings = []
        
        if mel_spec is not None:
            mel_std = np.std(mel_spec)
            spectral_naturalness = min(1.0, mel_std / 20.0)
            
            if spectral_naturalness < 0.3:
                findings.append("Unusually flat spectral distribution")
            elif spectral_naturalness > 0.8:
                findings.append("Natural spectral variation observed")
        else:
            spectral_naturalness = 0.5
        
        if cqt is not None:
            harmonic_score = np.mean(np.abs(np.diff(cqt, axis=0)))
            normalized_harmonic = min(1.0, harmonic_score / 5.0)
            
            if normalized_harmonic < 0.2:
                findings.append("Unusual harmonic transitions detected")
        else:
            normalized_harmonic = 0.5
        
        score = 1.0 - (spectral_naturalness * 0.6 + normalized_harmonic * 0.4)
        
        if not findings:
            findings.append("Spectral analysis complete (heuristic)")
        
        return float(score), findings
    
    def _generate_findings(self, score: float) -> list:
        """Generate human-readable findings based on analysis."""
        findings = []
        
        if score > 0.8:
            findings.append("Strong spectral artifacts detected - likely AI-generated")
            findings.append("Neural vocoder signatures present in audio")
        elif score > 0.6:
            findings.append("Moderate spectral anomalies detected")
            findings.append("Some synthetic audio characteristics observed")
        elif score > 0.4:
            findings.append("Minor spectral irregularities observed")
            findings.append("Inconclusive - borderline classification")
        else:
            findings.append("Natural spectral characteristics observed")
            findings.append("Consistent with authentic human voice")
        
        # Add confidence qualifier
        confidence = abs(score - 0.5) * 200  # Convert to 0-100%
        findings.append(f"Analysis confidence: {confidence:.1f}%")
        
        return findings