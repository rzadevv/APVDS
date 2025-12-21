"""
ARES Engine - Main coordination module v2.0.

Orchestrates the three parallel analysis streams and attention-based fusion.
"""

import os
import logging
import time
import json
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional

from .processor import AudioProcessor
from .config import get_config, ARESConfig
from ..models.spectrotemporal.cnn_model import SpectroTemporalModel
from ..models.acoustic.rnn_model import AcousticBiometricModel
from ..models.linguistic.transformer_model import LinguisticProsodyModel
from ..models.fusion.gradient_boosting import FusionModel

logger = logging.getLogger(__name__)


class ARESEngine:
    """
    Main ARES Engine v2.0 - Coordinates all analysis streams.
    
    Features:
    - Wav2Vec2-based spectral analysis
    - LSTM-based biometric analysis  
    - SLIM prosodic analysis
    - Attention-based multi-stream fusion
    """
    
    def __init__(self, models_dir: Optional[str] = None, config: Optional[ARESConfig] = None):
        """
        Initialize the ARES Engine.
        
        Args:
            models_dir: Directory containing pre-trained model weights
            config: ARES configuration
        """
        logger.info("Initializing ARES Engine v2.0")
        
        self.config = config or get_config()
        
        # Set models directory
        if models_dir is None:
            models_dir = str(self.config.models_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(self.config)
        
        # Initialize models
        logger.info("Loading analysis models...")
        self.load_models()
        
        logger.info("ARES Engine v2.0 initialized successfully")
    
    def load_models(self):
        """Load all analysis models with configuration."""
        # Stream A: Spectro-Temporal (Wav2Vec2)
        spectro_path = self.models_dir / "spectro_temporal.pt"
        self.spectro_model = SpectroTemporalModel(
            model_path=str(spectro_path) if spectro_path.exists() else None,
            config=self.config
        )
        
        # Stream B: Acoustic-Biometric (LSTM)
        acoustic_path = self.models_dir / "acoustic_biometric.pt"
        self.acoustic_model = AcousticBiometricModel(
            model_path=str(acoustic_path) if acoustic_path.exists() else None,
            config=self.config
        )
        
        # Stream C: Linguistic-Prosodic (SLIM)
        linguistic_path = self.models_dir / "linguistic_prosodic.pt"
        self.linguistic_model = LinguisticProsodyModel(
            model_path=str(linguistic_path) if linguistic_path.exists() else None,
            config=self.config
        )
        
        # Fusion Model
        fusion_path = self.models_dir / "fusion.pt"
        self.fusion_model = FusionModel(
            model_path=str(fusion_path) if fusion_path.exists() else None,
            config=self.config
        )
        
        logger.info("All models loaded")
    
    def save_models(self):
        """Save all model weights."""
        self.spectro_model.save_classifier(str(self.models_dir / "spectro_temporal.pt"))
        self.acoustic_model.save_model(str(self.models_dir / "acoustic_biometric.pt"))
        self.linguistic_model.save_model(str(self.models_dir / "linguistic_prosodic.pt"))
        self.fusion_model.save_model(str(self.models_dir / "fusion.pt"))
        logger.info(f"Models saved to {self.models_dir}")
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze an audio file for deepfake detection.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with classification, confidence, and detailed evidence
        """
        logger.info(f"Analyzing: {file_path}")
        start_time = time.time()
        
        try:
            # Process audio and extract features
            features = self.audio_processor.process_file(file_path)
            
            # Get raw waveform for Wav2Vec2
            waveform_16k, sr_16k = self.audio_processor.get_raw_waveform_for_wav2vec(file_path)
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return {
                'classification': 'Error',
                'confidence_score': 0.0,
                'evidence': {'error': str(e)}
            }
        
        # Run three analysis streams in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            stream_a_future = executor.submit(
                self._run_stream_a, 
                waveform_16k, sr_16k,
                features['mel_spectrogram'],
                features['constant_q_transform']
            )
            
            stream_b_future = executor.submit(
                self._run_stream_b,
                features
            )
            
            stream_c_future = executor.submit(
                self._run_stream_c,
                features['waveform'],
                features['sample_rate']
            )
            
            stream_a_result = stream_a_future.result()
            stream_b_result = stream_b_future.result()
            stream_c_result = stream_c_future.result()
        
        # Fuse results
        fusion_result = self._run_fusion(
            stream_a_result,
            stream_b_result,
            stream_c_result
        )
        
        # Prepare final result
        final_score = fusion_result['score']
        confidence = fusion_result.get('confidence', abs(final_score - 0.5) * 200)
        
        result = {
            'classification': 'Authentic' if final_score < 0.5 else 'Cloned (AI-Generated)',
            'confidence_score': confidence,
            'raw_score': final_score,
            'evidence': {
                'spectro_temporal': {
                    'score': stream_a_result['score'],
                    'findings': stream_a_result['findings']
                },
                'acoustic_biometric': {
                    'score': stream_b_result['score'],
                    'findings': stream_b_result['findings']
                },
                'linguistic_prosodic': {
                    'score': stream_c_result['score'],
                    'findings': stream_c_result['findings']
                },
                'fusion': {
                    'score': fusion_result['score'],
                    'stream_weights': fusion_result.get('stream_weights', {}),
                    'decision_factors': fusion_result['decision_factors']
                }
            },
            'analysis_time': time.time() - start_time
        }
        
        logger.info(f"Analysis complete: {result['classification']} "
                   f"({result['confidence_score']:.1f}% confidence) "
                   f"in {result['analysis_time']:.2f}s")
        
        return result
    
    def _run_stream_a(self, waveform: np.ndarray, sample_rate: int,
                      mel_spec: np.ndarray, cqt: np.ndarray) -> Dict[str, Any]:
        """Run Spectro-Temporal Analysis (Stream A)."""
        logger.debug("Running Stream A: Spectro-Temporal")
        
        try:
            result = self.spectro_model.analyze(
                mel_spectrogram=mel_spec,
                constant_q_transform=cqt,
                waveform=waveform,
                sample_rate=sample_rate
            )
            return result
            
        except Exception as e:
            logger.error(f"Stream A error: {e}")
            return {'score': 0.5, 'findings': [f"Error: {e}"]}
    
    def _run_stream_b(self, features: Dict) -> Dict[str, Any]:
        """Run Acoustic-Biometric Analysis (Stream B)."""
        logger.debug("Running Stream B: Acoustic-Biometric")
        
        try:
            acoustic_features = {
                'mfccs': features['mfccs'],
                'pitch_contour': features['pitch_contour'],
                'jitter': features['jitter'],
                'shimmer': features['shimmer']
            }
            
            result = self.acoustic_model.analyze(acoustic_features)
            return result
            
        except Exception as e:
            logger.error(f"Stream B error: {e}")
            return {'score': 0.5, 'findings': [f"Error: {e}"]}
    
    def _run_stream_c(self, waveform: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Run Linguistic-Prosodic Analysis (Stream C)."""
        logger.debug("Running Stream C: Linguistic-Prosodic")
        
        try:
            result = self.linguistic_model.analyze(waveform, sample_rate)
            return result
            
        except Exception as e:
            logger.error(f"Stream C error: {e}")
            return {'score': 0.5, 'findings': [f"Error: {e}"]}
    
    def _run_fusion(self, stream_a: Dict, stream_b: Dict, stream_c: Dict) -> Dict[str, Any]:
        """Run Fusion Engine to combine stream results."""
        logger.debug("Running Fusion Engine")
        
        try:
            # Extract extra features for attention
            stream_a_extra = {}
            if 'attention_weights' in stream_a:
                attn = np.array(stream_a['attention_weights'])
                stream_a_extra['attention_std'] = float(np.std(attn))
            
            stream_b_extra = {}
            if 'mfcc_attention' in stream_b:
                attn = np.array(stream_b['mfcc_attention'])
                stream_b_extra['attention_std'] = float(np.std(attn.flatten()))
            
            stream_c_extra = {}
            if 'details' in stream_c and 'mismatch_score' in stream_c['details']:
                stream_c_extra['quality_score'] = 1.0 - stream_c['details']['mismatch_score']
            
            result = self.fusion_model.analyze(
                stream_a_score=stream_a['score'],
                stream_b_score=stream_b['score'],
                stream_c_score=stream_c['score'],
                stream_a_extra=stream_a_extra,
                stream_b_extra=stream_b_extra,
                stream_c_extra=stream_c_extra
            )
            return result
            
        except Exception as e:
            logger.error(f"Fusion error: {e}")
            # Fallback to simple average
            scores = [stream_a['score'], stream_b['score'], stream_c['score']]
            return {
                'score': np.mean(scores),
                'decision_factors': [f"Fallback averaging due to error: {e}"]
            }
    
    def save_report(self, result: Dict, output_path: Optional[str] = None) -> str:
        """
        Save analysis report to JSON.
        
        Args:
            result: Analysis result dictionary
            output_path: Output file path (auto-generated if None)
            
        Returns:
            Path to saved report
        """
        if output_path is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = f"ares_report_{timestamp}.json"
        
        try:
            # Convert numpy types for JSON
            result_clean = self._prepare_for_json(result)
            
            with open(output_path, 'w') as f:
                json.dump(result_clean, f, indent=2)
            
            logger.info(f"Report saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return ""
    
    def _prepare_for_json(self, obj):
        """Convert numpy types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj