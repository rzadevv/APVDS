"""
Linguistic & Prosodic Analysis Model (Stream C) - ARES v2.0

Implements SLIM (Style-Linguistics Mismatch) concept for detecting
synthetic speech through analysis of prosodic naturalness and
content-style alignment.
"""

import os
import logging
import tempfile
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


class ProsodyEncoder(nn.Module):
    """
    Encodes prosodic features (pitch, energy, duration patterns).
    """
    
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        
        # Temporal prosody encoding
        self.temporal = nn.LSTM(
            input_size=4,  # pitch, energy, delta_pitch, delta_energy
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Attention for temporal pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Global prosody statistics encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(12, 64),  # pitch/energy/rhythm stats
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64)
        )
    
    def forward(self, temporal_features: torch.Tensor, 
                global_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prosodic features.
        
        Args:
            temporal_features: [batch, seq_len, 4]
            global_features: [batch, 12]
            
        Returns:
            Tuple of (encoded, attention_weights)
        """
        # Temporal encoding
        temporal_out, _ = self.temporal(temporal_features)  # [batch, seq, hidden*2]
        
        # Attention pooling
        attn_scores = self.attention(temporal_out).squeeze(-1)  # [batch, seq]
        attn_weights = F.softmax(attn_scores, dim=-1)
        temporal_pooled = torch.bmm(attn_weights.unsqueeze(1), temporal_out).squeeze(1)
        
        # Global encoding
        global_enc = self.global_encoder(global_features)
        
        # Combine
        combined = torch.cat([temporal_pooled, global_enc], dim=-1)
        
        return combined, attn_weights


class StyleLinguisticsMismatchDetector(nn.Module):
    """
    SLIM - Style-Linguistics Mismatch Detection.
    
    Detects mismatches between speaker style (prosody) and 
    linguistic content patterns that are characteristic of synthetic speech.
    """
    
    def __init__(self, prosody_size: int = 320, content_size: int = 256, hidden_size: int = 128):
        super().__init__()
        
        # Project prosody and content to same space
        self.prosody_proj = nn.Linear(prosody_size, hidden_size)
        self.content_proj = nn.Linear(content_size, hidden_size)
        
        # Cross-attention for mismatch detection
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Mismatch classifier
        self.mismatch_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, prosody_enc: torch.Tensor, 
                content_enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect style-linguistics mismatch.
        
        Args:
            prosody_enc: [batch, prosody_size]
            content_enc: [batch, content_size]
            
        Returns:
            Tuple of (mismatch_score, attention_weights)
        """
        # Project to common space
        prosody = self.prosody_proj(prosody_enc).unsqueeze(1)  # [batch, 1, hidden]
        content = self.content_proj(content_enc).unsqueeze(1)  # [batch, 1, hidden]
        
        # Cross-attention: how well does prosody match content?
        attended, attn_weights = self.cross_attention(
            query=prosody,
            key=content,
            value=content
        )
        
        # Combine for mismatch detection
        combined = torch.cat([prosody.squeeze(1), attended.squeeze(1)], dim=-1)
        
        mismatch_score = self.mismatch_classifier(combined)
        
        return mismatch_score, attn_weights.squeeze(1)


class LinguisticProsodyModel:
    """
    Enhanced Linguistic & Prosodic Analysis with SLIM concept.
    
    Analyzes:
    - Prosodic naturalness (pitch, energy, rhythm patterns)
    - Style-content alignment
    - Pause and timing patterns
    - Speech rate consistency
    """
    
    def __init__(self, model_path: Optional[str] = None, config=None):
        """
        Initialize the model.
        
        Args:
            model_path: Path to pre-trained weights
            config: ARES configuration
        """
        self.device = torch.device('cpu')
        self.config = config
        
        # Whisper for transcription (if available)
        self.whisper_model = None
        self.whisper_processor = None
        self._init_whisper()
        
        # Prosody encoder
        self.prosody_encoder = ProsodyEncoder(hidden_size=128).to(self.device)
        
        # Content encoder (simple for now, could be enhanced with text embeddings)
        self.content_encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256)
        ).to(self.device)
        
        # SLIM detector
        self.slim_detector = StyleLinguisticsMismatchDetector(
            prosody_size=320,  # 256 temporal + 64 global
            content_size=256,
            hidden_size=128
        ).to(self.device)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(320 + 1, 128),  # prosody + mismatch score
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        ).to(self.device)
        
        # Load weights
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        
        self._set_eval_mode()
        logger.info("LinguisticProsodyModel (SLIM) initialized")
    
    def _init_whisper(self):
        """Initialize Whisper for transcription."""
        try:
            import whisper
            
            model_size = "tiny"
            if self.config:
                model_size = self.config.model.whisper_model
            
            logger.info(f"Loading Whisper ({model_size})...")
            self.whisper_model = whisper.load_model(model_size)
            logger.info("Whisper loaded")
            
        except ImportError:
            logger.warning("Whisper not available, using SpeechRecognition fallback")
            self.whisper_model = None
        except Exception as e:
            logger.warning(f"Could not load Whisper: {e}")
            self.whisper_model = None
    
    def _set_eval_mode(self):
        """Set all modules to eval mode."""
        self.prosody_encoder.eval()
        self.content_encoder.eval()
        self.slim_detector.eval()
        self.classifier.eval()
    
    def _load_model(self, model_path: str):
        """Load pre-trained weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.prosody_encoder.load_state_dict(checkpoint['prosody_encoder'])
            self.content_encoder.load_state_dict(checkpoint['content_encoder'])
            self.slim_detector.load_state_dict(checkpoint['slim_detector'])
            self.classifier.load_state_dict(checkpoint['classifier'])
            logger.info(f"Loaded model from: {model_path}")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
    
    def save_model(self, model_path: str):
        """Save model weights."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        checkpoint = {
            'prosody_encoder': self.prosody_encoder.state_dict(),
            'content_encoder': self.content_encoder.state_dict(),
            'slim_detector': self.slim_detector.state_dict(),
            'classifier': self.classifier.state_dict()
        }
        torch.save(checkpoint, model_path)
        logger.info(f"Saved model to: {model_path}")
    
    def _transcribe(self, waveform: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio to text."""
        if self.whisper_model is not None:
            try:
                # Whisper expects 16kHz audio
                if sample_rate != 16000:
                    waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
                
                result = self.whisper_model.transcribe(
                    waveform.astype(np.float32),
                    language="en",
                    fp16=False
                )
                return result.get('text', '')
                
            except Exception as e:
                logger.warning(f"Whisper transcription failed: {e}")
        
        # Fallback to SpeechRecognition
        return self._transcribe_fallback(waveform, sample_rate)
    
    def _transcribe_fallback(self, waveform: np.ndarray, sample_rate: int) -> str:
        """Fallback transcription using SpeechRecognition."""
        try:
            import speech_recognition as sr
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, waveform, sample_rate)
            
            recognizer = sr.Recognizer()
            with sr.AudioFile(tmp_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
            
            os.unlink(tmp_path)
            return text
            
        except Exception as e:
            logger.warning(f"Fallback transcription failed: {e}")
            return ""
    
    def _extract_prosodic_features(self, waveform: np.ndarray, 
                                    sample_rate: int) -> Dict[str, Any]:
        """Extract comprehensive prosodic features."""
        features = {}
        
        # Pitch extraction
        f0, voiced, voiced_prob = librosa.pyin(
            waveform,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate
        )
        f0 = np.nan_to_num(f0, nan=0.0)
        
        # Energy (RMS)
        rms = librosa.feature.rms(y=waveform, hop_length=512)[0]
        
        # Compute deltas
        f0_delta = np.diff(f0, prepend=f0[0])
        rms_delta = np.diff(rms, prepend=rms[0])
        
        # Ensure same length
        min_len = min(len(f0), len(rms))
        
        # Temporal features [seq_len, 4]
        temporal = np.stack([
            f0[:min_len] / 500.0,  # Normalize pitch
            rms[:min_len] / (rms.max() + 1e-8),  # Normalize energy
            f0_delta[:min_len] / 100.0,  # Normalize delta pitch
            rms_delta[:min_len] / (rms.max() + 1e-8)  # Normalize delta energy
        ], axis=-1)
        
        features['temporal'] = temporal
        
        # Global statistics
        f0_voiced = f0[voiced.astype(bool)] if voiced.any() else np.array([0])
        
        if len(f0_voiced) > 0 and f0_voiced.max() > 0:
            features['global'] = {
                'pitch_mean': np.mean(f0_voiced),
                'pitch_std': np.std(f0_voiced),
                'pitch_range': np.ptp(f0_voiced),
                'pitch_cv': np.std(f0_voiced) / (np.mean(f0_voiced) + 1e-8),
                'energy_mean': np.mean(rms),
                'energy_std': np.std(rms),
                'energy_range': np.ptp(rms),
                'voiced_ratio': np.mean(voiced),
                'speech_rate': self._estimate_speech_rate(waveform, sample_rate),
                'pause_ratio': self._estimate_pause_ratio(rms),
                'rhythm_regularity': self._estimate_rhythm(waveform, sample_rate),
                'duration': len(waveform) / sample_rate
            }
        else:
            features['global'] = {k: 0.0 for k in [
                'pitch_mean', 'pitch_std', 'pitch_range', 'pitch_cv',
                'energy_mean', 'energy_std', 'energy_range', 'voiced_ratio',
                'speech_rate', 'pause_ratio', 'rhythm_regularity', 'duration'
            ]}
        
        return features
    
    def _estimate_speech_rate(self, waveform: np.ndarray, sample_rate: int) -> float:
        """Estimate speech rate (syllables/sec approximation)."""
        try:
            onset_env = librosa.onset.onset_strength(y=waveform, sr=sample_rate)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sample_rate)
            duration = len(waveform) / sample_rate
            return len(onsets) / duration if duration > 0 else 0.0
        except:
            return 0.0
    
    def _estimate_pause_ratio(self, rms: np.ndarray) -> float:
        """Estimate ratio of pauses (low energy) to speech."""
        threshold = np.mean(rms) * 0.5
        pause_frames = np.sum(rms < threshold)
        return pause_frames / len(rms) if len(rms) > 0 else 0.0
    
    def _estimate_rhythm(self, waveform: np.ndarray, sample_rate: int) -> float:
        """Estimate rhythm regularity."""
        try:
            onset_env = librosa.onset.onset_strength(y=waveform, sr=sample_rate)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sample_rate)
            
            if len(onsets) > 2:
                times = librosa.frames_to_time(onsets, sr=sample_rate)
                intervals = np.diff(times)
                cv = np.std(intervals) / (np.mean(intervals) + 1e-8)
                # Lower CV = more regular rhythm
                return 1.0 / (1.0 + cv)
            return 0.5
        except:
            return 0.5
    
    def _extract_content_features(self, text: str) -> np.ndarray:
        """Extract simple content features from transcription."""
        features = np.zeros(64)
        
        if not text:
            return features
        
        # Basic text statistics
        words = text.split()
        features[0] = len(words)  # Word count
        features[1] = len(text)   # Char count
        features[2] = np.mean([len(w) for w in words]) if words else 0  # Avg word length
        features[3] = len(set(words)) / (len(words) + 1e-8)  # Vocabulary richness
        
        # Punctuation patterns
        features[4] = text.count('.')
        features[5] = text.count(',')
        features[6] = text.count('?')
        features[7] = text.count('!')
        
        # Simple lexical features
        features[8] = sum(1 for w in words if len(w) > 6) / (len(words) + 1e-8)  # Long word ratio
        features[9] = sum(1 for w in words if w[0].isupper()) / (len(words) + 1e-8) if words else 0
        
        # Normalize
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def analyze(self, waveform: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Analyze linguistic and prosodic aspects.
        
        Args:
            waveform: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            Dictionary with score and findings
        """
        logger.info("Analyzing linguistic and prosodic features (SLIM)")
        
        try:
            # Extract transcription
            text = self._transcribe(waveform, sample_rate)
            logger.debug(f"Transcription: {text[:100]}...")
            
            # Extract prosodic features
            prosody = self._extract_prosodic_features(waveform, sample_rate)
            
            # Extract content features
            content_features = self._extract_content_features(text)
            
            # Prepare tensors
            temporal = torch.from_numpy(prosody['temporal']).float().unsqueeze(0).to(self.device)
            
            global_feats = np.array(list(prosody['global'].values()))
            global_tensor = torch.from_numpy(global_feats).float().unsqueeze(0).to(self.device)
            
            content_tensor = torch.from_numpy(content_features).float().unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Encode prosody
                prosody_enc, prosody_attn = self.prosody_encoder(temporal, global_tensor)
                
                # Encode content
                content_enc = self.content_encoder(content_tensor)
                
                # Detect mismatch
                mismatch_score, mismatch_attn = self.slim_detector(prosody_enc, content_enc)
                
                # Final classification
                combined = torch.cat([prosody_enc, mismatch_score], dim=-1)
                logits = self.classifier(combined)
                score = torch.sigmoid(logits).item()
            
            # Generate findings
            findings = self._generate_findings(
                score, prosody['global'], text, mismatch_score.item()
            )
            
            return {
                'score': float(score),
                'findings': findings,
                'details': {
                    'transcription': text,
                    'prosody': prosody['global'],
                    'mismatch_score': float(mismatch_score.item())
                }
            }
            
        except Exception as e:
            logger.error(f"Error in linguistic analysis: {e}")
            return self._heuristic_analysis(waveform, sample_rate)
    
    def _heuristic_analysis(self, waveform: np.ndarray, 
                            sample_rate: int) -> Dict[str, Any]:
        """Fallback heuristic analysis."""
        prosody = self._extract_prosodic_features(waveform, sample_rate)
        global_feats = prosody['global']
        
        score = 0.5
        findings = []
        
        # Heuristic checks
        pitch_cv = global_feats['pitch_cv']
        if pitch_cv < 0.1:
            score += 0.15
            findings.append("Unusually flat pitch contour - possible synthesis")
        elif pitch_cv > 0.5:
            findings.append("Natural pitch variation present")
        
        pause_ratio = global_feats['pause_ratio']
        if pause_ratio < 0.1 or pause_ratio > 0.5:
            score += 0.1
            findings.append(f"Unusual pause pattern ({pause_ratio:.1%})")
        
        rhythm = global_feats['rhythm_regularity']
        if rhythm > 0.8:
            score += 0.1
            findings.append("Unnaturally regular rhythm")
        
        if not findings:
            findings.append("Prosodic features within normal range")
        
        return {'score': min(score, 1.0), 'findings': findings, 'details': {'prosody': global_feats}}
    
    def _generate_findings(self, score: float, prosody: Dict,
                          text: str, mismatch: float) -> List[str]:
        """Generate human-readable findings."""
        findings = []
        
        # Score-based findings
        if score > 0.7:
            findings.append("Strong linguistic-prosodic anomalies detected")
        elif score > 0.5:
            findings.append("Moderate inconsistencies in speech patterns")
        else:
            findings.append("Natural linguistic and prosodic patterns")
        
        # Mismatch analysis
        if mismatch > 0.6:
            findings.append("Style-content mismatch detected (SLIM indicator)")
        elif mismatch < 0.3:
            findings.append("Style and content are well-aligned")
        
        # Prosody specifics
        if prosody['pitch_cv'] < 0.15:
            findings.append("Monotonous pitch - synthetic indicator")
        
        if prosody['rhythm_regularity'] > 0.75:
            findings.append("Overly regular rhythm - machine-like timing")
        
        speech_rate = prosody['speech_rate']
        if speech_rate > 0:
            if speech_rate < 2.0:
                findings.append(f"Slow speech rate ({speech_rate:.1f} syl/s)")
            elif speech_rate > 6.0:
                findings.append(f"Rapid speech rate ({speech_rate:.1f} syl/s)")
        
        return findings
    
    def train(self, train_data: Dict, epochs: int = 30, lr: float = 1e-4):
        """Train the model (placeholder for full implementation)."""
        logger.info("Training SLIM model...")
        # Implementation would follow similar pattern to other models
        return {'train_loss': []}
