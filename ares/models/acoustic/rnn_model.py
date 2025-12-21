"""
Acoustic & Biometric Analysis Model (Stream B) - ARES v2.0

Enhanced RNN model with bidirectional LSTM, attention mechanism,
and improved voice quality analysis for deepfake detection.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BiometricAttention(nn.Module):
    """
    Self-attention module for acoustic feature sequences.
    
    Learns to focus on regions with unusual biometric characteristics.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism.
        
        Args:
            x: [batch, seq_len, hidden_size]
            mask: Optional attention mask
            
        Returns:
            Tuple of (weighted_sum, attention_weights)
        """
        attn_scores = self.attention(x).squeeze(-1)  # [batch, seq_len]
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, seq_len]
        
        # Weighted sum
        weighted_sum = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # [batch, hidden]
        
        return weighted_sum, attn_weights


class AcousticEncoder(nn.Module):
    """
    Encodes MFCC and pitch sequences using bidirectional LSTM.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = BiometricAttention(hidden_size * 2)
        self.norm = nn.LayerNorm(hidden_size * 2)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequence with LSTM and attention.
        
        Args:
            x: [batch, seq_len, input_size]
            lengths: Optional sequence lengths for packing
            
        Returns:
            Tuple of (encoded, attention_weights)
        """
        # LSTM encoding
        outputs, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
        outputs = self.norm(outputs)
        
        # Attention pooling
        encoded, attn_weights = self.attention(outputs)
        
        return encoded, attn_weights


class VoiceQualityAnalyzer(nn.Module):
    """
    Analyzes voice quality metrics (jitter, shimmer, HNR, etc.)
    
    These metrics are strong indicators of synthetic vs natural speech.
    """
    
    def __init__(self, num_features: int = 8, hidden_size: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # Thresholds for voice quality metrics (learned)
        self.threshold_net = nn.Linear(num_features, num_features)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Analyze voice quality features.
        
        Args:
            features: [batch, num_features] - jitter, shimmer, etc.
            
        Returns:
            Tuple of (encoded, anomaly_scores)
        """
        encoded = self.encoder(features)
        
        # Compute anomaly scores based on learned thresholds
        thresholds = torch.sigmoid(self.threshold_net(features))
        anomaly_scores = torch.abs(features - thresholds)
        
        return encoded, {'voice_quality_anomalies': anomaly_scores}


class AcousticBiometricModel:
    """
    Enhanced Acoustic & Biometric Analysis using bidirectional LSTM
    with attention for deepfake detection.
    
    Analyzes:
    - MFCC trajectories
    - Pitch contour dynamics
    - Voice quality metrics (jitter, shimmer)
    - Formant transitions
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
        
        # Model hyperparameters
        hidden_size = 256
        if config:
            hidden_size = config.model.acoustic_hidden_size
        
        # Build model components
        self.mfcc_encoder = AcousticEncoder(
            input_size=120,  # 40 MFCCs + delta + delta-delta = 120 features
            hidden_size=hidden_size // 2,
            num_layers=2,
            dropout=0.3
        ).to(self.device)
        
        self.pitch_encoder = AcousticEncoder(
            input_size=3,  # f0, voiced_flag, voiced_prob
            hidden_size=hidden_size // 4,
            num_layers=1,
            dropout=0.2
        ).to(self.device)
        
        self.voice_quality = VoiceQualityAnalyzer(
            num_features=8,  # jitter, shimmer, HNR, etc.
            hidden_size=64
        ).to(self.device)
        
        # Fusion and classification
        fusion_input_size = hidden_size + hidden_size // 2 + 64
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        ).to(self.device)
        
        # Load weights if available
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        
        self._set_eval_mode()
        logger.info("AcousticBiometricModel (Enhanced) initialized")
    
    def _set_eval_mode(self):
        """Set all modules to eval mode."""
        self.mfcc_encoder.eval()
        self.pitch_encoder.eval()
        self.voice_quality.eval()
        self.classifier.eval()
    
    def _set_train_mode(self):
        """Set all modules to train mode."""
        self.mfcc_encoder.train()
        self.pitch_encoder.train()
        self.voice_quality.train()
        self.classifier.train()
    
    def _load_model(self, model_path: str):
        """Load pre-trained weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.mfcc_encoder.load_state_dict(checkpoint['mfcc_encoder'])
            self.pitch_encoder.load_state_dict(checkpoint['pitch_encoder'])
            self.voice_quality.load_state_dict(checkpoint['voice_quality'])
            self.classifier.load_state_dict(checkpoint['classifier'])
            logger.info(f"Loaded model from: {model_path}")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
    
    def save_model(self, model_path: str):
        """Save model weights."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        checkpoint = {
            'mfcc_encoder': self.mfcc_encoder.state_dict(),
            'pitch_encoder': self.pitch_encoder.state_dict(),
            'voice_quality': self.voice_quality.state_dict(),
            'classifier': self.classifier.state_dict()
        }
        torch.save(checkpoint, model_path)
        logger.info(f"Saved model to: {model_path}")
    
    def _prepare_features(self, features: Dict) -> Dict[str, torch.Tensor]:
        """
        Prepare input features for the model.
        
        Args:
            features: Dictionary with mfccs, pitch_contour, jitter, shimmer
            
        Returns:
            Dictionary of tensors
        """
        # MFCCs: [features, time] -> [batch, time, features]
        mfccs = features['mfccs']
        if isinstance(mfccs, np.ndarray):
            if len(mfccs.shape) == 2:
                mfccs = mfccs.T  # [time, features]
            mfccs = torch.from_numpy(mfccs).float().unsqueeze(0)
        
        # Normalize MFCCs
        mfccs = (mfccs - mfccs.mean(dim=1, keepdim=True)) / (mfccs.std(dim=1, keepdim=True) + 1e-8)
        
        # Pitch contour
        pitch_data = features['pitch_contour']
        if isinstance(pitch_data, dict):
            f0 = pitch_data.get('f0', np.zeros(100))
            voiced = pitch_data.get('voiced_flag', np.ones(100)).astype(float)
            voiced_prob = pitch_data.get('voiced_probs', voiced)
        else:
            f0 = pitch_data
            voiced = np.ones_like(f0)
            voiced_prob = voiced
        
        # Handle NaN in f0
        f0 = np.nan_to_num(f0, nan=0.0)
        voiced_prob = np.nan_to_num(voiced_prob, nan=0.0)
        
        # Normalize f0
        f0_voiced = f0[voiced > 0.5]
        if len(f0_voiced) > 0:
            f0_mean = np.mean(f0_voiced)
            f0_std = np.std(f0_voiced) + 1e-8
            f0_norm = (f0 - f0_mean) / f0_std
            f0_norm[voiced < 0.5] = 0
        else:
            f0_norm = np.zeros_like(f0)
        
        pitch_features = np.stack([f0_norm, voiced, voiced_prob], axis=-1)
        pitch_tensor = torch.from_numpy(pitch_features).float().unsqueeze(0)
        
        # Voice quality features
        jitter = features.get('jitter', 0.0)
        shimmer = features.get('shimmer', 0.0)
        
        # Typical ranges for additional derived features
        jitter_score = min(jitter / 0.02, 1.0)  # Normalize to 0-1
        shimmer_score = min(shimmer / 0.1, 1.0)
        
        # Derive additional voice quality metrics from pitch
        if len(f0_voiced) > 1:
            f0_cv = np.std(f0_voiced) / (np.mean(f0_voiced) + 1e-8)  # Coefficient of variation
            f0_range = (np.max(f0_voiced) - np.min(f0_voiced)) / (np.mean(f0_voiced) + 1e-8)
        else:
            f0_cv = 0.0
            f0_range = 0.0
        
        voice_quality_features = torch.tensor([
            jitter, shimmer, jitter_score, shimmer_score,
            f0_cv, f0_range, np.mean(voiced), np.std(f0_norm)
        ]).float().unsqueeze(0)
        
        return {
            'mfccs': mfccs.to(self.device),
            'pitch': pitch_tensor.to(self.device),
            'voice_quality': voice_quality_features.to(self.device)
        }
    
    def analyze(self, features: Dict) -> Dict[str, Any]:
        """
        Analyze acoustic and biometric characteristics.
        
        Args:
            features: Dictionary with mfccs, pitch_contour, jitter, shimmer
            
        Returns:
            Dictionary with score and findings
        """
        logger.info("Analyzing acoustic and biometric features")
        
        try:
            # Prepare inputs
            inputs = self._prepare_features(features)
            
            with torch.no_grad():
                # Encode MFCC sequence
                mfcc_encoded, mfcc_attn = self.mfcc_encoder(inputs['mfccs'])
                
                # Encode pitch sequence
                pitch_encoded, pitch_attn = self.pitch_encoder(inputs['pitch'])
                
                # Analyze voice quality
                vq_encoded, vq_anomalies = self.voice_quality(inputs['voice_quality'])
                
                # Concatenate all features
                combined = torch.cat([mfcc_encoded, pitch_encoded, vq_encoded], dim=-1)
                
                # Classification
                logits = self.classifier(combined)
                score = torch.sigmoid(logits).item()
            
            # Generate findings
            findings = self._generate_findings(
                score, 
                features.get('jitter', 0), 
                features.get('shimmer', 0),
                vq_anomalies,
                mfcc_attn.cpu().numpy(),
                pitch_attn.cpu().numpy()
            )
            
            return {
                'score': float(score),
                'findings': findings,
                'mfcc_attention': mfcc_attn.cpu().numpy().tolist(),
                'pitch_attention': pitch_attn.cpu().numpy().tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in acoustic analysis: {e}")
            # Fallback to heuristic analysis
            return self._heuristic_analysis(features)
    
    def _heuristic_analysis(self, features: Dict) -> Dict[str, Any]:
        """Fallback heuristic analysis without model inference."""
        jitter = features.get('jitter', 0.01)
        shimmer = features.get('shimmer', 0.03)
        
        # Heuristic scoring based on voice quality metrics
        # Very low jitter/shimmer can indicate synthetic audio
        jitter_suspicious = jitter < 0.003 or jitter > 0.05
        shimmer_suspicious = shimmer < 0.01 or shimmer > 0.15
        
        score = 0.5
        findings = []
        
        if jitter_suspicious:
            score += 0.15
            if jitter < 0.003:
                findings.append(f"Abnormally low jitter ({jitter:.6f}) - possible synthesis")
            else:
                findings.append(f"Elevated jitter ({jitter:.6f}) - unusual voice quality")
        
        if shimmer_suspicious:
            score += 0.15
            if shimmer < 0.01:
                findings.append(f"Abnormally low shimmer ({shimmer:.6f}) - possible synthesis")
            else:
                findings.append(f"Elevated shimmer ({shimmer:.6f}) - unusual voice quality")
        
        if not findings:
            findings.append(f"Normal jitter ({jitter:.6f}) and shimmer ({shimmer:.6f})")
            findings.append("Voice quality metrics consistent with natural speech")
        
        return {'score': min(score, 1.0), 'findings': findings}
    
    def _generate_findings(self, score: float, jitter: float, shimmer: float,
                          vq_anomalies: Dict, mfcc_attn: np.ndarray, 
                          pitch_attn: np.ndarray) -> List[str]:
        """Generate human-readable findings from analysis."""
        findings = []
        
        # Score-based findings
        if score > 0.7:
            findings.append("Strong acoustic anomalies detected - likely synthetic")
        elif score > 0.5:
            findings.append("Moderate acoustic irregularities observed")
        else:
            findings.append("Acoustic features consistent with natural speech")
        
        # Jitter/Shimmer analysis
        # Natural speech: jitter ~0.5-1%, shimmer ~3-5%
        if jitter < 0.003:
            findings.append(f"Jitter ({jitter:.4%}) abnormally low - synthetic indicator")
        elif jitter > 0.02:
            findings.append(f"Jitter ({jitter:.4%}) elevated - potential voice disorder or synthesis artifact")
        else:
            findings.append(f"Jitter ({jitter:.4%}) within normal range")
        
        if shimmer < 0.015:
            findings.append(f"Shimmer ({shimmer:.4%}) abnormally low - synthetic indicator")
        elif shimmer > 0.1:
            findings.append(f"Shimmer ({shimmer:.4%}) elevated - unusual amplitude variation")
        else:
            findings.append(f"Shimmer ({shimmer:.4%}) within normal range")
        
        # Attention-based findings
        mfcc_attn_flat = mfcc_attn.flatten() if len(mfcc_attn.shape) > 1 else mfcc_attn
        if len(mfcc_attn_flat) > 0 and np.std(mfcc_attn_flat) > 0.1:
            peak_region = np.argmax(mfcc_attn_flat) / len(mfcc_attn_flat) * 100
            findings.append(f"High attention at ~{peak_region:.0f}% of audio in MFCC analysis")
        
        return findings
    
    def train(self, train_data: Dict, validation_data: Optional[Dict] = None,
              epochs: int = 30, batch_size: int = 16, lr: float = 1e-4):
        """
        Train the model on labeled data.
        
        Args:
            train_data: Dict with features and labels
            validation_data: Optional validation data
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
            
        Returns:
            Training history
        """
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        
        self._set_train_mode()
        
        # Collect all parameters
        params = (
            list(self.mfcc_encoder.parameters()) +
            list(self.pitch_encoder.parameters()) +
            list(self.voice_quality.parameters()) +
            list(self.classifier.parameters())
        )
        
        optimizer = AdamW(params, lr=lr, weight_decay=0.01)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
        criterion = nn.BCEWithLogitsLoss()
        
        history = {'train_loss': [], 'val_loss': []}
        best_loss = float('inf')
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(train_data['features']), batch_size):
                batch_features = train_data['features'][i:i+batch_size]
                batch_labels = train_data['labels'][i:i+batch_size]
                
                optimizer.zero_grad()
                
                batch_loss = 0.0
                for feat, label in zip(batch_features, batch_labels):
                    inputs = self._prepare_features(feat)
                    
                    mfcc_enc, _ = self.mfcc_encoder(inputs['mfccs'])
                    pitch_enc, _ = self.pitch_encoder(inputs['pitch'])
                    vq_enc, _ = self.voice_quality(inputs['voice_quality'])
                    
                    combined = torch.cat([mfcc_enc, pitch_enc, vq_enc], dim=-1)
                    logits = self.classifier(combined)
                    
                    target = torch.tensor([[label]], dtype=torch.float32).to(self.device)
                    batch_loss += criterion(logits, target)
                
                batch_loss /= len(batch_features)
                batch_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / max(num_batches, 1)
            history['train_loss'].append(avg_loss)
            
            scheduler.step()
            
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
        
        self._set_eval_mode()
        logger.info("Training complete")
        
        return history