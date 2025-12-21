"""
Fusion Model (Meta-Learner) - ARES v2.0

Attention-based fusion of three analysis streams with 
confidence calibration for improved deepfake detection.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


class StreamAttention(nn.Module):
    """
    Attention mechanism for weighting stream contributions.
    
    Learns to dynamically weight streams based on input quality
    and reliability indicators.
    """
    
    def __init__(self, num_streams: int = 3, feature_size: int = 8, hidden_size: int = 32):
        super().__init__()
        
        # Each stream provides: score + confidence + additional features
        self.stream_encoder = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Query, Key, Value for self-attention
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.output = nn.Linear(hidden_size * num_streams, hidden_size)
    
    def forward(self, stream_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted fusion.
        
        Args:
            stream_features: [batch, num_streams, feature_size]
            
        Returns:
            Tuple of (fused_features, attention_weights)
        """
        batch_size, num_streams, _ = stream_features.shape
        
        # Encode streams
        encoded = self.stream_encoder(stream_features)  # [batch, streams, hidden]
        
        # Self-attention
        Q = self.query(encoded)
        K = self.key(encoded)
        V = self.value(encoded)
        
        # Attention scores
        scale = Q.shape[-1] ** 0.5
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / scale  # [batch, streams, streams]
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Weighted values
        attended = torch.bmm(attn_weights, V)  # [batch, streams, hidden]
        
        # Flatten and project
        flattened = attended.view(batch_size, -1)
        output = self.output(flattened)
        
        # Return attention weights for streams (diagonal of attention matrix)
        stream_weights = attn_weights.diagonal(dim1=1, dim2=2)  # [batch, streams]
        
        return output, stream_weights


class FusionClassifier(nn.Module):
    """
    Final classification head with confidence estimation.
    """
    
    def __init__(self, input_size: int = 32, hidden_size: int = 64):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Separate heads for classification and confidence
        self.class_head = nn.Linear(hidden_size // 2, 1)
        self.confidence_head = nn.Linear(hidden_size // 2, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify and estimate confidence.
        
        Args:
            x: [batch, input_size]
            
        Returns:
            Tuple of (class_logits, confidence_logits)
        """
        features = self.classifier(x)
        class_logits = self.class_head(features)
        confidence_logits = self.confidence_head(features)
        
        return class_logits, confidence_logits


class FusionModel:
    """
    Enhanced Fusion Model with attention-based stream aggregation
    and confidence calibration.
    
    Combines outputs from:
    - Stream A: Spectro-Temporal (Wav2Vec2)
    - Stream B: Acoustic-Biometric (LSTM)
    - Stream C: Linguistic-Prosodic (SLIM)
    """
    
    def __init__(self, model_path: Optional[str] = None, config=None):
        """
        Initialize the fusion model.
        
        Args:
            model_path: Path to pre-trained weights
            config: ARES configuration
        """
        self.device = torch.device('cpu')
        self.config = config
        
        # Neural attention-based fusion
        self.attention = StreamAttention(
            num_streams=3,
            feature_size=8,  # score + derived features
            hidden_size=32
        ).to(self.device)
        
        self.classifier = FusionClassifier(
            input_size=32,
            hidden_size=64
        ).to(self.device)
        
        # Gradient Boosting as backup/ensemble
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=5,
            random_state=42
        )
        self.gb_trained = False
        
        # Calibration temperature
        self.temperature = 1.0
        
        # Load weights
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        
        self._set_eval_mode()
        logger.info("FusionModel (Attention) initialized")
    
    def _set_eval_mode(self):
        """Set neural modules to eval mode."""
        self.attention.eval()
        self.classifier.eval()
    
    def _load_model(self, model_path: str):
        """Load pre-trained weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.attention.load_state_dict(checkpoint['attention'])
            self.classifier.load_state_dict(checkpoint['classifier'])
            self.temperature = checkpoint.get('temperature', 1.0)
            
            if 'gb_model' in checkpoint:
                self.gb_model = checkpoint['gb_model']
                self.gb_trained = True
            
            logger.info(f"Loaded fusion model from: {model_path}")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
    
    def save_model(self, model_path: str):
        """Save model weights."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        checkpoint = {
            'attention': self.attention.state_dict(),
            'classifier': self.classifier.state_dict(),
            'temperature': self.temperature
        }
        if self.gb_trained:
            checkpoint['gb_model'] = self.gb_model
        
        torch.save(checkpoint, model_path)
        logger.info(f"Saved fusion model to: {model_path}")
    
    def prepare_features(self, stream_a_score: float, stream_b_score: float,
                         stream_c_score: float, 
                         stream_a_extra: Dict = None,
                         stream_b_extra: Dict = None,
                         stream_c_extra: Dict = None) -> torch.Tensor:
        """
        Prepare features for fusion from stream outputs.
        
        Args:
            stream_*_score: Score from each stream
            stream_*_extra: Optional extra features from each stream
            
        Returns:
            Feature tensor [1, 3, 8]
        """
        # Build feature vector for each stream
        def stream_features(score, extra=None):
            feats = [
                score,  # Primary score
                abs(score - 0.5),  # Confidence (distance from uncertainty)
                1.0 if score > 0.5 else 0.0,  # Binary prediction
                min(score, 1 - score),  # Uncertainty
            ]
            
            # Add extra features if available
            if extra:
                feats.append(extra.get('attention_std', 0.0))
                feats.append(extra.get('quality_score', 0.5))
            else:
                feats.extend([0.0, 0.5])
            
            # Pad to 8 features
            while len(feats) < 8:
                feats.append(0.0)
            
            return feats[:8]
        
        stream_a = stream_features(stream_a_score, stream_a_extra)
        stream_b = stream_features(stream_b_score, stream_b_extra)
        stream_c = stream_features(stream_c_score, stream_c_extra)
        
        features = torch.tensor([stream_a, stream_b, stream_c]).float()
        features = features.unsqueeze(0).to(self.device)  # [1, 3, 8]
        
        return features
    
    def analyze(self, stream_a_score: float, stream_b_score: float, 
                stream_c_score: float,
                stream_a_extra: Dict = None,
                stream_b_extra: Dict = None,
                stream_c_extra: Dict = None) -> Dict[str, Any]:
        """
        Fuse stream outputs and make final decision.
        
        Args:
            stream_*_score: Score from each stream (0-1)
            stream_*_extra: Optional extra features
            
        Returns:
            Dictionary with final score, decision factors, and stream weights
        """
        logger.info("Running attention-based fusion")
        
        # Prepare features
        features = self.prepare_features(
            stream_a_score, stream_b_score, stream_c_score,
            stream_a_extra, stream_b_extra, stream_c_extra
        )
        
        with torch.no_grad():
            # Attention-based fusion
            fused, stream_weights = self.attention(features)
            
            # Classification with confidence
            class_logits, conf_logits = self.classifier(fused)
            
            # Temperature-scaled sigmoid
            scaled_logits = class_logits / self.temperature
            score = torch.sigmoid(scaled_logits).item()
            confidence = torch.sigmoid(conf_logits).item()
        
        # Gradient Boosting ensemble (if trained)
        if self.gb_trained:
            gb_features = np.array([[stream_a_score, stream_b_score, stream_c_score]])
            try:
                gb_prob = self.gb_model.predict_proba(gb_features)[0][1]
                # Ensemble: weighted average
                score = 0.7 * score + 0.3 * gb_prob
            except:
                pass  # Use neural score only
        
        # Stream contribution analysis
        weights = stream_weights.cpu().numpy().flatten()
        stream_names = ['Spectro-Temporal (A)', 'Acoustic-Biometric (B)', 'Linguistic-Prosodic (C)']
        scores = [stream_a_score, stream_b_score, stream_c_score]
        
        # Generate decision factors
        decision_factors = self._generate_decision_factors(
            scores, weights, stream_names, score
        )
        
        return {
            'score': float(score),
            'confidence': float(confidence),
            'stream_weights': {
                'stream_a': float(weights[0]),
                'stream_b': float(weights[1]),
                'stream_c': float(weights[2])
            },
            'decision_factors': decision_factors
        }
    
    def _generate_decision_factors(self, scores: List[float], weights: np.ndarray,
                                   names: List[str], final_score: float) -> List[str]:
        """Generate human-readable decision factors."""
        factors = []
        
        # Rank streams by contribution (score * weight)
        contributions = [(n, s, w) for n, s, w in zip(names, scores, weights)]
        contributions.sort(key=lambda x: abs(x[1] - 0.5) * x[2], reverse=True)
        
        for name, score, weight in contributions:
            weight_pct = weight * 100
            
            if score > 0.7:
                factors.append(f"{name} strongly indicates synthetic ({score:.0%}, weight: {weight_pct:.0f}%)")
            elif score > 0.55:
                factors.append(f"{name} suggests possible synthesis ({score:.0%}, weight: {weight_pct:.0f}%)")
            elif score < 0.3:
                factors.append(f"{name} strongly indicates authentic ({score:.0%}, weight: {weight_pct:.0f}%)")
            elif score < 0.45:
                factors.append(f"{name} suggests likely authentic ({score:.0%}, weight: {weight_pct:.0f}%)")
            else:
                factors.append(f"{name} inconclusive ({score:.0%}, weight: {weight_pct:.0f}%)")
        
        # Overall assessment
        if final_score > 0.7:
            factors.insert(0, "HIGH CONFIDENCE: Audio is likely AI-generated or cloned")
        elif final_score > 0.5:
            factors.insert(0, "MODERATE: Some synthetic characteristics detected")
        elif final_score < 0.3:
            factors.insert(0, "HIGH CONFIDENCE: Audio appears authentic")
        else:
            factors.insert(0, "LIKELY AUTHENTIC: Natural speech patterns detected")
        
        return factors
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50, lr: float = 1e-3) -> Dict[str, List[float]]:
        """
        Train the fusion model.
        
        Args:
            X_train: [N, 3] array of stream scores
            y_train: [N] array of labels (0=authentic, 1=synthetic)
            X_val: Optional validation features
            y_val: Optional validation labels
            epochs: Number of neural training epochs
            lr: Learning rate
            
        Returns:
            Training history
        """
        from torch.optim import AdamW
        
        logger.info("Training fusion model...")
        
        # Train Gradient Boosting first
        logger.info("Training Gradient Boosting ensemble...")
        self.gb_model.fit(X_train, y_train)
        self.gb_trained = True
        
        train_acc = self.gb_model.score(X_train, y_train)
        logger.info(f"GB Train accuracy: {train_acc:.4f}")
        
        if X_val is not None and y_val is not None:
            val_acc = self.gb_model.score(X_val, y_val)
            logger.info(f"GB Val accuracy: {val_acc:.4f}")
        
        # Train neural attention model
        logger.info("Training neural attention fusion...")
        
        self.attention.train()
        self.classifier.train()
        
        params = list(self.attention.parameters()) + list(self.classifier.parameters())
        optimizer = AdamW(params, lr=lr, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Mini-batch training
            batch_size = 32
            indices = np.random.permutation(len(X_train))
            
            for i in range(0, len(X_train), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_X = X_train[batch_idx]
                batch_y = y_train[batch_idx]
                
                optimizer.zero_grad()
                
                # Prepare features for batch
                batch_features = []
                for scores in batch_X:
                    feat = self.prepare_features(scores[0], scores[1], scores[2])
                    batch_features.append(feat)
                
                features = torch.cat(batch_features, dim=0)
                labels = torch.tensor(batch_y, dtype=torch.float32).to(self.device)
                
                # Forward
                fused, _ = self.attention(features)
                class_logits, _ = self.classifier(fused)
                
                loss = criterion(class_logits.squeeze(), labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / max(len(X_train) // batch_size, 1)
            history['train_loss'].append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        self._set_eval_mode()
        
        # Calibrate temperature using validation set
        if X_val is not None and y_val is not None:
            self._calibrate_temperature(X_val, y_val)
        
        logger.info("Fusion training complete")
        return history
    
    def _calibrate_temperature(self, X_val: np.ndarray, y_val: np.ndarray):
        """Calibrate temperature scaling for better confidence estimates."""
        logger.info("Calibrating confidence temperature...")
        
        # Grid search for best temperature
        best_temp = 1.0
        best_loss = float('inf')
        
        for temp in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            self.temperature = temp
            
            total_loss = 0.0
            for scores, label in zip(X_val, y_val):
                result = self.analyze(scores[0], scores[1], scores[2])
                pred = result['score']
                # Cross-entropy loss
                loss = -label * np.log(pred + 1e-8) - (1-label) * np.log(1-pred + 1e-8)
                total_loss += loss
            
            if total_loss < best_loss:
                best_loss = total_loss
                best_temp = temp
        
        self.temperature = best_temp
        logger.info(f"Calibrated temperature: {best_temp}")