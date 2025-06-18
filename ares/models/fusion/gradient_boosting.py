"""
Fusion Model using Gradient Boosting.

This module implements the fusion model that combines the outputs from the three
analysis streams to make a final decision.
"""

import os
import logging
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier

logger = logging.getLogger(__name__)

class FusionModel:
    """
    Fusion model that combines the outputs from the three analysis streams.
    
    This model uses a gradient boosting classifier to make the final decision
    based on the probability scores and features from each analysis stream.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the fusion model.
        
        Args:
            model_path: Path to a pre-trained model (optional)
        """
        self.model = None
        
        # Try to load a pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            # Create a new model
            self._build_model()
            
        logger.info("FusionModel initialized")
    
    def _build_model(self):
        """
        Build the gradient boosting model.
        """
        logger.info("Building new FusionModel")
        
        # Create a gradient boosting classifier
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        logger.info("Model built successfully")
    
    def _load_model(self, model_path):
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to the model file
        """
        try:
            logger.info(f"Loading model from: {model_path}")
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fall back to creating a new model
            self._build_model()
    
    def save_model(self, model_path):
        """
        Save the model.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            logger.error("Cannot save model: No model exists")
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save the model
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to: {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the fusion model.
        
        Args:
            X_train: Training features (scores from the three streams)
            y_train: Training labels (1 for cloned, 0 for authentic)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training accuracy
        """
        if self.model is None:
            self._build_model()
            
        # Train the model
        logger.info("Training fusion model")
        self.model.fit(X_train, y_train)
        
        # Evaluate on training set
        train_accuracy = self.model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_accuracy = self.model.score(X_val, y_val)
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        return train_accuracy
    
    def prepare_features(self, stream_a_score, stream_b_score, stream_c_score):
        """
        Prepare features for the fusion model.
        
        Args:
            stream_a_score: Score from Stream A (Spectro-Temporal Analysis)
            stream_b_score: Score from Stream B (Acoustic & Biometric Analysis)
            stream_c_score: Score from Stream C (Linguistic & Prosodic Analysis)
            
        Returns:
            Features array for the model
        """
        # Basic feature vector with just the scores
        features = np.array([stream_a_score, stream_b_score, stream_c_score]).reshape(1, -1)
        
        # In a real implementation, we might add more derived features like:
        # - Pairwise differences between scores
        # - Statistical moments of the scores
        # - Weighted combinations
        
        return features
    
    def analyze(self, stream_a_score, stream_b_score, stream_c_score):
        """
        Analyze the combined outputs from the three streams.
        
        Args:
            stream_a_score: Score from Stream A (Spectro-Temporal Analysis)
            stream_b_score: Score from Stream B (Acoustic & Biometric Analysis)
            stream_c_score: Score from Stream C (Linguistic & Prosodic Analysis)
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Running fusion analysis")
        
        # For demonstration purposes, we'll return a weighted average
        # In a real implementation, this would use the trained model
        
        # Prepare features
        features = self.prepare_features(stream_a_score, stream_b_score, stream_c_score)
        
        # If we have a trained model, use it
        # if self.model is not None:
        #     probability = self.model.predict_proba(features)[0][1]  # Probability of being cloned
        # else:
        #     # Fallback to weighted average if no model is available
        
        # For demonstration, use a weighted average of the scores
        # Stream A has the highest weight as spectral artifacts are most reliable
        weights = [0.4, 0.3, 0.3]  # Weights for Stream A, B, C
        scores = [stream_a_score, stream_b_score, stream_c_score]
        weighted_score = np.average(scores, weights=weights)
        
        # Determine which streams contributed most to the decision
        stream_names = ['Spectro-Temporal', 'Acoustic-Biometric', 'Linguistic-Prosodic']
        ranked_streams = sorted(
            zip(stream_names, scores),
            key=lambda x: abs(x[1] - 0.5),  # Sort by distance from neutral (0.5)
            reverse=True
        )
        
        # Generate decision factors based on the scores
        decision_factors = []
        for stream, score in ranked_streams:
            if score > 0.6:
                decision_factors.append(f"{stream} analysis strongly indicates cloned voice")
            elif score > 0.5:
                decision_factors.append(f"{stream} analysis suggests possible cloning")
            elif score < 0.4:
                decision_factors.append(f"{stream} analysis strongly indicates authentic voice")
            else:
                decision_factors.append(f"{stream} analysis suggests likely authentic voice")
        
        return {
            'score': float(weighted_score),
            'decision_factors': decision_factors
        } 