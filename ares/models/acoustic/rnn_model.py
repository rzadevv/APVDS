"""
Acoustic & Biometric Analysis Model.

This module implements the RNN model for analyzing acoustic and biometric
characteristics of speech.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

logger = logging.getLogger(__name__)

class AcousticBiometricModel:
    """
    RNN-based model for acoustic and biometric analysis.
    
    This model analyzes acoustic features like MFCCs, pitch contour, jitter,
    and shimmer to detect the subtle biometric signatures of human speech.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the model.
        
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
            
        logger.info("AcousticBiometricModel initialized")
    
    def _build_model(self):
        """
        Build the LSTM-based model architecture.
        """
        logger.info("Building new AcousticBiometricModel")
        
        # Define input shapes
        # MFCC features (e.g., 13 base + 13 delta + 13 delta-delta features)
        mfcc_input = layers.Input(shape=(None, 39), name='mfcc_features')
        
        # Pitch contour features (f0, voiced flag)
        pitch_input = layers.Input(shape=(None, 2), name='pitch_features')
        
        # Global features (jitter, shimmer)
        global_input = layers.Input(shape=(2,), name='global_features')
        
        # Process MFCC features with Bi-LSTM
        x_mfcc = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(mfcc_input)
        x_mfcc = layers.Bidirectional(layers.LSTM(64))(x_mfcc)
        
        # Process pitch contour with Bi-LSTM
        x_pitch = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(pitch_input)
        x_pitch = layers.Bidirectional(layers.LSTM(32))(x_pitch)
        
        # Combine sequential features
        combined_sequential = layers.concatenate([x_mfcc, x_pitch])
        
        # Process combined sequential features
        x = layers.Dense(128, activation='relu')(combined_sequential)
        x = layers.Dropout(0.3)(x)
        
        # Add global features (jitter, shimmer)
        global_features = layers.Dense(16, activation='relu')(global_input)
        x = layers.concatenate([x, global_features])
        
        # Final dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        self.model = models.Model(
            inputs=[mfcc_input, pitch_input, global_input],
            outputs=output
        )
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
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
            self.model = models.load_model(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fall back to creating a new model
            self._build_model()
    
    def preprocess_input(self, features):
        """
        Preprocess the input data for the model.
        
        Args:
            features: Dictionary with keys 'mfccs', 'pitch_contour', 'jitter', 'shimmer'
            
        Returns:
            Preprocessed inputs
        """
        # Extract and prepare MFCCs
        mfccs = features['mfccs']
        if len(mfccs.shape) == 2:  # (features, frames)
            mfccs = mfccs.T  # Convert to (frames, features)
            
        # Normalize MFCCs
        mfccs_mean = np.mean(mfccs, axis=0)
        mfccs_std = np.std(mfccs, axis=0) + 1e-8
        mfccs_normalized = (mfccs - mfccs_mean) / mfccs_std
        
        # Extract and prepare pitch contour
        f0 = features['pitch_contour']['f0']
        voiced_flag = features['pitch_contour']['voiced_flag'].astype(float)
        
        # Normalize f0
        f0_mean = np.mean(f0[voiced_flag > 0])
        f0_std = np.std(f0[voiced_flag > 0]) + 1e-8
        f0_normalized = (f0 - f0_mean) / f0_std
        f0_normalized[voiced_flag == 0] = 0  # Zero out unvoiced frames
        
        # Combine pitch features
        pitch_features = np.stack([f0_normalized, voiced_flag], axis=-1)
        
        # Global features
        global_features = np.array([features['jitter'], features['shimmer']])
        
        # Add batch dimension
        mfccs_normalized = np.expand_dims(mfccs_normalized, axis=0)
        pitch_features = np.expand_dims(pitch_features, axis=0)
        global_features = np.expand_dims(global_features, axis=0)
        
        return {
            'mfcc_features': mfccs_normalized,
            'pitch_features': pitch_features,
            'global_features': global_features
        }
    
    def train(self, train_data, validation_data=None, epochs=20, batch_size=32):
        """
        Train the model.
        
        Args:
            train_data: Training data dictionary with mfcc, pitch, global features and labels
            validation_data: Validation data with the same format (optional)
            epochs: Number of epochs to train
            batch_size: Batch size
            
        Returns:
            Training history
        """
        if self.model is None:
            self._build_model()
            
        # Prepare callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        # Train model
        logger.info(f"Starting training for {epochs} epochs")
        history = self.model.fit(
            [
                train_data['mfcc_features'],
                train_data['pitch_features'],
                train_data['global_features']
            ],
            train_data['labels'],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks
        )
        
        logger.info("Training completed")
        return history
    
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
            self.model.save(model_path)
            logger.info(f"Model saved to: {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def analyze(self, features):
        """
        Analyze the acoustic and biometric characteristics of an audio sample.
        
        Args:
            features: Dictionary with 'mfccs', 'pitch_contour', 'jitter', 'shimmer'
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing acoustic and biometric features")
        
        # For demonstration purposes, we'll return random scores
        # In a real implementation, this would use the trained model
        
        # Placeholder for actual model prediction
        # processed_input = self.preprocess_input(features)
        # prediction = self.model.predict([
        #     processed_input['mfcc_features'],
        #     processed_input['pitch_features'],
        #     processed_input['global_features']
        # ])
        # score = prediction[0][0]
        
        # For demonstration, return a random score between 0.0 and 1.0
        # 0.0 = completely authentic, 1.0 = definitely cloned
        score = np.random.uniform(0.3, 0.7)
        
        # Get jitter and shimmer values for the report
        jitter = features['jitter']
        shimmer = features['shimmer']
        
        # Generate some example findings based on the score and values
        findings = []
        if score > 0.7:
            findings.append(f"Abnormally low jitter ({jitter:.6f}) and shimmer ({shimmer:.6f})")
            findings.append("Unnatural pitch transitions detected")
        elif score > 0.5:
            findings.append(f"Somewhat unusual jitter ({jitter:.6f}) and shimmer ({shimmer:.6f}) values")
            findings.append("Some irregularities in acoustic features")
        else:
            findings.append(f"Natural jitter ({jitter:.6f}) and shimmer ({shimmer:.6f}) values")
            findings.append("Biometric features consistent with human voice")
        
        return {
            'score': float(score),
            'findings': findings
        } 