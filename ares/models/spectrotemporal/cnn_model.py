"""
Spectro-Temporal Analysis Model.

This module implements the CNN model for analyzing spectral artifacts in audio.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import layers, models, optimizers

logger = logging.getLogger(__name__)

class SpectroTemporalModel:
    """
    CNN-based model for spectro-temporal analysis.
    
    This model analyzes mel-spectrograms and CQTs to detect spectral artifacts
    that are characteristic of AI-generated or cloned voices.
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
            
        logger.info("SpectroTemporalModel initialized")
    
    def _build_model(self):
        """
        Build the CNN model architecture.
        
        This implements a ResNet-style CNN for analyzing spectrograms.
        """
        logger.info("Building new SpectroTemporalModel")
        
        # Define input shapes
        mel_spec_input = layers.Input(shape=(128, None, 1), name='mel_spectrogram')
        cqt_input = layers.Input(shape=(84, None, 1), name='constant_q_transform')
        
        # Process mel-spectrogram
        x_mel = self._create_conv_block(mel_spec_input, 32)
        x_mel = self._create_resnet_block(x_mel, 32)
        x_mel = layers.MaxPooling2D((2, 2))(x_mel)
        
        x_mel = self._create_conv_block(x_mel, 64)
        x_mel = self._create_resnet_block(x_mel, 64)
        x_mel = layers.MaxPooling2D((2, 2))(x_mel)
        
        x_mel = self._create_conv_block(x_mel, 128)
        x_mel = self._create_resnet_block(x_mel, 128)
        x_mel = layers.GlobalAveragePooling2D()(x_mel)
        
        # Process CQT
        x_cqt = self._create_conv_block(cqt_input, 32)
        x_cqt = self._create_resnet_block(x_cqt, 32)
        x_cqt = layers.MaxPooling2D((2, 2))(x_cqt)
        
        x_cqt = self._create_conv_block(x_cqt, 64)
        x_cqt = self._create_resnet_block(x_cqt, 64)
        x_cqt = layers.MaxPooling2D((2, 2))(x_cqt)
        
        x_cqt = self._create_conv_block(x_cqt, 128)
        x_cqt = self._create_resnet_block(x_cqt, 128)
        x_cqt = layers.GlobalAveragePooling2D()(x_cqt)
        
        # Combine features
        combined = layers.concatenate([x_mel, x_cqt])
        
        # Fully connected layers
        x = layers.Dense(256, activation='relu')(combined)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        self.model = models.Model(
            inputs=[mel_spec_input, cqt_input],
            outputs=output
        )
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model built successfully")
    
    def _create_conv_block(self, input_tensor, filters):
        """
        Create a standard convolutional block.
        
        Args:
            input_tensor: Input tensor
            filters: Number of filters
            
        Returns:
            Output tensor
        """
        x = layers.Conv2D(filters, (3, 3), padding='same')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    def _create_resnet_block(self, input_tensor, filters):
        """
        Create a ResNet-style residual block.
        
        Args:
            input_tensor: Input tensor
            filters: Number of filters
            
        Returns:
            Output tensor
        """
        x = self._create_conv_block(input_tensor, filters)
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Add residual connection
        x = layers.add([x, input_tensor])
        x = layers.Activation('relu')(x)
        return x
    
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
    
    def preprocess_input(self, mel_spectrogram, constant_q_transform):
        """
        Preprocess the input data for the model.
        
        Args:
            mel_spectrogram: Mel-spectrogram as a numpy array
            constant_q_transform: CQT as a numpy array
            
        Returns:
            Preprocessed inputs
        """
        # Add channel dimension if needed
        if len(mel_spectrogram.shape) == 2:
            mel_spectrogram = mel_spectrogram[..., np.newaxis]
            
        if len(constant_q_transform.shape) == 2:
            constant_q_transform = constant_q_transform[..., np.newaxis]
            
        # Normalize
        mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / (mel_spectrogram.std() + 1e-8)
        constant_q_transform = (constant_q_transform - constant_q_transform.mean()) / (constant_q_transform.std() + 1e-8)
        
        # Expand batch dimension
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
        constant_q_transform = np.expand_dims(constant_q_transform, axis=0)
        
        return {
            'mel_spectrogram': mel_spectrogram,
            'constant_q_transform': constant_q_transform
        }
    
    def train(self, train_data, validation_data=None, epochs=10, batch_size=32):
        """
        Train the model.
        
        Args:
            train_data: Training data dictionary with keys 'mel_spectrogram', 'constant_q_transform', 'labels'
            validation_data: Validation data with the same format as train_data (optional)
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
            [train_data['mel_spectrogram'], train_data['constant_q_transform']],
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
    
    def analyze(self, mel_spectrogram, constant_q_transform):
        """
        Analyze the spectro-temporal characteristics of an audio sample.
        
        Args:
            mel_spectrogram: Mel-spectrogram as a numpy array
            constant_q_transform: CQT as a numpy array
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing spectro-temporal features")
        
        # For demonstration purposes, we'll return random scores
        # In a real implementation, this would use the trained model
        
        # Placeholder for actual model prediction
        # processed_input = self.preprocess_input(mel_spectrogram, constant_q_transform)
        # prediction = self.model.predict([processed_input['mel_spectrogram'], processed_input['constant_q_transform']])
        # score = prediction[0][0]
        
        # For demonstration, return a random score between 0.0 and 1.0
        # 0.0 = completely authentic, 1.0 = definitely cloned
        score = np.random.uniform(0.3, 0.7)
        
        # Generate some example findings based on the score
        findings = []
        if score > 0.7:
            findings.append("High levels of spectral artifacting detected")
            findings.append("Unnatural harmonic distribution in upper frequencies")
        elif score > 0.5:
            findings.append("Moderate spectral anomalies detected")
            findings.append("Some inconsistencies in time-frequency representation")
        else:
            findings.append("Natural spectral characteristics observed")
            findings.append("Consistent time-frequency patterns")
        
        return {
            'score': float(score),
            'findings': findings
        } 