"""
ARES Engine - Main coordination module.

This module orchestrates the three parallel analysis streams and fusion engine.
"""

import os
import logging
import time
import json
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from ..core.processor import AudioProcessor
from ..models.spectrotemporal.cnn_model import SpectroTemporalModel
from ..models.acoustic.rnn_model import AcousticBiometricModel
from ..models.linguistic.transformer_model import LinguisticProsodyModel
from ..models.fusion.gradient_boosting import FusionModel

logger = logging.getLogger(__name__)

class ARESEngine:
    """
    Main ARES Engine that coordinates all analysis streams.
    
    The ARESEngine is responsible for:
    1. Preprocessing the audio input
    2. Running the three analysis streams in parallel
    3. Combining the results using the fusion model
    4. Generating the final verdict and report
    """
    
    def __init__(self, models_dir=None):
        """
        Initialize the ARES Engine.
        
        Args:
            models_dir: Directory containing pre-trained models
        """
        logger.info("Initializing ARES Engine")
        
        # Set models directory
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        self.models_dir = Path(models_dir)
        
        # Initialize the audio processor
        self.audio_processor = AudioProcessor()
        
        # Initialize analysis models
        logger.info("Loading analysis models")
        self.load_models()
        
        logger.info("ARES Engine initialized successfully")
        
    def load_models(self):
        """
        Load all the analysis models.
        
        This includes the three stream models and the fusion model.
        """
        # Stream A: Spectro-Temporal Analysis Model
        self.spectro_model = SpectroTemporalModel()
        
        # Stream B: Acoustic & Biometric Analysis Model
        self.acoustic_model = AcousticBiometricModel()
        
        # Stream C: Linguistic & Prosodic Analysis Model
        self.linguistic_model = LinguisticProsodyModel()
        
        # Fusion Model
        self.fusion_model = FusionModel()
    
    def analyze_file(self, file_path):
        """
        Analyze a single audio file.
        
        Args:
            file_path: Path to the audio file to analyze
            
        Returns:
            Dictionary containing analysis results and confidence scores
        """
        logger.info(f"Starting analysis of file: {file_path}")
        start_time = time.time()
        
        # Process the audio file and extract features
        try:
            features = self.audio_processor.process_file(file_path)
        except Exception as e:
            logger.error(f"Error processing audio file: {str(e)}")
            return {
                'classification': 'Error',
                'confidence_score': 0.0,
                'evidence': {
                    'error': str(e)
                }
            }
        
        # Run the three analysis streams in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit the tasks
            stream_a_future = executor.submit(
                self.run_stream_a, 
                features['mel_spectrogram'], 
                features['constant_q_transform']
            )
            
            stream_b_future = executor.submit(
                self.run_stream_b,
                features['mfccs'],
                features['pitch_contour'],
                features['jitter'],
                features['shimmer']
            )
            
            stream_c_future = executor.submit(
                self.run_stream_c,
                features['waveform'],
                features['sample_rate']
            )
            
            # Get the results
            stream_a_result = stream_a_future.result()
            stream_b_result = stream_b_future.result()
            stream_c_result = stream_c_future.result()
        
        # Combine the results using the fusion model
        fusion_result = self.run_fusion(
            stream_a_result,
            stream_b_result,
            stream_c_result
        )
        
        # Prepare the final result
        result = {
            'classification': 'Authentic' if fusion_result['score'] < 0.5 else 'Cloned (AI-Generated)',
            'confidence_score': (1 - fusion_result['score']) * 100 if fusion_result['score'] < 0.5 else fusion_result['score'] * 100,
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
                    'decision_factors': fusion_result['decision_factors']
                }
            },
            'analysis_time': time.time() - start_time
        }
        
        logger.info(f"Analysis complete. Classification: {result['classification']}, " +
                   f"Confidence: {result['confidence_score']:.2f}%, " +
                   f"Time: {result['analysis_time']:.2f}s")
        
        return result
    
    def run_stream_a(self, mel_spectrogram, constant_q_transform):
        """
        Run the Spectro-Temporal Analysis stream (Stream A).
        
        Args:
            mel_spectrogram: Extracted mel-spectrogram
            constant_q_transform: Extracted CQT
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Running Spectro-Temporal Analysis (Stream A)")
        
        try:
            # Run the CNN model
            result = self.spectro_model.analyze(mel_spectrogram, constant_q_transform)
            
            # For demonstration, create some sample findings
            # In a real implementation, these would come from the model
            score = result.get('score', 0.5)  # Default to 0.5 if not provided
            
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
                'score': score,
                'findings': findings
            }
            
        except Exception as e:
            logger.error(f"Error in Spectro-Temporal Analysis: {str(e)}")
            return {
                'score': 0.5,  # Neutral score on error
                'findings': [f"Error: {str(e)}"]
            }
    
    def run_stream_b(self, mfccs, pitch_contour, jitter, shimmer):
        """
        Run the Acoustic & Biometric Analysis stream (Stream B).
        
        Args:
            mfccs: Extracted MFCC features
            pitch_contour: Extracted pitch contour
            jitter: Measured jitter
            shimmer: Measured shimmer
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Running Acoustic & Biometric Analysis (Stream B)")
        
        try:
            # Package the features
            features = {
                'mfccs': mfccs,
                'pitch_contour': pitch_contour,
                'jitter': jitter,
                'shimmer': shimmer
            }
            
            # Run the RNN model
            result = self.acoustic_model.analyze(features)
            
            # For demonstration, create some sample findings
            # In a real implementation, these would come from the model
            score = result.get('score', 0.5)  # Default to 0.5 if not provided
            
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
                'score': score,
                'findings': findings
            }
            
        except Exception as e:
            logger.error(f"Error in Acoustic & Biometric Analysis: {str(e)}")
            return {
                'score': 0.5,  # Neutral score on error
                'findings': [f"Error: {str(e)}"]
            }
    
    def run_stream_c(self, waveform, sample_rate):
        """
        Run the Linguistic & Prosodic Analysis stream (Stream C).
        
        Args:
            waveform: Processed audio waveform
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Running Linguistic & Prosodic Analysis (Stream C)")
        
        try:
            # Run the Transformer model
            result = self.linguistic_model.analyze(waveform, sample_rate)
            
            # For demonstration, create some sample findings
            # In a real implementation, these would come from the model
            score = result.get('score', 0.5)  # Default to 0.5 if not provided
            
            findings = []
            if score > 0.7:
                findings.append("Unnatural pause patterns detected")
                findings.append("Mismatch between semantic content and prosodic delivery")
            elif score > 0.5:
                findings.append("Some inconsistencies in prosodic patterns")
                findings.append("Partial emotional-semantic mismatch detected")
            else:
                findings.append("Natural linguistic flow and prosody")
                findings.append("Consistent emotional-semantic alignment")
                
            return {
                'score': score,
                'findings': findings
            }
            
        except Exception as e:
            logger.error(f"Error in Linguistic & Prosodic Analysis: {str(e)}")
            return {
                'score': 0.5,  # Neutral score on error
                'findings': [f"Error: {str(e)}"]
            }
    
    def run_fusion(self, stream_a_result, stream_b_result, stream_c_result):
        """
        Run the Fusion & Classification Engine.
        
        Args:
            stream_a_result: Results from Stream A
            stream_b_result: Results from Stream B
            stream_c_result: Results from Stream C
            
        Returns:
            Dictionary with the final analysis result
        """
        logger.info("Running Fusion & Classification Engine")
        
        try:
            # Extract the scores
            scores = [
                stream_a_result['score'],
                stream_b_result['score'],
                stream_c_result['score']
            ]
            
            # Run the fusion model
            fusion_result = self.fusion_model.analyze(
                stream_a_score=stream_a_result['score'],
                stream_b_score=stream_b_result['score'],
                stream_c_score=stream_c_result['score']
            )
            
            # Determine which streams contributed most to the decision
            # For simplicity, we'll just rank them by their scores
            stream_names = ['Spectro-Temporal', 'Acoustic-Biometric', 'Linguistic-Prosodic']
            ranked_streams = sorted(
                zip(stream_names, scores),
                key=lambda x: abs(x[1] - 0.5),  # Sort by distance from neutral (0.5)
                reverse=True
            )
            
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
                'score': fusion_result['score'],
                'decision_factors': decision_factors
            }
            
        except Exception as e:
            logger.error(f"Error in Fusion & Classification: {str(e)}")
            return {
                'score': 0.5,  # Neutral score on error
                'decision_factors': [f"Error: {str(e)}"]
            }
    
    def analyze_stream(self, audio_stream, chunk_size=1024, sample_rate=16000):
        """
        Analyze a real-time audio stream.
        
        Args:
            audio_stream: Real-time audio stream
            chunk_size: Size of audio chunks to process
            sample_rate: Sample rate of the audio stream
            
        Returns:
            Generator yielding analysis results
        """
        logger.info("Starting real-time audio stream analysis")
        
        # Buffer for accumulating audio chunks
        buffer = []
        buffer_duration = 0
        target_duration = 5  # seconds
        
        while True:
            try:
                # Get the next chunk of audio
                chunk = audio_stream.read(chunk_size)
                if not chunk:
                    break
                    
                # Add the chunk to the buffer
                buffer.append(chunk)
                buffer_duration += chunk_size / sample_rate
                
                # If we have enough data, analyze it
                if buffer_duration >= target_duration:
                    # Convert buffer to a single array
                    audio_data = np.concatenate(buffer)
                    
                    # Process and analyze
                    features = self.audio_processor.preprocess(audio_data, sample_rate)
                    result = self.analyze_features(features)
                    
                    # Yield the result
                    yield result
                    
                    # Reset the buffer (with overlap for continuity)
                    overlap = int(1 * sample_rate)  # 1 second overlap
                    if len(audio_data) > overlap:
                        buffer = [audio_data[-overlap:]]
                        buffer_duration = overlap / sample_rate
                    else:
                        buffer = []
                        buffer_duration = 0
                        
            except Exception as e:
                logger.error(f"Error in stream analysis: {str(e)}")
                yield {
                    'classification': 'Error',
                    'confidence_score': 0.0,
                    'evidence': {
                        'error': str(e)
                    }
                }
    
    def analyze_features(self, features):
        """
        Analyze pre-extracted features.
        
        This is a helper method for real-time analysis.
        
        Args:
            features: Dictionary of pre-extracted features
            
        Returns:
            Dictionary with analysis results
        """
        # Run the three analysis streams
        stream_a_result = self.run_stream_a(
            features['mel_spectrogram'], 
            features['constant_q_transform']
        )
        
        stream_b_result = self.run_stream_b(
            features['mfccs'],
            features['pitch_contour'],
            features['jitter'],
            features['shimmer']
        )
        
        stream_c_result = self.run_stream_c(
            features['waveform'],
            features['sample_rate']
        )
        
        # Combine the results
        fusion_result = self.run_fusion(
            stream_a_result,
            stream_b_result,
            stream_c_result
        )
        
        # Prepare the final result
        result = {
            'classification': 'Authentic' if fusion_result['score'] < 0.5 else 'Cloned (AI-Generated)',
            'confidence_score': (1 - fusion_result['score']) * 100 if fusion_result['score'] < 0.5 else fusion_result['score'] * 100,
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
                    'decision_factors': fusion_result['decision_factors']
                }
            }
        }
        
        return result
        
    def save_report(self, result, output_path=None):
        """
        Save an analysis report to a JSON file.
        
        Args:
            result: Analysis result dictionary
            output_path: Path to save the report (if None, will generate one)
            
        Returns:
            Path to the saved report
        """
        if output_path is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = f"ares_report_{timestamp}.json"
            
        try:
            # Convert numpy arrays to lists for JSON serialization
            result_copy = self._prepare_for_json(result)
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(result_copy, f, indent=2)
                
            logger.info(f"Report saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            return None
            
    def _prepare_for_json(self, obj):
        """
        Prepare an object for JSON serialization by converting numpy types.
        
        Args:
            obj: The object to convert
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()
        else:
            return obj 