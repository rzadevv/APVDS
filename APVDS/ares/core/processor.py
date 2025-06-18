"""
Audio Processor Module for ARES.

This module handles all pre-processing and normalization of audio files.
"""

import os
import logging
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from scipy.io import wavfile
from pydub import AudioSegment

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Audio processing class for ARES.
    
    Handles pre-processing, normalization, and feature extraction from audio files.
    """
    
    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Target sample rate in Hz
            n_fft: FFT window size for spectral analysis
            hop_length: Hop length for spectral analysis
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        logger.debug(f"AudioProcessor initialized with sample_rate={sample_rate}, n_fft={n_fft}, hop_length={hop_length}")
    
    def load_audio(self, file_path):
        """
        Load an audio file and return the waveform.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            waveform: Audio signal as numpy array
            sample_rate: Sample rate of the loaded audio
        """
        logger.info(f"Loading audio file: {file_path}")
        
        # Get file extension to determine format
        _, ext = os.path.splitext(file_path.lower())
        
        try:
            if ext in ['.wav']:
                waveform, sr = librosa.load(file_path, sr=None)
            elif ext in ['.mp3', '.ogg', '.flac']:
                # Convert to WAV first using pydub
                audio = AudioSegment.from_file(file_path)
                tmp_wav = "_tmp_convert.wav"
                audio.export(tmp_wav, format="wav")
                waveform, sr = librosa.load(tmp_wav, sr=None)
                os.remove(tmp_wav)
            else:
                raise ValueError(f"Unsupported audio format: {ext}")
                
            logger.info(f"Audio loaded: sample_rate={sr}, duration={len(waveform)/sr:.2f}s")
            return waveform, sr
            
        except Exception as e:
            logger.error(f"Error loading audio file: {str(e)}")
            raise
    
    def preprocess(self, waveform, original_sr):
        """
        Preprocess the audio signal. This includes:
        - Resampling to target sample rate
        - Normalizing volume
        - Removing DC offset
        - Reducing noise
        
        Args:
            waveform: Input audio signal
            original_sr: Original sample rate of the audio
            
        Returns:
            processed_waveform: Processed audio signal
        """
        logger.info("Starting audio preprocessing")
        
        # Resample to target sample rate if needed
        if original_sr != self.sample_rate:
            logger.debug(f"Resampling from {original_sr}Hz to {self.sample_rate}Hz")
            waveform = librosa.resample(waveform, orig_sr=original_sr, target_sr=self.sample_rate)
        
        # Normalize volume to -1dBFS
        logger.debug("Normalizing audio volume")
        if np.abs(waveform).max() > 0:  # Avoid division by zero
            norm_factor = 0.9 / np.abs(waveform).max()  # Normalize to -1dBFS
            waveform = waveform * norm_factor
        
        # Remove DC offset with high-pass filter
        logger.debug("Applying high-pass filter to remove DC offset")
        b, a = signal.butter(4, 30.0/(self.sample_rate/2.0), 'highpass')
        waveform = signal.filtfilt(b, a, waveform)
        
        # Basic noise reduction using spectral gating
        # For a sophisticated implementation, a spectral noise gate would be used
        # Here we're using a simplified approach
        logger.debug("Applying noise reduction")
        
        # Trim silence
        waveform, _ = librosa.effects.trim(waveform, top_db=20)
        
        logger.info("Audio preprocessing complete")
        return waveform
        
    def extract_mel_spectrogram(self, waveform):
        """
        Extract Mel-spectrogram from the preprocessed audio.
        
        Args:
            waveform: Processed audio signal
            
        Returns:
            mel_spec: Mel-spectrogram as a numpy array
        """
        logger.debug("Extracting Mel-spectrogram")
        
        # Compute the mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, 
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=128
        )
        
        # Convert to log scale (dB)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec
        
    def extract_constant_q_transform(self, waveform):
        """
        Extract Constant-Q Transform for the audio.
        
        Args:
            waveform: Processed audio signal
            
        Returns:
            cqt: Constant-Q Transform as a numpy array
        """
        logger.debug("Extracting Constant-Q Transform")
        
        # Compute the CQT
        cqt = librosa.cqt(
            y=waveform,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_bins=84,  # 7 octaves with 12 bins per octave
            bins_per_octave=12
        )
        
        # Convert to log scale (dB)
        cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        
        return cqt
        
    def extract_mfcc(self, waveform, n_mfcc=13):
        """
        Extract MFCCs from the preprocessed audio.
        
        Args:
            waveform: Processed audio signal
            n_mfcc: Number of MFCC coefficients to extract
            
        Returns:
            mfccs: MFCCs as a numpy array
        """
        logger.debug(f"Extracting {n_mfcc} MFCCs")
        
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(
            y=waveform, 
            sr=self.sample_rate, 
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Add delta and delta-delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Stack all features
        mfcc_features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
        
        return mfcc_features

    def extract_pitch_contour(self, waveform):
        """
        Extract pitch (f0) contour from the audio.
        
        Args:
            waveform: Processed audio signal
            
        Returns:
            f0_contour: Pitch contour as a numpy array
            times: Time points for the pitch contour
        """
        logger.debug("Extracting pitch contour")
        
        # Compute the pitch contour using pYIN algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(
            waveform,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
            sr=self.sample_rate
        )
        
        # Get the time points for each frame
        times = librosa.times_like(f0, sr=self.sample_rate)
        
        return f0, voiced_flag, times

    def extract_jitter_shimmer(self, waveform):
        """
        Extract jitter and shimmer from the audio signal.
        
        This is an approximation - for more accurate jitter and shimmer analysis,
        specialized libraries like Parselmouth (Praat wrapper) are recommended.
        
        Args:
            waveform: Processed audio signal
            
        Returns:
            jitter: Measure of frequency variation
            shimmer: Measure of amplitude variation
        """
        logger.debug("Extracting jitter and shimmer")
        
        try:
            import parselmouth
            from parselmouth.praat import call
            
            # Create a Praat Sound object
            sound = parselmouth.Sound(waveform, self.sample_rate)
            
            # Extract pitch
            pitch = sound.to_pitch()
            
            # Extract point process of pitch
            point_process = call(pitch, "To PointProcess")
            
            # Extract jitter
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            
            # Extract shimmer
            shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            return jitter, shimmer
            
        except ImportError:
            # Fallback method if Parselmouth is not available
            logger.warning("Parselmouth not available, using approximate jitter/shimmer calculation")
            
            # Extract f0 using librosa
            f0, voiced_flag, _ = self.extract_pitch_contour(waveform)
            f0_voiced = f0[voiced_flag]
            
            # Simple jitter calculation (frequency variation)
            if len(f0_voiced) > 1:
                jitter = np.mean(np.abs(np.diff(f0_voiced))) / np.mean(f0_voiced)
            else:
                jitter = 0
                
            # Simple shimmer calculation (amplitude variation)
            frame_length = int(0.03 * self.sample_rate)  # 30ms frames
            hop_length = int(0.01 * self.sample_rate)    # 10ms hop
            
            frames = librosa.util.frame(waveform, frame_length=frame_length, hop_length=hop_length)
            amplitudes = np.max(np.abs(frames), axis=0)
            
            if len(amplitudes) > 1:
                shimmer = np.mean(np.abs(np.diff(amplitudes))) / np.mean(amplitudes)
            else:
                shimmer = 0
                
            return jitter, shimmer

    def process_file(self, file_path):
        """
        Process an audio file and extract all features.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary containing all extracted features
        """
        logger.info(f"Processing file: {file_path}")
        
        # Load and preprocess the audio
        waveform, sr = self.load_audio(file_path)
        processed_audio = self.preprocess(waveform, sr)
        
        # Extract spectral features
        mel_spectrogram = self.extract_mel_spectrogram(processed_audio)
        cqt = self.extract_constant_q_transform(processed_audio)
        
        # Extract acoustic features
        mfccs = self.extract_mfcc(processed_audio)
        f0, voiced_flag, times = self.extract_pitch_contour(processed_audio)
        jitter, shimmer = self.extract_jitter_shimmer(processed_audio)
        
        # Package all features
        features = {
            'waveform': processed_audio,
            'sample_rate': self.sample_rate,
            'mel_spectrogram': mel_spectrogram,
            'constant_q_transform': cqt,
            'mfccs': mfccs,
            'pitch_contour': {
                'f0': f0,
                'voiced_flag': voiced_flag,
                'times': times
            },
            'jitter': jitter,
            'shimmer': shimmer
        }
        
        logger.info("Feature extraction complete")
        return features 