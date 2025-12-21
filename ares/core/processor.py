"""
Audio Processor Module for ARES v2.0.

Enhanced audio processing with augmentation, Wav2Vec2 feature extraction,
and robust preprocessing for real-world audio conditions.
"""

import os
import logging
import tempfile
from typing import Dict, Any, Optional, Tuple
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from pydub import AudioSegment

from .config import get_config, ARESConfig

logger = logging.getLogger(__name__)


class AudioAugmenter:
    """
    Audio augmentation pipeline for training robustness.
    
    Applies various augmentations to simulate real-world audio conditions:
    - Additive noise
    - Codec compression artifacts
    - Pitch shifting
    - Time stretching
    """
    
    def __init__(self, config: Optional[ARESConfig] = None):
        """Initialize with augmentation configuration."""
        self.config = config or get_config()
        self.aug_config = self.config.augmentation
        self._augmentations = None
        
    def _build_pipeline(self):
        """Build the audiomentations pipeline (lazy initialization)."""
        if self._augmentations is not None:
            return self._augmentations
            
        try:
            import audiomentations as am
            
            transforms = []
            
            # Additive noise
            if self.aug_config.noise_prob > 0:
                transforms.append(am.AddGaussianNoise(
                    min_amplitude=0.001,
                    max_amplitude=0.015,
                    p=self.aug_config.noise_prob
                ))
            
            # Pitch shifting
            if self.aug_config.pitch_shift_prob > 0:
                transforms.append(am.PitchShift(
                    min_semitones=-self.aug_config.pitch_shift_semitones,
                    max_semitones=self.aug_config.pitch_shift_semitones,
                    p=self.aug_config.pitch_shift_prob
                ))
            
            # Time stretching
            if self.aug_config.time_stretch_prob > 0:
                transforms.append(am.TimeStretch(
                    min_rate=self.aug_config.time_stretch_min_rate,
                    max_rate=self.aug_config.time_stretch_max_rate,
                    p=self.aug_config.time_stretch_prob
                ))
            
            # Low-pass filter (simulates phone/codec quality)
            transforms.append(am.LowPassFilter(
                min_cutoff_freq=4000,
                max_cutoff_freq=7500,
                p=0.2
            ))
            
            # Gain variation
            transforms.append(am.Gain(
                min_gain_db=-6,
                max_gain_db=6,
                p=0.3
            ))
            
            self._augmentations = am.Compose(transforms)
            logger.info("Audio augmentation pipeline initialized")
            
        except ImportError:
            logger.warning("audiomentations not installed. Augmentation disabled.")
            self._augmentations = None
            
        return self._augmentations
    
    def augment(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply augmentations to the audio.
        
        Args:
            waveform: Input audio as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Augmented audio waveform
        """
        if not self.aug_config.enabled:
            return waveform
            
        pipeline = self._build_pipeline()
        if pipeline is None:
            return waveform
            
        try:
            augmented = pipeline(samples=waveform, sample_rate=sample_rate)
            return augmented
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}. Returning original audio.")
            return waveform


class AudioProcessor:
    """
    Enhanced audio processing class for ARES v2.0.
    
    Handles pre-processing, normalization, feature extraction,
    and augmentation for audio deepfake detection.
    """
    
    def __init__(self, config: Optional[ARESConfig] = None):
        """
        Initialize the audio processor.
        
        Args:
            config: ARES configuration (uses default if not provided)
        """
        self.config = config or get_config()
        self.sample_rate = self.config.audio.sample_rate
        self.n_fft = self.config.audio.n_fft
        self.hop_length = self.config.audio.hop_length
        self.n_mels = self.config.audio.n_mels
        self.n_mfcc = self.config.audio.n_mfcc
        
        # Augmentation pipeline
        self.augmenter = AudioAugmenter(self.config)
        
        logger.debug(f"AudioProcessor initialized: sr={self.sample_rate}, "
                    f"n_fft={self.n_fft}, hop={self.hop_length}")
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load an audio file and return the waveform.
        
        Supports: WAV, MP3, OGG, FLAC, M4A, AAC
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (waveform, sample_rate)
        """
        logger.info(f"Loading audio: {file_path}")
        
        _, ext = os.path.splitext(file_path.lower())
        
        try:
            # Try direct loading with librosa first
            if ext in ['.wav', '.flac']:
                waveform, sr = librosa.load(file_path, sr=None, mono=True)
            
            # Use pydub for compressed formats
            elif ext in ['.mp3', '.ogg', '.m4a', '.aac', '.wma']:
                audio = AudioSegment.from_file(file_path)
                # Convert to mono if stereo
                if audio.channels > 1:
                    audio = audio.set_channels(1)
                
                # Export to temp WAV
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name
                    audio.export(tmp_path, format='wav')
                
                waveform, sr = librosa.load(tmp_path, sr=None, mono=True)
                os.unlink(tmp_path)
            
            else:
                # Try generic loading
                waveform, sr = librosa.load(file_path, sr=None, mono=True)
            
            duration = len(waveform) / sr
            logger.info(f"Audio loaded: {sr}Hz, {duration:.2f}s")
            
            # Validate duration
            if duration < self.config.audio.min_duration:
                raise ValueError(f"Audio too short: {duration:.2f}s < {self.config.audio.min_duration}s")
            
            if duration > self.config.audio.max_duration:
                logger.warning(f"Audio truncated from {duration:.2f}s to {self.config.audio.max_duration}s")
                max_samples = int(self.config.audio.max_duration * sr)
                waveform = waveform[:max_samples]
            
            return waveform, sr
            
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise
    
    def preprocess(self, waveform: np.ndarray, original_sr: int, 
                   apply_augmentation: bool = False) -> np.ndarray:
        """
        Preprocess the audio signal.
        
        Steps:
        1. Resample to target sample rate
        2. Normalize volume
        3. Remove DC offset (high-pass filter)
        4. Trim silence
        5. Optionally apply augmentation
        
        Args:
            waveform: Input audio signal
            original_sr: Original sample rate
            apply_augmentation: Whether to apply augmentation
            
        Returns:
            Processed audio signal
        """
        logger.debug("Preprocessing audio...")
        
        # Resample if needed
        if original_sr != self.sample_rate:
            logger.debug(f"Resampling: {original_sr}Hz -> {self.sample_rate}Hz")
            waveform = librosa.resample(waveform, orig_sr=original_sr, target_sr=self.sample_rate)
        
        # Normalize to -1dBFS (0.9 peak amplitude)
        max_amp = np.abs(waveform).max()
        if max_amp > 0:
            waveform = waveform * (0.9 / max_amp)
        
        # High-pass filter to remove DC offset and low-frequency rumble
        nyquist = self.sample_rate / 2.0
        cutoff = 80.0 / nyquist  # 80Hz high-pass
        if cutoff < 1.0:
            b, a = signal.butter(4, cutoff, 'highpass')
            waveform = signal.filtfilt(b, a, waveform)
        
        # Trim leading/trailing silence
        waveform, _ = librosa.effects.trim(waveform, top_db=25)
        
        # Apply augmentation if requested (training mode)
        if apply_augmentation:
            waveform = self.augmenter.augment(waveform, self.sample_rate)
        
        logger.debug(f"Preprocessing complete: {len(waveform)/self.sample_rate:.2f}s")
        return waveform
    
    def extract_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """Extract Mel-spectrogram from audio."""
        logger.debug("Extracting Mel-spectrogram")
        
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, 
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=80,
            fmax=7600
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_constant_q_transform(self, waveform: np.ndarray) -> np.ndarray:
        """Extract Constant-Q Transform."""
        logger.debug("Extracting CQT")
        
        cqt = librosa.cqt(
            y=waveform,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_bins=84,  # 7 octaves
            bins_per_octave=12
        )
        
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        return cqt_db
    
    def extract_mfcc(self, waveform: np.ndarray) -> np.ndarray:
        """Extract MFCCs with delta and delta-delta features."""
        logger.debug(f"Extracting {self.n_mfcc} MFCCs")
        
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(
            y=waveform, 
            sr=self.sample_rate, 
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Add delta and delta-delta features
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Stack all features [n_mfcc * 3, time]
        return np.vstack([mfccs, delta, delta2])
    
    def extract_pitch_contour(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract pitch (F0) contour using pYIN."""
        logger.debug("Extracting pitch contour")
        
        f0, voiced_flag, voiced_probs = librosa.pyin(
            waveform,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
            sr=self.sample_rate
        )
        
        times = librosa.times_like(f0, sr=self.sample_rate, hop_length=self.hop_length)
        
        return {
            'f0': f0,
            'voiced_flag': voiced_flag,
            'voiced_probs': voiced_probs,
            'times': times
        }
    
    def extract_jitter_shimmer(self, waveform: np.ndarray) -> Tuple[float, float]:
        """
        Extract jitter and shimmer using Parselmouth (Praat).
        
        Jitter: Frequency perturbation (voice unsteadiness)
        Shimmer: Amplitude perturbation
        """
        logger.debug("Extracting jitter and shimmer")
        
        try:
            import parselmouth
            from parselmouth.praat import call
            
            sound = parselmouth.Sound(waveform, self.sample_rate)
            pitch = sound.to_pitch()
            point_process = call(pitch, "To PointProcess")
            
            # Jitter (local) - typical range: 0 to 0.02 (0-2%)
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            
            # Shimmer (local) - typical range: 0 to 0.1 (0-10%)
            shimmer = call([sound, point_process], "Get shimmer (local)", 
                          0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            # Handle NaN values
            jitter = 0.0 if np.isnan(jitter) else jitter
            shimmer = 0.0 if np.isnan(shimmer) else shimmer
            
            return jitter, shimmer
            
        except ImportError:
            logger.warning("Parselmouth not available, using fallback jitter/shimmer estimation")
            return self._estimate_jitter_shimmer_fallback(waveform)
    
    def _estimate_jitter_shimmer_fallback(self, waveform: np.ndarray) -> Tuple[float, float]:
        """Fallback jitter/shimmer estimation without Parselmouth."""
        # Extract pitch
        pitch_data = self.extract_pitch_contour(waveform)
        f0 = pitch_data['f0']
        voiced = pitch_data['voiced_flag']
        
        f0_voiced = f0[voiced & ~np.isnan(f0)]
        
        # Jitter approximation
        if len(f0_voiced) > 1:
            jitter = np.mean(np.abs(np.diff(f0_voiced))) / np.mean(f0_voiced)
        else:
            jitter = 0.0
        
        # Shimmer approximation
        frame_length = int(0.03 * self.sample_rate)
        hop = int(0.01 * self.sample_rate)
        
        if len(waveform) >= frame_length:
            frames = librosa.util.frame(waveform, frame_length=frame_length, hop_length=hop)
            amplitudes = np.max(np.abs(frames), axis=0)
            
            if len(amplitudes) > 1:
                shimmer = np.mean(np.abs(np.diff(amplitudes))) / np.mean(amplitudes)
            else:
                shimmer = 0.0
        else:
            shimmer = 0.0
        
        return float(jitter), float(shimmer)
    
    def extract_formants(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract formant frequencies (F1-F4).
        
        Formants are resonance frequencies of the vocal tract,
        useful for detecting unnatural voice characteristics.
        """
        logger.debug("Extracting formants")
        
        try:
            import parselmouth
            from parselmouth.praat import call
            
            sound = parselmouth.Sound(waveform, self.sample_rate)
            formant = sound.to_formant_burg()
            
            num_frames = formant.get_number_of_frames()
            times = np.array([formant.get_time_from_frame_number(i+1) 
                             for i in range(num_frames)])
            
            formants = {}
            for f_num in range(1, 5):  # F1-F4
                f_values = np.array([formant.get_value_at_time(f_num, t) 
                                    for t in times])
                formants[f'F{f_num}'] = f_values
            
            formants['times'] = times
            return formants
            
        except ImportError:
            logger.warning("Parselmouth not available for formant extraction")
            return {'F1': np.array([]), 'F2': np.array([]), 
                    'F3': np.array([]), 'F4': np.array([]), 'times': np.array([])}
    
    def extract_spectral_features(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract additional spectral features.
        
        These capture timbral qualities useful for detecting synthetic audio.
        """
        logger.debug("Extracting spectral features")
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=waveform, sr=self.sample_rate, 
            n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=waveform, sr=self.sample_rate, 
            n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=waveform, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Spectral flatness (tonality measure)
        spectral_flatness = librosa.feature.spectral_flatness(
            y=waveform, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            waveform, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]
        
        return {
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_flatness': spectral_flatness,
            'zero_crossing_rate': zcr
        }
    
    def process_file(self, file_path: str, 
                     apply_augmentation: bool = False) -> Dict[str, Any]:
        """
        Process an audio file and extract all features.
        
        Args:
            file_path: Path to the audio file
            apply_augmentation: Whether to apply augmentation (training mode)
            
        Returns:
            Dictionary containing all extracted features
        """
        logger.info(f"Processing: {file_path}")
        
        # Load and preprocess
        waveform, sr = self.load_audio(file_path)
        processed = self.preprocess(waveform, sr, apply_augmentation)
        
        # Extract all features
        features = {
            # Raw audio
            'waveform': processed,
            'sample_rate': self.sample_rate,
            
            # Spectrograms (for Stream A)
            'mel_spectrogram': self.extract_mel_spectrogram(processed),
            'constant_q_transform': self.extract_constant_q_transform(processed),
            
            # MFCCs (for Stream B)
            'mfccs': self.extract_mfcc(processed),
            
            # Pitch and voice quality (for Stream B)
            'pitch_contour': self.extract_pitch_contour(processed),
            'jitter': self.extract_jitter_shimmer(processed)[0],
            'shimmer': self.extract_jitter_shimmer(processed)[1],
            
            # Formants (for Stream B/C)
            'formants': self.extract_formants(processed),
            
            # Spectral features (for all streams)
            'spectral_features': self.extract_spectral_features(processed),
        }
        
        logger.info("Feature extraction complete")
        return features
    
    def get_raw_waveform_for_wav2vec(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Get minimally processed waveform for Wav2Vec2 input.
        
        Wav2Vec2 expects 16kHz audio with minimal preprocessing.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (waveform at 16kHz, sample_rate)
        """
        waveform, sr = self.load_audio(file_path)
        
        # Resample to 16kHz (Wav2Vec2 requirement)
        if sr != 16000:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        
        # Light normalization only
        max_amp = np.abs(waveform).max()
        if max_amp > 0:
            waveform = waveform / max_amp * 0.95
        
        return waveform, 16000