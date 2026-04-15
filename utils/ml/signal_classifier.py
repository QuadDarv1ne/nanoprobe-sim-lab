"""
TFLite Signal Classification with graceful degradation.

Supports: 'sstv', 'cw', 'rtty', 'fm', 'noise', 'unknown'
When tflite_runtime is unavailable, falls back to heuristic energy-based classification.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional TFLite dependency
TFLITE_AVAILABLE = False
try:
    import tflite_runtime.interpreter as tflite

    TFLITE_AVAILABLE = True
except ImportError:
    logger.debug("tflite_runtime not available, using heuristic classification")

SUPPORTED_CLASSES = ["sstv", "cw", "rtty", "fm", "noise", "unknown"]


class SignalClassifier:
    """Classify radio signals from complex IQ samples.

    Uses a TFLite model when available, otherwise falls back to
    simple heuristic classification based on spectral characteristics.
    """

    # Heuristic spectral-width thresholds (fraction of sample_rate)
    _FM_WIDTH = 0.05  # FM occupies >5% of spectrum
    _NARROW_WIDTH = 0.01  # CW/RTTY occupy <1%

    # Energy threshold for noise vs signal detection
    _NOISE_THRESHOLD = 0.02

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.available = False
        self._interpreter = None
        self._input_details = None
        self._output_details = None

        if model_path and TFLITE_AVAILABLE:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        """Load a TFLite model from file."""
        try:
            self._interpreter = tflite.Interpreter(model_path=str(path))
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self.available = True
            logger.info(f"TFLite model loaded: {path}")
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            self.available = False

    def classify(self, samples: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        """Classify signal from complex IQ samples.

        Args:
            samples: Complex IQ samples (numpy array of complex64/128).
            sample_rate: Sample rate in Hz.

        Returns:
            Tuple of (label, confidence) where label is one of SUPPORTED_CLASSES.
        """
        if samples is None or len(samples) == 0:
            logger.warning("Empty samples provided for classification")
            return "unknown", 0.0

        if self.available and self._interpreter is not None:
            return self._classify_tflite(samples, sample_rate)
        return self._classify_heuristic(samples, sample_rate)

    def _preprocess_samples(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Convert raw IQ samples to spectrogram features for model input.

        Returns a magnitude spectrogram suitable for TFLite model input.
        """
        # Segment into overlapping frames
        frame_size = min(1024, len(samples))
        hop = frame_size // 2
        frames = []
        for start in range(0, len(samples) - frame_size + 1, hop):
            frames.append(samples[start : start + frame_size])

        if not frames:
            frames = [samples[:frame_size]]

        frame_array = np.array(frames)

        # FFT magnitude spectrum
        spectrum = np.abs(np.fft.rfft(frame_array, n=frame_size))
        # Log-scale normalization
        spectrum = np.log1p(spectrum)
        # Normalize to [0, 1]
        spec_max = spectrum.max()
        if spec_max > 0:
            spectrum /= spec_max

        return spectrum.astype(np.float32)

    def _classify_tflite(self, samples: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        """Classify using TFLite model."""
        try:
            features = self._preprocess_samples(samples, sample_rate)
            # Ensure input shape matches model expectations
            input_shape = self._input_details[0]["shape"]
            if features.ndim < len(input_shape):
                features = np.expand_dims(features, axis=0)

            self._interpreter.set_tensor(self._input_details[0]["index"], features)
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(self._output_details[0]["index"])

            # Handle various output shapes
            if output.ndim > 1:
                output = output[0]
            if output.dtype == np.uint8:
                scale, zero_point = self._output_details[0].get("quantization", (1.0, 0))
                output = (output.astype(np.float32) - zero_point) * scale

            idx = int(np.argmax(output))
            confidence = float(output[idx])

            if idx < len(SUPPORTED_CLASSES):
                return SUPPORTED_CLASSES[idx], confidence
            return "unknown", confidence

        except Exception as e:
            logger.error(f"TFLite inference failed: {e}")
            return self._classify_heuristic(samples, sample_rate)

    def _classify_heuristic(self, samples: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        """Fallback heuristic classification based on spectral analysis.

        Uses energy levels and spectral width to distinguish:
        - noise (low energy)
        - fm (wide bandwidth)
        - cw/rtty (narrow bandwidth, different patterns)
        - sstv (medium bandwidth, specific frequency range)
        """
        try:
            # Compute power spectral density
            fft_result = np.fft.rfft(samples)
            power = np.abs(fft_result) ** 2
            power_norm = power / (power.max() + 1e-12)

            # Total energy
            total_energy = float(np.mean(np.abs(samples) ** 2))

            # Noise detection: very low energy
            if total_energy < self._NOISE_THRESHOLD:
                return "noise", min(0.95, 1.0 - total_energy / self._NOISE_THRESHOLD)

            # Spectral width: find bandwidth containing 90% of energy
            sorted_power = np.sort(power_norm)[::-1]
            cumsum = np.cumsum(sorted_power) / (sorted_power.sum() + 1e-12)
            bw_idx = int(np.searchsorted(cumsum, 0.9))
            bw_fraction = bw_idx / len(power_norm)

            # FM: wide bandwidth
            if bw_fraction > self._FM_WIDTH:
                confidence = min(0.9, bw_fraction / 0.2)
                return "fm", confidence

            # Find dominant frequency
            freqs = np.fft.rfftfreq(len(samples), 1.0 / sample_rate)
            dominant_freq = float(freqs[np.argmax(power)])

            # CW: very narrow, single tone
            if bw_fraction < self._NARROW_WIDTH * 0.3:
                return "cw", min(0.85, 0.5 + (self._NARROW_WIDTH - bw_fraction) * 50)

            # RTTY: narrow, two-tone pattern
            if bw_fraction < self._NARROW_WIDTH:
                # Check for two-tone characteristic
                peak_indices = self._find_top_peaks(power_norm, n=2)
                if len(peak_indices) >= 2:
                    return "rtty", 0.65
                return "cw", 0.5

            # SSTV: medium bandwidth, audio-range frequencies
            if dominant_freq < 4000:
                return "sstv", min(0.75, 0.4 + bw_fraction * 10)

            return "unknown", 0.3

        except Exception as e:
            logger.error(f"Heuristic classification failed: {e}")
            return "unknown", 0.0

    @staticmethod
    def _find_top_peaks(power: np.ndarray, n: int = 2) -> List[int]:
        """Find indices of the top n peaks in power spectrum."""
        # Simple peak detection: local maxima above threshold
        from scipy.signal import find_peaks as _find_peaks

        peaks, _ = _find_peaks(power, height=np.mean(power), distance=len(power) // 20)
        if len(peaks) == 0:
            return []
        top_indices = peaks[np.argsort(power[peaks])[::-1][:n]]
        return top_indices.tolist()

    def classify_from_file(self, wav_path: str) -> Tuple[str, float]:
        """Classify signal from a WAV file.

        Args:
            wav_path: Path to WAV file.

        Returns:
            Tuple of (label, confidence).
        """
        import wave

        wav_path = Path(wav_path)
        if not wav_path.exists():
            logger.error(f"WAV file not found: {wav_path}")
            return "unknown", 0.0

        try:
            with wave.open(str(wav_path), "r") as wf:
                frames = wf.readframes(wf.getnframes())
                sample_rate = wf.getframerate()
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                audio /= 32768.0

            # For WAV files we treat as real signal — convert to analytic (complex) signal
            from scipy.signal import hilbert

            samples = hilbert(audio)
            return self.classify(samples, sample_rate)

        except Exception as e:
            logger.error(f"Failed to classify WAV file {wav_path}: {e}")
            return "unknown", 0.0

    def get_available_classes(self) -> List[str]:
        """Return list of supported signal classes."""
        return list(SUPPORTED_CLASSES)
