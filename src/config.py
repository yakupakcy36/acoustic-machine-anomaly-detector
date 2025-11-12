"""Configuration settings for Industrial Acoustic Machine Anomaly Detector."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioConfig:
    """Master configuration for acoustic signal sampling, spectrogram extraction, and autoencoder."""
    sampling_rate_hz: int = 16000
    duration_sec: float = 2.0
    n_samples: int = 32000  # sampling_rate * duration
    n_fft: int = 512
    hop_length: int = 256
    n_mels: int = 32
    anomaly_threshold: float = 0.28
    artifacts_dir: Path = Path("artifacts")
    model_filename: str = "acoustic_autoencoder_bundle.joblib"


DEFAULT_AUDIO_CONFIG = AudioConfig()
