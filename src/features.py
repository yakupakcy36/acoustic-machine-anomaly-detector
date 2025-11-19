"""Acoustic feature extractor: Log-Mel spectrogram, Spectral Centroid, Rolloff, & ZCR."""

import numpy as np
from src.config import DEFAULT_AUDIO_CONFIG, AudioConfig


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Constructs triangular Mel filterbank matrix."""
    f_min = 0.0
    f_max = float(sr // 2)
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    weights = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center > left:
            weights[i, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            weights[i, center:right] = (right - np.arange(center, right)) / (right - center)

    return weights


class AcousticFeatureExtractor:
    """Extracts compact acoustic representations for machine health modeling."""

    def __init__(self, config: AudioConfig = DEFAULT_AUDIO_CONFIG):
        self.config = config
        self.mel_filters = create_mel_filterbank(
            sr=config.sampling_rate_hz,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
        )

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extracts 1D feature vector:
        - Mean Log-Mel Energy across 32 bands
        - Spectral Centroid
        - Spectral Rolloff (85%)
        - RMS Energy
        - Zero-Crossing Rate (ZCR)
        """
        # Ensure audio has at least n_fft samples by zero-padding if necessary
        if len(audio) < self.config.n_fft:
            pad_width = self.config.n_fft - len(audio)
            audio = np.pad(audio, (0, pad_width), mode="constant")

        # 1. Framing & STFT
        hop = self.config.hop_length
        n_fft = self.config.n_fft
        num_frames = max(1, (len(audio) - n_fft) // hop + 1)

        window = np.hanning(n_fft)
        stft_matrix = np.empty((num_frames, n_fft // 2 + 1), dtype=np.float32)

        for i in range(num_frames):
            frame = audio[i * hop : i * hop + n_fft] * window
            fft_spec = np.fft.rfft(frame, n=n_fft)
            stft_matrix[i] = np.abs(fft_spec) ** 2

        # 2. Log-Mel Spectrogram Energy
        mel_spec = np.dot(stft_matrix, self.mel_filters.T)
        log_mel = np.log(np.maximum(1e-6, mel_spec))
        mean_log_mel = np.mean(log_mel, axis=0)  # Shape: (32,)

        # 3. Spectral Centroid
        freqs = np.linspace(0.0, self.config.sampling_rate_hz // 2, n_fft // 2 + 1)
        power_sum = np.sum(stft_matrix, axis=1) + 1e-6
        centroids = np.sum(stft_matrix * freqs, axis=1) / power_sum
        mean_centroid = float(np.mean(centroids))

        # 4. Spectral Rolloff (85% energy threshold)
        cumsum = np.cumsum(stft_matrix, axis=1)
        threshold = 0.85 * cumsum[:, -1:]
        rolloff_bins = np.argmax(cumsum >= threshold, axis=1)
        mean_rolloff = float(np.mean(freqs[rolloff_bins]))

        # 5. RMS Energy & Zero Crossing Rate
        rms = float(np.sqrt(np.mean(audio ** 2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0)

        # Assemble compact 36-dimensional feature vector
        scalar_features = np.array([mean_centroid, mean_rolloff, rms, zcr], dtype=np.float32)
        return np.concatenate([mean_log_mel, scalar_features])
