"""Unsupervised Acoustic Autoencoder & Anomaly Detection Pipeline."""

import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_AUDIO_CONFIG, AudioConfig
from src.features import AcousticFeatureExtractor
from src.generator import generate_machine_audio


@dataclass
class AcousticAnomalyResult:
    is_anomaly: bool
    anomaly_score: float  # Reconstruction error MSE
    threshold: float
    health_status: str  # NORMAL, WARNING, CRITICAL
    diagnosed_sound_profile: str
    spectral_centroid_hz: float
    rms_energy: float


class AcousticAnomalyDetectorBundle:
    """Production acoustic anomaly detector trained exclusively on nominal operational audio."""

    def __init__(
        self,
        scaler: StandardScaler,
        autoencoder_pca: PCA,
        feature_extractor: AcousticFeatureExtractor,
        threshold: float,
        config: AudioConfig = DEFAULT_AUDIO_CONFIG,
    ):
        self.scaler = scaler
        self.autoencoder_pca = autoencoder_pca
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        self.config = config

    def predict_audio(self, audio: np.ndarray) -> AcousticAnomalyResult:
        """Extracts features, reconstructs through autoencoder bottleneck, and computes MSE."""
        feats = self.feature_extractor.extract_features(audio)
        scaled_feats = self.scaler.transform(feats.reshape(1, -1))

        # Project to latent bottleneck & reconstruct
        latent = self.autoencoder_pca.transform(scaled_feats)
        reconstructed = self.autoencoder_pca.inverse_transform(latent)

        # Reconstruction Error (Mean Squared Error)
        mse = float(np.mean((scaled_feats - reconstructed) ** 2))

        is_anomaly = mse >= self.threshold
        if mse >= self.threshold * 1.6:
            status_str = "CRITICAL"
        elif is_anomaly:
            status_str = "WARNING"
        else:
            status_str = "NORMAL"

        # Diagnostic spectral profiling
        centroid = float(feats[32])
        rolloff = float(feats[33])
        rms = float(feats[34])

        if not is_anomaly:
            profile = "Normal Operational Hum"
        elif centroid > 2200.0 or rolloff > 4500.0:
            profile = "High-Frequency Cavitation / Turbulent Flow"
        elif rms > 0.45:
            profile = "High-Energy Friction Squeal / Rubbing"
        else:
            profile = "Mechanical Impact / Bearing Knock"

        return AcousticAnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=round(mse, 4),
            threshold=round(self.threshold, 4),
            health_status=status_str,
            diagnosed_sound_profile=profile,
            spectral_centroid_hz=round(centroid, 1),
            rms_energy=round(rms, 4),
        )

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "AcousticAnomalyDetectorBundle":
        return joblib.load(filepath)


def train_acoustic_detector(
    n_normal_samples: int = 150,
    config: AudioConfig = DEFAULT_AUDIO_CONFIG,
) -> Tuple[AcousticAnomalyDetectorBundle, float]:
    """Train unsupervised autoencoder on purely normal acoustic emissions."""
    extractor = AcousticFeatureExtractor(config=config)
    normal_features = []

    for i in range(n_normal_samples):
        audio, _ = generate_machine_audio(condition="normal", seed=100 + i, config=config)
        feats = extractor.extract_features(audio)
        normal_features.append(feats)

    X = np.array(normal_features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality reduction bottleneck (36 dims -> 8 latent components)
    pca = PCA(n_components=8, random_state=42)
    latent = pca.fit_transform(X_scaled)
    reconstructed = pca.inverse_transform(latent)

    # 98th percentile reconstruction error on normal data establishes threshold
    train_errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)
    threshold = float(np.percentile(train_errors, 98.0) * 1.35)

    bundle = AcousticAnomalyDetectorBundle(
        scaler=scaler,
        autoencoder_pca=pca,
        feature_extractor=extractor,
        threshold=threshold,
        config=config,
    )
    return bundle, threshold
