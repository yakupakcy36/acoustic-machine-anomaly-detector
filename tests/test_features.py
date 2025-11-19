"""Unit tests for acoustic feature extraction & log-mel filterbanks."""

import numpy as np
import pytest
from src.config import DEFAULT_AUDIO_CONFIG
from src.features import AcousticFeatureExtractor
from src.generator import generate_machine_audio


def test_feature_extractor_dimension():
    extractor = AcousticFeatureExtractor(config=DEFAULT_AUDIO_CONFIG)
    audio, _ = generate_machine_audio(condition="normal", seed=10)
    feats = extractor.extract_features(audio)

    # 32 mel bands + 4 spectral summary statistics = 36 features
    assert feats.shape == (36,)
    assert np.isfinite(feats).all()
    # Spectral centroid and rolloff must be positive
    assert feats[32] > 0.0  # Centroid
    assert feats[33] > 0.0  # Rolloff


def test_short_audio_padding():
    # Audio with fewer samples than n_fft should be padded without error
    extractor = AcousticFeatureExtractor(config=DEFAULT_AUDIO_CONFIG)
    short_audio = np.random.randn(256).astype(np.float32)
    feats = extractor.extract_features(short_audio)
    assert feats.shape == (36,)
    assert np.isfinite(feats).all()
