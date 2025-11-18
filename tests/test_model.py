"""Unit tests for unsupervised acoustic autoencoder model."""

import tempfile
from pathlib import Path
import pytest
from src.config import DEFAULT_AUDIO_CONFIG
from src.generator import generate_machine_audio
from src.model import train_acoustic_detector, AcousticAnomalyDetectorBundle


@pytest.fixture(scope="module")
def trained_bundle():
    bundle, threshold = train_acoustic_detector(n_normal_samples=60, config=DEFAULT_AUDIO_CONFIG)
    return bundle, threshold


def test_bundle_normal_inference(trained_bundle):
    bundle, threshold = trained_bundle
    audio_norm, _ = generate_machine_audio(condition="normal", seed=42)
    res = bundle.predict_audio(audio_norm)
    assert not res.is_anomaly
    assert res.health_status == "NORMAL"
    assert res.anomaly_score < threshold


def test_bundle_anomaly_detection(trained_bundle):
    bundle, _ = trained_bundle
    # Bearing knock should trigger reconstruction error
    audio_knock, _ = generate_machine_audio(condition="bearing_knock", seed=42)
    res = bundle.predict_audio(audio_knock)
    assert res.is_anomaly
    assert res.health_status in ["WARNING", "CRITICAL"]


def test_bundle_save_and_load(trained_bundle):
    bundle, _ = trained_bundle
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "bundle.joblib"
        bundle.save(str(model_path))
        loaded = AcousticAnomalyDetectorBundle.load(str(model_path))

        audio, _ = generate_machine_audio(condition="cavitation", seed=10)
        res_orig = bundle.predict_audio(audio)
        res_loaded = loaded.predict_audio(audio)

        assert res_orig.is_anomaly == res_loaded.is_anomaly
        assert res_orig.anomaly_score == res_loaded.anomaly_score
