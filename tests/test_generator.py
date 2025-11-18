"""Unit tests for synthetic industrial acoustic generator."""

import numpy as np
import pytest
from src.config import DEFAULT_AUDIO_CONFIG
from src.generator import generate_machine_audio


def test_generator_output_shape_and_range():
    config = DEFAULT_AUDIO_CONFIG
    audio, label = generate_machine_audio(machine_type="pump", condition="normal", seed=42, config=config)
    expected_len = int(config.sampling_rate_hz * config.duration_sec)
    assert len(audio) == expected_len
    assert label == "Normal Operation"
    assert -1.0 <= np.max(audio) <= 1.0
    assert -1.0 <= np.min(audio) <= 1.0


def test_generator_conditions():
    for cond in ["normal", "bearing_knock", "cavitation", "friction_screech"]:
        audio, label = generate_machine_audio(machine_type="motor", condition=cond, seed=123)
        assert len(audio) > 0
        assert isinstance(label, str)
        assert np.isfinite(audio).all()


def test_generator_deterministic_seed():
    audio1, _ = generate_machine_audio(seed=777)
    audio2, _ = generate_machine_audio(seed=777)
    np.testing.assert_allclose(audio1, audio2, rtol=1e-5)
