"""Acoustic audio signal synthesizer for industrial machinery (MIMII/ToyADMOS benchmark aligned)."""

from typing import Tuple
import numpy as np
from src.config import DEFAULT_AUDIO_CONFIG, AudioConfig


def generate_machine_audio(
    machine_type: str = "pump",
    condition: str = "normal",
    seed: int = 42,
    config: AudioConfig = DEFAULT_AUDIO_CONFIG,
) -> Tuple[np.ndarray, str]:
    """
    Synthesize physical acoustic emissions for industrial machinery.
    Conditions: 'normal', 'bearing_knock', 'cavitation', 'friction_screech'
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, config.duration_sec, config.n_samples, endpoint=False)

    # 1. Base nominal acoustic profile: 60Hz motor hum + harmonics + flow noise
    f0 = 60.0  # 60 Hz fundamental electrical/rotational frequency
    harmonics = (
        0.40 * np.sin(2.0 * np.pi * f0 * t)
        + 0.25 * np.sin(2.0 * np.pi * (2 * f0) * t)
        + 0.15 * np.sin(2.0 * np.pi * (4 * f0) * t)
        + 0.08 * np.sin(2.0 * np.pi * (8 * f0) * t)
    )
    # Ambient pink/fluid flow noise
    flow_noise = rng.normal(loc=0.0, scale=0.08, size=config.n_samples)
    audio = harmonics + flow_noise

    cond = condition.lower()
    if cond == "bearing_knock":
        # Periodic high-impact transient acoustic clicks (every ~80ms)
        click_interval = int(config.sampling_rate_hz * 0.08)
        for idx in range(click_interval // 2, config.n_samples, click_interval):
            burst_len = min(60, config.n_samples - idx)
            decay = np.exp(-np.linspace(0, 5, burst_len))
            audio[idx : idx + burst_len] += 0.85 * decay * np.sin(2 * np.pi * 1200.0 * t[:burst_len])
        label = "Bearing Knock / Impact"

    elif cond == "cavitation":
        # High-frequency turbulent hiss burst (2.5 kHz - 4 kHz band)
        hiss = rng.normal(loc=0.0, scale=0.35, size=config.n_samples)
        # Modulate with fluid pressure pulsation
        audio += hiss * (1.0 + 0.4 * np.sin(2 * np.pi * 12.0 * t))
        label = "Pump Cavitation / Void"

    elif cond == "friction_screech":
        # Intense high-frequency friction squeal (1850 Hz)
        audio += 0.65 * np.sin(2.0 * np.pi * 1850.0 * t)
        label = "Dry Friction / Rubbing"

    else:
        label = "Normal Operation"

    # Normalize audio to [-1.0, 1.0]
    peak = max(1e-5, float(np.max(np.abs(audio))))
    normalized_audio = np.clip(audio / peak, -1.0, 1.0).astype(np.float32)
    return normalized_audio, label
