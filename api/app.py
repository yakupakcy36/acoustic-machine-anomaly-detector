"""FastAPI application for Industrial Acoustic Machine Anomaly Detection."""

from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, status

from src.config import DEFAULT_AUDIO_CONFIG
from src.generator import generate_machine_audio
from src.model import train_acoustic_detector, AcousticAnomalyDetectorBundle
from api.schemas import (
    AcousticSimulationRequest,
    RawAudioAnalysisRequest,
    AcousticAnalysisResponse,
    HealthResponse,
)

bundle: Optional[AcousticAnomalyDetectorBundle] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle
    artifact_path = DEFAULT_AUDIO_CONFIG.artifacts_dir / DEFAULT_AUDIO_CONFIG.model_filename
    if artifact_path.exists():
        try:
            bundle = AcousticAnomalyDetectorBundle.load(str(artifact_path))
        except Exception:
            bundle, _ = train_acoustic_detector(n_normal_samples=150, config=DEFAULT_AUDIO_CONFIG)
    else:
        bundle, _ = train_acoustic_detector(n_normal_samples=150, config=DEFAULT_AUDIO_CONFIG)
    yield


app = FastAPI(
    title="Industrial Acoustic Machine Anomaly Detection API",
    description=(
        "Production audio signal processing service for machine health monitoring. "
        "Extracts Log-Mel spectrograms and identifies anomalous acoustic signatures "
        "(bearing knock, cavitation, friction squeal) via reconstruction error."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health():
    return HealthResponse(
        status="healthy",
        detector_loaded=(bundle is not None),
        sampling_rate_hz=DEFAULT_AUDIO_CONFIG.sampling_rate_hz,
        duration_sec=DEFAULT_AUDIO_CONFIG.duration_sec,
        calibrated_threshold=bundle.threshold if bundle else DEFAULT_AUDIO_CONFIG.anomaly_threshold,
    )


@app.post("/analyze/synthetic", response_model=AcousticAnalysisResponse, tags=["Acoustic Analysis"])
def analyze_synthetic_sound(req: AcousticSimulationRequest):
    """Generates synthetic machine audio under chosen condition and evaluates anomaly status."""
    if bundle is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Acoustic detector unavailable")

    audio, _ = generate_machine_audio(
        machine_type=req.machine_type,
        condition=req.condition,
        seed=req.seed,
        config=DEFAULT_AUDIO_CONFIG,
    )
    res = bundle.predict_audio(audio)
    return AcousticAnalysisResponse(**res.__dict__)


@app.post("/analyze/audio", response_model=AcousticAnalysisResponse, tags=["Acoustic Analysis"])
def analyze_raw_audio(req: RawAudioAnalysisRequest):
    """Evaluates raw acoustic audio waveform array directly for anomaly reconstruction error."""
    if bundle is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Acoustic detector unavailable")

    import numpy as np
    audio = np.array(req.audio_samples, dtype=np.float32)
    res = bundle.predict_audio(audio)
    return AcousticAnalysisResponse(**res.__dict__)
