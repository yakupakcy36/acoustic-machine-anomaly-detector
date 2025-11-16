"""Pydantic v2 schemas for the Acoustic Machine Anomaly API."""

from pydantic import BaseModel, Field


class AcousticSimulationRequest(BaseModel):
    machine_type: str = Field(default="pump", description="pump, motor, fan")
    condition: str = Field(default="bearing_knock", description="normal, bearing_knock, cavitation, friction_screech")
    seed: int = Field(default=42, ge=0)


class AcousticAnalysisResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    health_status: str = Field(..., description="NORMAL, WARNING, CRITICAL")
    diagnosed_sound_profile: str
    spectral_centroid_hz: float
    rms_energy: float


class HealthResponse(BaseModel):
    status: str
    detector_loaded: bool
    sampling_rate_hz: int
    duration_sec: float
    calibrated_threshold: float
