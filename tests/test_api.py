"""Integration tests for FastAPI acoustic anomaly detection endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["detector_loaded"] is True
    assert data["sampling_rate_hz"] > 0
    assert data["calibrated_threshold"] > 0


def test_analyze_synthetic_normal(client):
    payload = {
        "machine_type": "pump",
        "condition": "normal",
        "seed": 42
    }
    response = client.post("/analyze/synthetic", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_anomaly"] is False
    assert data["health_status"] == "NORMAL"
    assert "Normal" in data["diagnosed_sound_profile"]


def test_analyze_synthetic_anomaly(client):
    payload = {
        "machine_type": "motor",
        "condition": "bearing_knock",
        "seed": 100
    }
    response = client.post("/analyze/synthetic", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_anomaly"] is True
    assert data["health_status"] in ["WARNING", "CRITICAL"]
    assert data["anomaly_score"] >= data["threshold"]
