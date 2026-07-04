"""
api_server.py — FastAPI backend for the Predictive-Maintenance web app.

Author: Aman Kushwah (2024AC05064) — Group 105

Serves the multi-output condition model + real-time stability model to the React
frontend, guarded by the application-wide security layer, and enriched with RAG
repair guidance (with optional OpenRouter LLM generation).

Run:  uvicorn api_server:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import os
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.security_layer import (
    validate_sensor_payload, HYDRAULIC_BOUNDS, ApiKeyAuthenticator, RateLimiter,
    AuditTrail, compute_file_sha256, SecurityError,
)
from src.maintenance_advisor import MaintenanceAdvisor

REGISTRY = "model_registry"
COND_PATH = os.path.join(REGISTRY, "condition_model.joblib")
STAB_PATH = os.path.join(REGISTRY, "stability_model.joblib")
SCHEMA_PATH = os.path.join(REGISTRY, "schema.json")
API_KEY = os.getenv("HYDRAULICS_API_KEY", "fleet-ops-secret-key")

if not (os.path.exists(COND_PATH) and os.path.exists(SCHEMA_PATH)):
    raise RuntimeError("Models not found. Run:  python3 train_and_save.py")

_condition_model = joblib.load(COND_PATH)
_stability_model = joblib.load(STAB_PATH)
with open(SCHEMA_PATH) as f:
    SCHEMA = json.load(f)
NUMERIC = SCHEMA["numeric_features"]
FEATURES = NUMERIC + SCHEMA["categorical_features"]
TARGETS = SCHEMA["targets"]
HEALTHY = SCHEMA["healthy_class"]
_model_sha = compute_file_sha256(COND_PATH)

_auth = ApiKeyAuthenticator([API_KEY])
_rate = RateLimiter(max_calls=30, window_seconds=1.0)
_audit = AuditTrail()
_advisor = MaintenanceAdvisor()

# human-readable severity ranking so we can pick the worst component to advise on
SEVERITY = {
    "cooler_condition":     {100: 0, 20: 2, 3: 3},
    "valve_condition":      {100: 0, 90: 1, 80: 2, 73: 3},
    "pump_leakage":         {0: 0, 1: 2, 2: 3},
    "accumulator_pressure": {130: 0, 115: 1, 100: 2, 90: 3},
}

app = FastAPI(title="Predictive Maintenance — Mobile Hydraulics API",
              description="Group 105 · condition monitoring + stability + RAG repair advisor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Warm up both models at startup so the first real request hits steady-state latency
_warm = pd.DataFrame([[0]*len(NUMERIC) + [SCHEMA["machine_types"][0]]], columns=FEATURES)
try:
    _condition_model.predict(_warm); _stability_model.predict(_warm)
except Exception:
    pass


class SensorReading(BaseModel):
    operating_hours: float = Field(..., example=900)
    pressure_mean_bar: float = Field(..., example=165)
    pressure_std_bar: float = Field(..., example=6.5)
    flow_mean_lpm: float = Field(..., example=7.2)
    oil_temp_mean_c: float = Field(..., example=68)
    vibration_rms_mms: float = Field(..., example=3.5)
    motor_power_kw: float = Field(..., example=18)
    pump_speed_mean_rpm: float = Field(..., example=1400)
    cooling_efficiency_pct: float = Field(..., example=78)
    machine_type: str = Field(..., example="Excavator")


class ComponentResult(BaseModel):
    component: str
    predicted_class: int
    status: str          # "healthy" or "attention"


class PredictionResponse(BaseModel):
    components: list[ComponentResult]
    stability: str                     # "stable" / "unstable"
    flagged_component: str | None
    repair_procedure: str | None
    repair_guidance: str | None
    llm_recommendation: str | None
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "healthy", "model_sha256": _model_sha[:12],
            "targets": TARGETS, "machine_types": SCHEMA["machine_types"],
            "llm_enabled": bool(os.getenv("OPENROUTER_API_KEY"))}


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading, x_api_key: str = Header(default=API_KEY)):
    import time
    t0 = time.perf_counter()
    payload = reading.model_dump()
    machine_type = payload.pop("machine_type")
    try:
        _auth.authenticate(x_api_key)
        _rate.check("frontend")
        clean = validate_sensor_payload(payload, bounds=HYDRAULIC_BOUNDS)   # validate 9 sensors
    except SecurityError as e:
        raise HTTPException(status_code=400, detail=str(e))

    clean["machine_type"] = machine_type
    X = pd.DataFrame([[clean[c] for c in FEATURES]], columns=FEATURES)

    cond_pred = _condition_model.predict(X)[0]
    stability = "unstable" if int(_stability_model.predict(X)[0]) == 1 else "stable"
    # model inference latency (excludes the optional LLM enrichment) — this is the
    # real-time number that must stay under the 100ms Quality Requirement
    latency = round((time.perf_counter() - t0) * 1000, 2)

    components, worst = [], None
    worst_sev = 0
    for i, target in enumerate(TARGETS):
        cls = int(cond_pred[i])
        healthy = cls == HEALTHY[target]
        components.append(ComponentResult(component=target, predicted_class=cls,
                                          status="healthy" if healthy else "attention"))
        sev = SEVERITY.get(target, {}).get(cls, 0)
        if not healthy and sev > worst_sev:
            worst_sev, worst = sev, target

    procedure = guidance = llm = None
    if worst:
        advice = _advisor.advise(worst, k=1)[0]
        procedure, guidance = advice["procedure"], advice["guidance"]
        llm = advice.get("llm_recommendation")

    _audit.record("frontend", "predict", f"stability={stability} worst={worst}")

    return PredictionResponse(
        components=components, stability=stability, flagged_component=worst,
        repair_procedure=procedure, repair_guidance=guidance,
        llm_recommendation=llm, latency_ms=latency)
