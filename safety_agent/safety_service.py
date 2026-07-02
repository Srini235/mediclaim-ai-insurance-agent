from fastapi import FastAPI
from pydantic import BaseModel

app_safety_service = FastAPI(title="Safety Service", version="1.0.0")

class SafetyInput(BaseModel):
    drowsiness_score: float
    sensor_fault_detected: bool

@app.post("/evaluate")
def evaluate_safety(data: SafetyInput):
    if data.sensor_fault_detected and data.drowsiness_score > 0.7:
            risk = "CRITICAL_HAZARD"
        elif data.sensor_fault_detected or data.drowsiness_score > 0.6:
            risk = "ELEVATED_RISK"
        else:
            risk = "NORMAL"
        return {"risk_level": risk}