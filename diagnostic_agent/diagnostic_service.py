from fastapi import FastAPI
import uvicorn
import requests
from pydantic import BaseModel

app_diag_service = FastAPI(title="Diagnostic Service", version="1.0.0")

class DataInput(BaseModel):
    drowsiness_score: float
    sensor_fault_detected: bool

@app.post("/process")
def process_diagnostics(data: DataInput):
    # Pipe and filter logic: clean up boundaries, compute direct state
    fault = data.sensor_fault_detected or data.drowsiness_score > 0.85
    
    return {
        "drowsiness_score": data.drowsiness_score,
        "sensor_fault_detected": fault
    }