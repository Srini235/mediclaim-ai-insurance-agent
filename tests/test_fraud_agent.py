import pytest
from fastapi.testclient import TestClient
# Import the actual 'app' instance from your microservice file
from src.agents.fraud_predictor import app

client = TestClient(app)

def test_fraud_agent_happy_path():
    """Verify the microservice correctly evaluates low-risk claims."""
    payload = {
        "customer_age": 34,
        "policy_deductible": 1000,
        "claim_amount": 45000.0,
        "past_claims_count": 0,
        "incident_hour": 14
    }
    # Send a mock network request directly to the local FastAPI router
    response = client.post("/api/v1/evaluate-risk", json=payload)
    
    # Assert network codes and payload contracts match expectations
    assert response.status_code == 200
    json_data = response.json()
    assert "fraud_probability" in json_data
    assert "suspect_anomaly" in json_data
    assert isinstance(json_data["suspect_anomaly"], bool)