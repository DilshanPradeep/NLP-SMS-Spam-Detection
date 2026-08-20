import time
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
import fastapi_app

@pytest.fixture(scope="module")
def client():
    with TestClient(fastapi_app.app) as c:
        yield c

def test_1_get_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "NLP SMS Spam Detection API"
    assert data["version"] == "1.0.0"
    assert data["model"] == "XGBoost V2"
    assert data["status"] == "running"

def test_2_get_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["vectorizer_loaded"] is True
    assert data["model_name"] == "XGBoost V2"
    assert data["version"] == "1.0.0"

def test_3_get_docs(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower() or "html" in response.text.lower()

def test_4_get_openapi_json(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "paths" in schema
    assert "/api/v1/predict" in schema["paths"]
    assert "/api/v1/batch" in schema["paths"]

def test_5_predict_ham_message(client):
    payload = {"text": "Hey, are you free to meet for coffee this afternoon?"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    assert data["label"] == "HAM"
    assert data["ham_probability"] > 0.5
    assert data["model"] == "XGBoost V2"

def test_6_predict_spam_message(client):
    payload = {"text": "URGENT! You have won a £1000 cash prize! Call 09061701461 to claim your reward now!"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["label"] == "SPAM"
    assert data["spam_probability"] > 0.5
    assert data["confidence"] >= 90.0

def test_7_empty_sms_validation(client):
    payload = {"text": ""}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422

def test_8_whitespace_only_sms_validation(client):
    payload = {"text": "      "}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422

def test_9_maximum_sms_length_validation(client):
    payload = {"text": "a" * 5001}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422

def test_10_url_preprocessing(client):
    payload = {"text": "Verify your identity at http://www.secure-login-bank.co.uk/auth"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "tok_url" in data["cleaned_text"]

def test_11_currency_preprocessing(client):
    payload = {"text": "You were awarded £500 and $1000 cash into your account"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "tok_money" in data["cleaned_text"]

def test_12_phone_preprocessing(client):
    payload = {"text": "Call our hotline on 08000930705 or +447786200117 to claim"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "tok_phone" in data["cleaned_text"]

def test_13_batch_prediction(client):
    messages = [
        "WINNER! Claim £1000 prize call 0800123456",
        "Hi Mom, I will be home by 6pm.",
        "Your account is suspended. Click http://scam-link.cc to fix."
    ]
    response = client.post("/api/v1/batch", json={"messages": messages})
    assert response.status_code == 200
    data = response.json()
    assert data["total_messages"] == 3
    assert data["spam_count"] == 2
    assert data["ham_count"] == 1
    assert len(data["results"]) == 3
    assert data["results"][0]["label"] == "SPAM"
    assert data["results"][1]["label"] == "HAM"
    assert data["results"][2]["label"] == "SPAM"

def test_14_empty_batch_validation(client):
    payload = {"messages": []}
    response = client.post("/api/v1/batch", json=payload)
    assert response.status_code == 422

def test_15_batch_size_limit_validation(client):
    payload = {"messages": ["Test message"] * 101}
    response = client.post("/api/v1/batch", json=payload)
    assert response.status_code == 422

def test_16_model_health_and_availability(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["vectorizer_loaded"] is True

def test_17_response_schema_validation(client):
    payload = {"text": "Testing response schema types and fields."}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert isinstance(data["original_text"], str)
    assert isinstance(data["cleaned_text"], str)
    assert isinstance(data["prediction"], int)
    assert data["prediction"] in (0, 1)
    assert isinstance(data["label"], str)
    assert data["label"] in ("SPAM", "HAM")
    assert isinstance(data["spam_probability"], (float, int))
    assert isinstance(data["ham_probability"], (float, int))
    assert isinstance(data["confidence"], (float, int))
    assert isinstance(data["model"], str)
    assert isinstance(data["processing_time_ms"], (float, int))

def test_18_prediction_performance_smoke(client):
    test_samples = [
        "WINNER! Claim £1000 cash prize now!",
        "Hey, are you free tonight for dinner?",
        "Your parcel is waiting at http://track.delivery.com",
        "Can you send the slides before 5pm?"
    ] * 5  # 20 samples

    latencies = []
    for text in test_samples:
        t0 = time.perf_counter()
        response = client.post("/api/v1/predict", json={"text": text})
        dt = (time.perf_counter() - t0) * 1000.0
        assert response.status_code == 200
        latencies.append(dt)

    assert len(latencies) == 20
    avg_latency = sum(latencies) / len(latencies)
    assert avg_latency < 50.0

def test_19_config_defaults():
    cfg = fastapi_app.AppConfig()
    assert cfg.HOST in ("127.0.0.1", "localhost")
    assert cfg.PORT == 8000
    assert cfg.LOG_LEVEL in ("INFO", "DEBUG", "WARNING", "ERROR")
    assert "member3_v2" in cfg.MODEL_DIR
    assert cfg.get_model_path().endswith("xgb_model.pkl")
    assert cfg.get_vectorizer_path().endswith("tfidf.pkl")

def test_20_health_degraded_when_model_unloaded(client):
    with patch.dict(fastapi_app.ml_models, {"model": None, "vectorizer": None}, clear=True):
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False
        assert data["vectorizer_loaded"] is False

def test_21_predict_503_when_model_unloaded(client):
    with patch.dict(fastapi_app.ml_models, {"model": None, "vectorizer": None}, clear=True):
        response = client.post("/api/v1/predict", json={"text": "Test message"})
        assert response.status_code == 503
        data = response.json()
        assert "unavailable" in data["detail"].lower()
        # Verify absolute paths are not exposed
        assert "models" not in data["detail"].lower()
        assert ".pkl" not in data["detail"].lower()

def test_22_batch_503_when_model_unloaded(client):
    with patch.dict(fastapi_app.ml_models, {"model": None, "vectorizer": None}, clear=True):
        response = client.post("/api/v1/batch", json={"messages": ["Test message 1", "Test message 2"]})
        assert response.status_code == 503
        data = response.json()
        assert "unavailable" in data["detail"].lower()

def test_23_internal_error_safe_handling(client):
    with patch.object(fastapi_app.ml_models["model"], "predict", side_effect=RuntimeError("Simulated engine fault")):
        response = client.post("/api/v1/predict", json={"text": "Hello world"})
        assert response.status_code == 500
        data = response.json()
        assert "internal" in data["detail"].lower()
        # Verify no traceback or path leakage
        assert "Simulated engine fault" not in data["detail"]
        assert "Traceback" not in data["detail"]
