import os
import json
import time
import pickle
import logging
import threading
from collections import deque
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from utils.preprocessor import preprocess_sms

class AppConfig:
    HOST: str = os.getenv("API_HOST", os.getenv("HOST", "127.0.0.1"))
    PORT: int = int(os.getenv("API_PORT", os.getenv("PORT", "8000")))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    MODEL_DIR: str = os.getenv("MODEL_DIR", os.path.join("models", "member3_v2"))
    MODEL_NAME: str = "XGBoost V2"
    API_VERSION: str = "1.0.0"
    API_TITLE: str = "NLP SMS Spam Detection API"

    @classmethod
    def get_vectorizer_path(cls) -> str:
        return os.path.join(cls.MODEL_DIR, "tfidf.pkl")

    @classmethod
    def get_model_path(cls) -> str:
        return os.path.join(cls.MODEL_DIR, "xgb_model.pkl")

config = AppConfig()

numeric_log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=numeric_log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("sms_spam_api")

class MetricsTracker:
    def __init__(self, max_latency_samples: int = 500):
        self._lock = threading.Lock()
        self.start_time: float = time.time()
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.failed_requests: int = 0
        self.total_batch_requests: int = 0
        self.failed_batch_requests: int = 0
        self._latencies: deque = deque(maxlen=max_latency_samples)
        self.last_request_timestamp: Optional[float] = None

    def record_prediction(self, latency_ms: float = 0.0, success: bool = True) -> None:
        try:
            with self._lock:
                self.total_requests += 1
                self.last_request_timestamp = time.time()
                if success:
                    self.successful_requests += 1
                    self._latencies.append(latency_ms)
                else:
                    self.failed_requests += 1
        except Exception:
            pass

    def record_batch(self, success: bool = True) -> None:
        try:
            with self._lock:
                self.total_batch_requests += 1
                if not success:
                    self.failed_batch_requests += 1
        except Exception:
            pass

    def get_summary(self) -> dict:
        try:
            with self._lock:
                avg_lat = round(sum(self._latencies) / len(self._latencies), 3) if self._latencies else 0.0
                return {
                    "uptime_seconds": round(time.time() - self.start_time, 2),
                    "total_requests": self.total_requests,
                    "successful_requests": self.successful_requests,
                    "failed_requests": self.failed_requests,
                    "total_batch_requests": self.total_batch_requests,
                    "failed_batch_requests": self.failed_batch_requests,
                    "average_latency_ms": avg_lat,
                    "last_request_timestamp": self.last_request_timestamp,
                }
        except Exception:
            return {
                "uptime_seconds": 0.0,
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_batch_requests": 0,
                "failed_batch_requests": 0,
                "average_latency_ms": 0.0,
                "last_request_timestamp": None,
            }

metrics_tracker = MetricsTracker()
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing %s (v%s)...", config.API_TITLE, config.API_VERSION)
    vectorizer_path = config.get_vectorizer_path()
    model_path = config.get_model_path()

    logger.info("Loading model artifacts from configured directory...")
    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        logger.error("Required model artifacts missing in configured directory.")
        ml_models["vectorizer"] = None
        ml_models["model"] = None
    else:
        try:
            with open(vectorizer_path, "rb") as f:
                ml_models["vectorizer"] = pickle.load(f)
            with open(model_path, "rb") as f:
                ml_models["model"] = pickle.load(f)
            logger.info("%s model and vectorizer loaded successfully into memory.", config.MODEL_NAME)
        except Exception as e:
            logger.error("Failed to load V2 model artifacts: %s", type(e).__name__)
            ml_models["vectorizer"] = None
            ml_models["model"] = None

    yield

    logger.info("Shutting down API service and clearing model memory...")
    ml_models.clear()

app = FastAPI(
    title=config.API_TITLE,
    description="Real-Time SMS Spam Detection API powered by XGBoost V2",
    version=config.API_VERSION,
    lifespan=lifespan
)

ALLOWED_ORIGINS = [
    "http://127.0.0.1",
    "http://localhost",
    "http://127.0.0.1:5000",
    "http://localhost:5000",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "null",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
    logger.error("Unhandled exception during request [%s]: %s", request.url.path, type(exc).__name__)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."}
    )

class SMSPredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="SMS message text to classify (1 to 5000 characters)",
        examples=["WINNER! Claim £1000 cash prize now by calling 0800123456!"]
    )

class SMSPredictResponse(BaseModel):
    original_text: str
    cleaned_text: str
    prediction: int
    label: str
    spam_probability: float
    ham_probability: float
    confidence: float
    model: str
    processing_time_ms: float

class SMSBatchRequest(BaseModel):
    messages: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of SMS messages to classify (1 to 100 messages)",
        examples=[[
            "WINNER! Claim £1000 cash prize now!",
            "Hey, are we still meeting for lunch today?"
        ]]
    )

class SMSBatchResponse(BaseModel):
    total_messages: int
    spam_count: int
    ham_count: int
    total_processing_time_ms: float
    results: List[SMSPredictResponse]

class CompareResultItem(BaseModel):
    original_text: str
    cleaned_text: str
    prediction: int
    label: str
    spam_probability: float
    ham_probability: float
    confidence: float
    model: str
    model_name: Optional[str] = None
    model_type: Optional[str] = "ML"
    is_best: Optional[bool] = True
    processing_time_ms: float

class CompareResponse(BaseModel):
    results: List[CompareResultItem]

class RootResponse(BaseModel):
    service: str
    version: str
    model: str
    status: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vectorizer_loaded: bool
    model_name: str
    version: str
    uptime_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency_ms: float

@app.get("/", response_model=RootResponse)
def read_root():
    return RootResponse(
        service=config.API_TITLE,
        version=config.API_VERSION,
        model=config.MODEL_NAME,
        status="running"
    )

@app.get("/health", response_model=HealthResponse)
@app.get("/api/v1/health", response_model=HealthResponse)
def health_check(response: Response):
    is_model_loaded = ml_models.get("model") is not None
    is_vec_loaded = ml_models.get("vectorizer") is not None
    is_healthy = is_model_loaded and is_vec_loaded
    summary = metrics_tracker.get_summary()

    if not is_healthy:
        logger.warning("Health check reporting degraded state: models unavailable.")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="degraded",
            model_loaded=is_model_loaded,
            vectorizer_loaded=is_vec_loaded,
            model_name=config.MODEL_NAME,
            version=config.API_VERSION,
            uptime_seconds=summary["uptime_seconds"],
            total_requests=summary["total_requests"],
            successful_requests=summary["successful_requests"],
            failed_requests=summary["failed_requests"],
            average_latency_ms=summary["average_latency_ms"]
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        vectorizer_loaded=True,
        model_name=config.MODEL_NAME,
        version=config.API_VERSION,
        uptime_seconds=summary["uptime_seconds"],
        total_requests=summary["total_requests"],
        successful_requests=summary["successful_requests"],
        failed_requests=summary["failed_requests"],
        average_latency_ms=summary["average_latency_ms"]
    )

@app.post("/api/v1/predict", response_model=SMSPredictResponse)
def predict_sms(payload: SMSPredictRequest):
    if ml_models.get("model") is None or ml_models.get("vectorizer") is None:
        logger.error("Prediction attempted while models are unavailable in memory.")
        metrics_tracker.record_prediction(latency_ms=0.0, success=False)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is currently unavailable."
        )

    raw_text = payload.text.strip()
    if not raw_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message text cannot be empty or whitespace only."
        )

    start_time = time.perf_counter()

    try:
        cleaned = preprocess_sms(raw_text)
        vec = ml_models["vectorizer"].transform([cleaned])
        pred = int(ml_models["model"].predict(vec)[0])
        probabilities = ml_models["model"].predict_proba(vec)[0]
    except Exception as e:
        logger.error("Unexpected prediction failure: exception=%s", type(e).__name__)
        metrics_tracker.record_prediction(latency_ms=0.0, success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an internal processing error."
        )

    ham_prob = float(probabilities[0])
    spam_prob = float(probabilities[1])
    confidence = float(probabilities[pred]) * 100.0
    label = "SPAM" if pred == 1 else "HAM"

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info("Prediction completed in %.2f ms (prediction=%d)", elapsed_ms, pred)
    metrics_tracker.record_prediction(latency_ms=elapsed_ms, success=True)

    return SMSPredictResponse(
        original_text=raw_text,
        cleaned_text=cleaned,
        prediction=pred,
        label=label,
        spam_probability=round(spam_prob, 4),
        ham_probability=round(ham_prob, 4),
        confidence=round(confidence, 2),
        model=config.MODEL_NAME,
        processing_time_ms=round(elapsed_ms, 3)
    )

@app.post("/api/v1/batch", response_model=SMSBatchResponse)
def predict_batch(payload: SMSBatchRequest):
    if ml_models.get("model") is None or ml_models.get("vectorizer") is None:
        logger.error("Batch prediction attempted while models are unavailable in memory.")
        metrics_tracker.record_batch(success=False)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is currently unavailable."
        )

    start_time = time.perf_counter()

    results: List[SMSPredictResponse] = []
    spam_counter = 0
    ham_counter = 0

    for idx, msg in enumerate(payload.messages):
        raw_text = str(msg).strip()
        if len(raw_text) > 5000:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Message at index {idx} exceeds maximum length of 5000 characters."
            )

        if not raw_text:
            cleaned = ""
            pred = 0
            label = "HAM"
            ham_prob = 1.0
            spam_prob = 0.0
            confidence = 100.0
            elapsed_single_ms = 0.0
        else:
            t0 = time.perf_counter()
            try:
                cleaned = preprocess_sms(raw_text)
                vec = ml_models["vectorizer"].transform([cleaned])
                pred = int(ml_models["model"].predict(vec)[0])
                probabilities = ml_models["model"].predict_proba(vec)[0]
            except Exception as e:
                logger.error("Unexpected failure during batch item %d: exception=%s", idx, type(e).__name__)
                metrics_tracker.record_batch(success=False)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Batch prediction failed due to an internal processing error."
                )

            ham_prob = float(probabilities[0])
            spam_prob = float(probabilities[1])
            confidence = float(probabilities[pred]) * 100.0
            label = "SPAM" if pred == 1 else "HAM"
            elapsed_single_ms = (time.perf_counter() - t0) * 1000.0

        if pred == 1:
            spam_counter += 1
        else:
            ham_counter += 1

        results.append(
            SMSPredictResponse(
                original_text=raw_text,
                cleaned_text=cleaned,
                prediction=pred,
                label=label,
                spam_probability=round(spam_prob, 4),
                ham_probability=round(ham_prob, 4),
                confidence=round(confidence, 2),
                model=config.MODEL_NAME,
                processing_time_ms=round(elapsed_single_ms, 3)
            )
        )

    total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info("Batch prediction of %d messages completed in %.2f ms", len(payload.messages), total_elapsed_ms)
    metrics_tracker.record_batch(success=True)

    return SMSBatchResponse(
        total_messages=len(payload.messages),
        spam_count=spam_counter,
        ham_count=ham_counter,
        total_processing_time_ms=round(total_elapsed_ms, 3),
        results=results
    )

@app.post("/compare", response_model=CompareResponse)
@app.post("/api/v1/compare", response_model=CompareResponse)
def compare_sms(payload: SMSPredictRequest):
    result = predict_sms(payload)
    return CompareResponse(
        results=[
            CompareResultItem(
                original_text=result.original_text,
                cleaned_text=result.cleaned_text,
                prediction=result.prediction,
                label=result.label,
                spam_probability=result.spam_probability,
                ham_probability=result.ham_probability,
                confidence=result.confidence,
                model=result.model,
                model_name=result.model,
                model_type="ML",
                is_best=True,
                processing_time_ms=result.processing_time_ms
            )
        ]
    )

DEFAULT_EVALUATION_METRICS = {
    "lr": {
        "name": "Logistic Regression",
        "short": "LR",
        "type": "ML",
        "member": 1,
        "accuracy": 97.84,
        "precision": 96.20,
        "recall": 93.40,
        "f1": 94.78,
        "color": "#3b82f6"
    },
    "rf": {
        "name": "Random Forest",
        "short": "RF",
        "type": "ML",
        "member": 2,
        "accuracy": 97.42,
        "precision": 95.80,
        "recall": 92.10,
        "f1": 93.91,
        "color": "#06b6d4"
    },
    "xgb": {
        "name": "XGBoost",
        "short": "XGB",
        "type": "ML",
        "member": 3,
        "accuracy": 98.56,
        "precision": 98.97,
        "recall": 89.72,
        "f1": 94.12,
        "color": "#f59e0b"
    },
    "cnn": {
        "name": "1D CNN",
        "short": "CNN",
        "type": "DL",
        "member": 1,
        "accuracy": 98.44,
        "precision": 97.50,
        "recall": 95.80,
        "f1": 96.64,
        "color": "#8b5cf6"
    },
    "lstm": {
        "name": "LSTM",
        "short": "LSTM",
        "type": "DL",
        "member": 2,
        "accuracy": 98.21,
        "precision": 97.20,
        "recall": 95.10,
        "f1": 96.14,
        "color": "#ec4899"
    },
    "transformer": {
        "name": "Transformer",
        "short": "TF",
        "type": "DL",
        "member": 3,
        "accuracy": 98.74,
        "precision": 98.10,
        "recall": 96.40,
        "f1": 97.24,
        "color": "#10b981"
    }
}

@app.get("/metrics")
@app.get("/api/v1/metrics")
def get_metrics():
    metrics_path = "evaluation_metrics.json"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed reading evaluation_metrics.json, using defaults: %s", e)
    return DEFAULT_EVALUATION_METRICS

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host=config.HOST, port=config.PORT, reload=True)
