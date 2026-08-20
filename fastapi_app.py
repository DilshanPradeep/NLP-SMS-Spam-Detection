import os
import time
import pickle
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, Response, status
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

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing %s (v%s)...", config.API_TITLE, config.API_VERSION)
    vectorizer_path = config.get_vectorizer_path()
    model_path = config.get_model_path()

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

@app.get("/", response_model=RootResponse)
def read_root():
    return RootResponse(
        service=config.API_TITLE,
        version=config.API_VERSION,
        model=config.MODEL_NAME,
        status="running"
    )

@app.get("/health", response_model=HealthResponse)
def health_check(response: Response):
    is_model_loaded = ml_models.get("model") is not None
    is_vec_loaded = ml_models.get("vectorizer") is not None
    is_healthy = is_model_loaded and is_vec_loaded

    if not is_healthy:
        logger.warning("Health check reporting degraded state: models unavailable.")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="degraded",
            model_loaded=is_model_loaded,
            vectorizer_loaded=is_vec_loaded,
            model_name=config.MODEL_NAME,
            version=config.API_VERSION
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        vectorizer_loaded=True,
        model_name=config.MODEL_NAME,
        version=config.API_VERSION
    )

@app.post("/api/v1/predict", response_model=SMSPredictResponse)
def predict_sms(payload: SMSPredictRequest):
    if ml_models.get("model") is None or ml_models.get("vectorizer") is None:
        logger.error("Prediction attempted while models are unavailable in memory.")
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
        logger.error("Unexpected prediction failure: %s", type(e).__name__)
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
                logger.error("Unexpected failure during batch item %d: %s", idx, type(e).__name__)
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

    return SMSBatchResponse(
        total_messages=len(payload.messages),
        spam_count=spam_counter,
        ham_count=ham_counter,
        total_processing_time_ms=round(total_elapsed_ms, 3),
        results=results
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host=config.HOST, port=config.PORT, reload=True)
