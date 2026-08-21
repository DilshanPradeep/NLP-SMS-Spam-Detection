# NLP SMS Spam Detection System — Member 3 (Chanuka) Module & Production FastAPI Inference Service

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-V2-orange.svg)](https://xgboost.readthedocs.io/)
[![Keras](https://img.shields.io/badge/Keras%2FTensorFlow-2.x-FF6F00.svg)](https://keras.io/)
[![Pytest](https://img.shields.io/badge/Pytest-36%20Passed-brightgreen.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 1. Project Title & Overview

This repository branch (`member3-Chanuka`) houses the **Member 3 (Chanuka) Module** and the **Production FastAPI Inference Microservice** for the NLP SMS Spam Detection System. 

The primary objective of this module is to deliver an enterprise-grade NLP pipeline that transitions experimental ML/DL models into a low-latency, production-ready inference API. Member 3's technical scope encompasses end-to-end responsibility for:
- Advanced gradient boosting (**XGBoost V2**) and deep learning (**Transformer Network V2**) model engineering.
- Robust NLP preprocessing and regex-based token normalization (`utils/preprocessor.py`).
- High-throughput asynchronous REST microservice development (`fastapi_app.py`) featuring real-time health monitoring, in-memory metric tracking, batch processing, and multi-model output comparison.
- Enterprise test engineering, including 36 comprehensive Pytest automated unit, validation, CORS, and performance smoke tests (`tests/test_fastapi.py`).

---

## 2. Key Deliverables & Architectural Contributions (Member 3 Scope)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               MEMBER 3 ARCHITECTURE OVERVIEW                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
   
  ┌───────────────────────┐      ┌─────────────────────────┐      ┌───────────────────────┐
  │  Raw SMS Text Payload │ ---> │  NLP Preprocessing Engine│ ---> │ TF-IDF / Tokenization │
  └───────────────────────┘      │ (Regex Normalization)   │      └───────────────────────┘
                                 └─────────────────────────┘                  │
                                                                              ▼
  ┌───────────────────────┐      ┌─────────────────────────┐      ┌───────────────────────┐
  │ Pytest Test Suite     │ <--- │ Production FastAPI App  │ <--- │  Inference Engine     │
  │ (36 Automated Tests)  │      │ (Asynchronous REST API) │      │ (XGBoost V2 / Transf) │
  └───────────────────────┘      └─────────────────────────┘      └───────────────────────┘
```

### 1. Advanced Model Architecture (`4_member3_train_v2.py`)
- **XGBoost V2 (Enhanced)**: Engineered using TF-IDF vectorization with n-gram range `(1, 2)` (sublinear TF scaling, max features = 5,000). Hyperparameter tuned with `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, and `colsample_bytree=0.8` to resolve baseline recall deficiencies.
- **Transformer Network V2**: Custom Multi-Head Self-Attention architecture incorporating positional encoding, layer normalization, dropout (`0.1`), global average pooling, and dense classification layers optimized with AdamW and learning rate warmup.

### 2. NLP Preprocessing Engine (`utils/preprocessor.py`, `prepare_data_v2.py`)
- Standardized text normalization pipeline featuring lowercasing, regex-based URL token substitution (`tok_url`), phone number tokenization (`tok_phone`), currency symbol masking (`tok_money`), exclamation mark preservation (`tok_exclam`), and whitespace stripping.
- Prevents data leakage between training, validation, and test splits while maximizing model robustness against adversarial text obfuscation.

### 3. Production FastAPI Microservice (`fastapi_app.py`)
- Asynchronous ASGI microservice engineered with `FastAPI` and `Uvicorn`.
- Lifespan context management for single-allocation model artifact loading into memory during startup.
- Thread-safe `MetricsTracker` recording uptime, total requests, success/failure counts, average latency, and batch metrics.
- Complete CORS middleware support configured for frontend origins (`http://127.0.0.1:5500`, `http://localhost:5500`).

### 4. Comprehensive Testing Suite (`tests/test_fastapi.py`)
- **36 Pytest test cases** covering root endpoints, OpenAPI documentation, health monitoring, single prediction, batch processing, 422 input validation, 503 degraded state handling, CORS preflight headers, and sub-50ms latency smoke testing.

### 5. Automated Evaluation & Benchmarking (`evaluate_v1_vs_v2.py`)
- Automated comparison framework evaluating baseline (v1) versus optimized (v2) models on identical unseen test splits, outputting structured metrics to `evaluation_comparison_v1_vs_v2.json`.

---

## 3. Repository Structure

```text
NLP-SMS-Spam-Detection/
├── 4_member3_train.py               # Member 3 Baseline (v1) Training Script
├── 4_member3_train_v2.py            # Member 3 Enhanced (v2) XGBoost & Transformer Pipeline
├── evaluate_v1_vs_v2.py             # Model Evaluation & Comparison Benchmarking Script
├── evaluation_comparison_v1_vs_v2.json # Benchmarking Metric Results (v1 vs v2)
├── evaluation_metrics.json          # Global Model Benchmark Metrics for API & UI
├── fastapi_app.py                   # Production FastAPI REST Microservice
├── prepare_data_v2.py               # Data Preprocessing & Token Normalization Script
├── README.md                        # Production Repository Documentation
├── requirements.txt                 # Project Dependencies & Version Locks
├── data/
│   ├── test_raw.csv                 # Raw SMS Test Split
│   ├── test_v2.csv                  # Preprocessed V2 Test Split
│   ├── train_v2.csv                 # Preprocessed V2 Train Split
│   └── val_v2.csv                   # Preprocessed V2 Validation Split
├── models/
│   ├── member3/                     # Baseline V1 Artifacts (XGBoost & Transformer)
│   └── member3_v2/                  # Production V2 Model Artifacts
│       ├── tfidf.pkl                # Fitted TF-IDF Vectorizer
│       ├── tokenizer.pkl            # Keras Tokenizer Object
│       ├── transformer_model.keras  # Trained Keras Transformer Model
│       └── xgb_model.pkl            # Trained XGBoost V2 Classifier
├── tests/
│   ├── __init__.py                  # Test Package Init
│   └── test_fastapi.py              # 36 Automated Pytest Unit & Integration Tests
├── ui/
│   ├── app.js                       # Frontend SPA Logic & Comparison Engine
│   ├── index.html                   # HTML5 Single Page Application UI
│   └── style.css                    # Modern Glassmorphic Dashboard Styles
└── utils/
    ├── __init__.py
    └── preprocessor.py              # Core NLP Regex Normalization Module
```

---

## 4. Performance & Evaluation Benchmarks (v1 vs v2)

Evaluation performed on **836 unseen test messages** (`data/test_v2.csv`).

| Model Variant | Model Type | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Avg Latency (ms) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost V1 (Baseline)** | ML | 97.37% | 94.74% | 84.11% | 89.11% | ~4.2 ms | Legacy |
| **XGBoost V2 (Enhanced)** | ML | **98.56%** | **98.97%** | **89.72%** | **94.12%** | **~2.8 ms** | **Production API** |
| **Transformer V1 (Baseline)** | DL | 98.68% | 97.06% | 92.52% | 94.74% | ~28.5 ms | Benchmark |
| **Transformer V2 (Corrected)** | DL | 98.33% | 92.66% | 94.39% | 93.52% | ~24.1 ms | Benchmark |
| **1D CNN (Member 1)** | DL | 98.44% | 97.50% | 95.80% | 96.64% | ~18.0 ms | Comparative |
| **LSTM (Member 2)** | DL | 98.21% | 97.20% | 95.10% | 96.14% | ~22.0 ms | Comparative |

> **Key Architectural Insight**: XGBoost V2 achieved a **+5.01% increase in F1-score** and **+5.61% increase in Recall** over V1 by introducing n-gram sublinear TF scaling and custom probability calibration, maintaining an ultra-fast sub-3ms average inference latency.

---

## 5. API Endpoint Specifications

The microservice exposes the following RESTful OpenAPI endpoints:

| HTTP Method | Endpoint Path | Description | Request Payload | Response Schema / Status |
| :---: | :--- | :--- | :--- | :--- |
| `GET` | `/` | Service root & version metadata | None | `RootResponse` (200 OK) |
| `GET` | `/health` | Live service health & metrics tracker | None | `HealthResponse` (200 OK / 503 Service Unavailable) |
| `GET` | `/metrics` | Model performance benchmark metrics | None | `JSON Dict` of 6 model metrics (200 OK) |
| `POST` | `/api/v1/predict` | Single SMS spam classification | `{"text": "string"}` | `SMSPredictResponse` (200 OK / 422 / 503) |
| `POST` | `/api/v1/batch` | Batch SMS spam classification (1-100) | `{"messages": ["str"]}` | `SMSBatchResponse` (200 OK / 422 / 503) |
| `POST` | `/compare` | Multi-model output comparison | `{"text": "string"}` | `CompareResponse` (200 OK / 422 / 503) |

### Request & Response Schemas

#### Single Prediction Request (`POST /api/v1/predict`)
```json
{
  "text": "WINNER! Claim your £1000 cash prize now by calling 0800123456!"
}
```

#### Single Prediction Response (`200 OK`)
```json
{
  "original_text": "WINNER! Claim your £1000 cash prize now by calling 0800123456!",
  "cleaned_text": "winner tok_exclam claim your tok_money cash prize now by calling tok_phone tok_exclam",
  "prediction": 1,
  "label": "SPAM",
  "spam_probability": 0.9842,
  "ham_probability": 0.0158,
  "confidence": 98.42,
  "model": "XGBoost V2",
  "processing_time_ms": 2.451
}
```

---

## 6. Local Environment Setup & Execution Guide

### Prerequisites
- Python 3.11 or higher
- PowerShell (Windows) or Bash (Linux/macOS)

### 1. Environment Setup
Clone the repository, switch to the Member 3 branch, and create a virtual environment:

```powershell
# Switch to Member 3 branch
git checkout member3-Chanuka

# Create and activate virtual environment
python -m venv .venv
& "e:/00_  My/Project system/.venv/Scripts/Activate.ps1"

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Preprocess Data & Train V2 Models (Optional Re-training)
To re-run the NLP data preparation and train Member 3's XGBoost V2 model:

```powershell
# Preprocess raw dataset
python prepare_data_v2.py

# Train XGBoost V2 & Transformer V2
python 4_member3_train_v2.py

# Run V1 vs V2 benchmarking script
python evaluate_v1_vs_v2.py
```

### 3. Launch FastAPI Server
Start the Uvicorn ASGI server hosting the microservice:

```powershell
python -m uvicorn fastapi_app:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 4. Launch Frontend Server
In a separate terminal window, launch the HTTP server to serve the SPA frontend:

```powershell
python -m http.server 5500 --directory ui
```

- **Frontend Application UI**: [http://127.0.0.1:5500](http://127.0.0.1:5500)

### 5. Execute Test Suite
Run the 36 automated unit, schema, and performance tests using Pytest:

```powershell
pytest tests/test_fastapi.py -v
```

---

## 7. Engineering Best Practices Implemented

1. **Graceful Degraded-State Handling**:
   - If model binary files (`xgb_model.pkl` or `tfidf.pkl`) fail to load or are missing from `models/member3_v2/`, the microservice catches the lifecycle exception cleanly.
   - `/health` automatically returns `HTTP 503 Service Unavailable` with `"status": "degraded"`.
   - Prediction endpoints return structured `503 Service Unavailable` JSON responses without exposing internal disk paths or stack trace details.

2. **Strict Input Schema Validation**:
   - Pydantic models validate input length (`1` to `5000` characters for single requests, max `100` items for batch requests).
   - Empty strings or whitespace-only payloads are intercepted with `HTTP 422 Unprocessable Content`.

3. **CORS Security & Preflight Configuration**:
   - Explicitly handles browser preflight `OPTIONS` requests from origins `http://127.0.0.1:5500` and `http://localhost:5500`.

4. **Thread-Safe Metrics & Monitoring**:
   - Implements `threading.Lock()` inside `MetricsTracker` to track request counts and latencies reliably across asynchronous worker threads.

---

## 8. Author & Contribution Details

- **Author**: Chanuka (Member 3)
- **Role**: Senior Machine Learning & Backend Engineer
- **Branch**: `member3-Chanuka`
- **Project**: NLP SMS Spam Detection System
