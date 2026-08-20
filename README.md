# 🛡️ SpamGuard AI — SMS Spam Detection System

<div align="center">

**AI-Powered SMS Spam Detection & Model Comparison Platform**

Detect whether an SMS message is **Spam** or **Ham** using multiple Machine Learning and Deep Learning models.

### 🤖 6 AI Models   |   📊 Model Evaluation   |   🌐 Web Dashboard   |   🔌 REST API

</div>

---

## 📌 Overview

**SpamGuard AI** is an NLP-based SMS spam detection system developed to automatically classify text messages as either:

* 🟢 **Ham** — Legitimate and safe SMS messages
* 🔴 **Spam** — Unwanted, suspicious, promotional, or potentially malicious SMS messages

Instead of relying on a single machine learning algorithm, this project implements and compares **six different ML and Deep Learning models**.

The system combines traditional NLP techniques such as **TF-IDF** with modern neural network architectures including **CNN, LSTM, and Transformer-based models**.

A web-based dashboard allows users to enter an SMS message, select a model, and view the prediction and confidence score.

---

## ✨ Key Features

* SMS Spam / Ham classification
* Six different ML & DL models
* NLP-based text preprocessing
* TF-IDF feature extraction
* Tokenization and sequence padding
* Accuracy, Precision, Recall and F1-Score evaluation
* Confusion matrix generation
* Model comparison
* Interactive web dashboard
* Flask REST API
* Prediction confidence scores
* Compare predictions from multiple models
* API health monitoring
* Responsive web interface

---

## 🧠 Models Implemented

The project contains **3 traditional Machine Learning models** and **3 Deep Learning models**.

| Member   | Machine Learning    | Deep Learning      |
| -------- | ------------------- | ------------------ |
| Member 1 | Logistic Regression | 1D CNN             |
| Member 2 | Random Forest       | LSTM               |
| Member 3 | XGBoost             | Custom Transformer |

### 1️⃣ Logistic Regression

Uses **TF-IDF** features to represent SMS text and classify messages into Spam or Ham.

**Configuration:**

* TF-IDF features: 3,000
* Logistic Regression classifier
* Maximum iterations: 200

---

### 2️⃣ 1D CNN

A Convolutional Neural Network designed for text classification.

**Architecture:**

`Embedding → Conv1D → Global Max Pooling → Dense`

The CNN learns important local word and phrase patterns that may indicate spam.

---

### 3️⃣ Random Forest

An ensemble machine learning model consisting of multiple decision trees.

**Configuration:**

* TF-IDF features: 3,000
* 50 decision trees
* Random state: 42

---

### 4️⃣ LSTM

A Long Short-Term Memory neural network designed to capture sequential dependencies within SMS text.

**Architecture:**

`Embedding → LSTM → Dense`

---

### 5️⃣ XGBoost

An optimized gradient boosting classifier used to identify complex relationships within TF-IDF text features.
---

### 6️⃣ Custom Transformer

A lightweight Transformer architecture developed specifically for SMS classification.

**Architecture:**

`Embedding + Positional Encoding → Multi-Head Attention → Layer Normalization → Feed Forward Network → Global Average Pooling → Dense`

The model uses **2 attention heads** to learn relationships between words in the message.

---

## 📊 Dataset

The project uses the **SMS Spam Collection Dataset**.

### Dataset Information

| Property       | Details                    |
| -------------- | -------------------------- |
| Total Messages | 5,574                      |
| Language       | English                    |
| Classes        | Ham / Spam                 |
| Ham            | ~86.6%                     |
| Spam           | ~13.4%                     |
| Task           | Binary Text Classification |

The dataset contains real SMS messages and is suitable for evaluating spam detection systems.

### Dataset Split

The data is divided into:

* 🟦 Training Set — ~70%
* 🟨 Validation Set — ~15%
* 🟥 Test Set — 15%

---

## 🔄 NLP Preprocessing Pipeline

Before training the models, SMS messages go through a preprocessing pipeline:

```text
Raw SMS
   ↓
Convert to Lowercase
   ↓
Remove Special Characters
   ↓
Clean Text
   ↓
Train / Validation / Test Split
   ↓
Feature Extraction
   ↓
Machine Learning / Deep Learning Model
   ↓
Spam / Ham Prediction
```

### ML Pipeline

```text
SMS
 ↓
Text Cleaning
 ↓
TF-IDF Vectorization
 ↓
ML Classifier
 ↓
Spam / Ham
```

### DL Pipeline

```text
SMS
 ↓
Text Cleaning
 ↓
Tokenization
 ↓
Sequence Padding
 ↓
Embedding
 ↓
Neural Network
 ↓
Spam / Ham
```

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │      User SMS       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Text Preprocess   │
                    │ Lowercase + Clean   │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
       ┌─────────────────┐          ┌─────────────────┐
       │   ML Pipeline   │          │   DL Pipeline   │
       │     TF-IDF      │          │ Tokenization    │
       └────────┬────────┘          └────────┬────────┘
                │                            │
        ┌───────┼────────┐          ┌────────┼────────┐
        ▼       ▼        ▼          ▼        ▼        ▼
       LR       RF       XGB        CNN      LSTM   Transformer
        │       │        │           │        │        │
        └───────┴────────┴───────────┴────────┴────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Spam / Ham Result   │
                    │ + Confidence Score  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Web Dashboard     │
                    │    / REST API       │
                    └─────────────────────┘
```

---

# 🌐 Web Application

The project includes an interactive web interface called:

## **SpamGuard AI**

The dashboard provides:

### 🔎 Spam Detector

Users can enter an SMS message and select one of the six trained models.

The system returns:

* Prediction
* Spam / Ham label
* Confidence score
* Model name
* Model type
* Group member/model information

### 📊 Analytics

The dashboard provides model evaluation information and comparison visualizations.

### ℹ️ About

Provides information about the system, models and project.

---

# 🔌 REST API

The backend is implemented using **Flask**.

## Available Endpoints

### Health Check

```http
GET /health
```

Returns the API status and loaded models.

---

### Single Model Prediction

```http
POST /predict
```

Example request:

```json
{
    "text": "Congratulations! You have won a free prize!",
    "model": "lr"
}
```

Example response:

```json
{
    "model_key": "lr",
    "model_name": "Logistic Regression",
    "model_type": "ML",
    "prediction": 1,
    "label": "spam",
    "confidence": 98.52
}
```

---

### Compare All Models

```http
POST /compare
```

Example request:

```json
{
    "text": "Congratulations! You have won a free prize!"
}
```

This endpoint runs the message through the available models and returns their predictions and confidence scores.

---

### Evaluation Metrics

```http
GET /metrics
```

Returns the stored model evaluation metrics.

---

# 📊 Evaluation Metrics

Because the dataset is imbalanced, accuracy alone is not enough to determine the best model.

The project evaluates models using:

### Accuracy

Measures the percentage of correctly classified messages.

### Precision

Measures how many messages predicted as Spam were actually Spam.

### Recall

Measures how many actual Spam messages were successfully detected.

### F1-Score

Provides a balance between Precision and Recall.

### Confusion Matrix

Used to analyse:

* True Positives
* True Negatives
* False Positives
* False Negatives

The **F1-Score, Precision and Recall** are particularly important for this project because incorrectly classifying legitimate messages as spam can negatively affect users.

---

# 📁 Project Structure

```text
NLP-SMS-Spam-Detection/
│
├── 1_prepare_data.py
├── 2_member1_train.py
├── 3_member2_train.py
├── 4_member3_train.py
├── 5_test_model.py
├── 6_evaluate_models.py
│
├── api.py
│
├── env.txt
├── run_instructions.txt
├── project_documentation.txt
│
├── generate_presentation.py
├── generate_report.py
│
├── ui/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── models/
│   ├── member1/
│   │   ├── lr_model.pkl
│   │   ├── tfidf.pkl
│   │   ├── cnn_model.keras
│   │   └── tokenizer.pkl
│   │
│   ├── member2/
│   │   ├── rf_model.pkl
│   │   ├── tfidf.pkl
│   │   ├── lstm_model.keras
│   │   └── tokenizer.pkl
│   │
│   └── member3/
│       ├── xgb_model.pkl
│       ├── tfidf.pkl
│       ├── transformer_model.keras
│       └── tokenizer.pkl
│
└── evaluation_metrics.json
```

> **Note:** Generated datasets, trained model files and evaluation outputs may be excluded from GitHub depending on repository size and `.gitignore` configuration.

---

# ⚙️ Technologies Used

### Programming

* 🐍 Python 3.10

### Machine Learning

* Scikit-learn
* XGBoost

### Deep Learning

* TensorFlow / Keras

### Natural Language Processing

* TF-IDF
* Tokenization
* Sequence Padding
* Text preprocessing

### Backend

* Flask
* REST API

### Frontend

* HTML5
* CSS3
* JavaScript
* Chart.js

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn
* Chart.js

---

# 🚀 Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

```bash
cd NLP-SMS-Spam-Detection
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install pandas scikit-learn tensorflow xgboost flask numpy matplotlib seaborn
```

---

# 📥 Prepare the Dataset

Run:

```bash
python 1_prepare_data.py
```

This will:

1. Download the SMS dataset
2. Clean the SMS text
3. Convert labels to numerical values
4. Split the dataset
5. Create the `data/` directory

---

# 🧠 Train the Models

### Member 1

```bash
python 2_member1_train.py
```

Trains:

* Logistic Regression
* 1D CNN

### Member 2

```bash
python 3_member2_train.py
```

Trains:

* Random Forest
* LSTM

### Member 3

```bash
python 4_member3_train.py
```

Trains:

* XGBoost
* Transformer

---

# 📈 Evaluate All Models

After training:

```bash
python 6_evaluate_models.py
```

This evaluates the available models using:

```text
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
```

---

# 🧪 Test a Model from Terminal

Run:

```bash
python 5_test_model.py
```

You can select:

```text
1 → Logistic Regression
2 → 1D CNN
3 → Random Forest
4 → LSTM
5 → XGBoost
6 → Transformer
```

Then enter an SMS message.

Example:

```text
Congratulations! You have won a free cash prize. Call now!
```

Possible output:

```text
🚨 RESULT: This is a SPAM message!
```

---

# 🌐 Run the Web Application

Make sure the models have been trained first.

Then run:

```bash
python api.py
```

The application will start at:

```text
http://localhost:5000
```

Open the address in your web browser.

---

# 🧪 Example Messages

### Spam Example

```text
Congratulations! You have won £1000. Call now to claim your prize!
```

Expected:

```text
🚨 SPAM
```

### Ham Example

```text
Hey, are we still meeting at 6 pm today?
```

Expected:

```text
✅ HAM
```

> Predictions may vary between models because each model uses a different learning architecture.

---

# 🔬 Why Multiple Models?

Using multiple models allows the project to compare different approaches to NLP classification.

### Traditional ML

**Logistic Regression, Random Forest and XGBoost**

These models use TF-IDF numerical representations of the SMS text.

### Deep Learning

**CNN, LSTM and Transformer**

These models learn representations from tokenized text sequences.

The comparison helps identify which approach performs best for the SMS spam classification problem.

---

# 🎯 Project Objectives

* Develop an automated SMS spam detection system.
* Apply NLP preprocessing techniques to SMS data.
* Implement traditional machine learning classifiers.
* Implement deep learning architectures for text classification.
* Compare ML and DL approaches.
* Evaluate models using multiple performance metrics.
* Develop a user-friendly web interface.
* Provide a REST API for real-time predictions.

---

# 👥 Team Structure

This project was developed as a **3-member NLP group project**.

| Member   | ML Model            | DL Model    |
| -------- | ------------------- | ----------- |
| Member 1 | Logistic Regression | 1D CNN      |
| Member 2 | Random Forest       | LSTM        |
| Member 3 | XGBoost             | Transformer |

---

# 📚 Learning Outcomes

Through this project, we gained practical experience in:

* Natural Language Processing
* Text classification
* Feature engineering
* TF-IDF
* Machine Learning
* Deep Learning
* CNN for NLP
* LSTM networks
* Transformer architecture
* Model evaluation
* REST API development
* Frontend integration
* AI model deployment concepts

---

# ⚠️ Limitations

* The dataset contains only English SMS messages.
* The dataset is relatively small compared with modern NLP datasets.
* Text preprocessing removes some punctuation and special-character information.
* Models may perform differently on modern spam patterns that were not represented in the training data.
* The system should be treated as a classification aid rather than a guaranteed spam filter.

---

# 🔮 Future Improvements

Possible future enhancements include:

* 🌍 Multilingual SMS spam detection
* 🤖 BERT / RoBERTa-based classification
* 📱 Mobile application integration
* 🔄 Real-time SMS filtering
* 🛡️ Phishing URL detection
* 🔐 Explainable AI predictions
* 📊 Advanced analytics dashboard
* ☁️ Cloud deployment
* 📚 Larger and more diverse datasets
* 🔄 Continuous model retraining

---

# 📄 Dataset Reference

The project uses the **SMS Spam Collection Dataset**, originally associated with the UCI Machine Learning Repository.

Dataset source:

```text
https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
```

A tab-separated version is downloaded automatically by `1_prepare_data.py`.

---

# 👩‍💻 Project Type

**Academic NLP / Artificial Intelligence Group Project**

### Main Areas

`NLP` · `Machine Learning` · `Deep Learning` · `Text Classification` · `Flask API` · `Web Development`

---

<div align="center">

### 🛡️ SpamGuard AI

**Detect Spam. Protect Messages. Compare Intelligence.**

Built with ❤️ using Python, NLP, Machine Learning & Deep Learning.

</div>
