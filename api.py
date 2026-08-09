"""
SpamGuard AI — Flask API Backend
=================================
Endpoints:
  GET  /           → serves the UI (ui/index.html)
  GET  /health     → server & model status
  GET  /metrics    → evaluation_metrics.json
  POST /predict    → single-model prediction
  POST /compare    → all-model comparison
"""

import os
import re
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── App Setup ──────────────────────────────────────────────────
app = Flask(__name__, static_folder='ui', static_url_path='')
CORS(app)

# ── Model Registry ─────────────────────────────────────────────
models = {}

MODEL_CONFIG = [
    {
        'key':      'lr',
        'name':     'Logistic Regression',
        'type':     'ml',
        'member':   1,
        'model_path':  'models/member1/lr_model.pkl',
        'vector_path': 'models/member1/tfidf.pkl',
    },
    {
        'key':      'cnn',
        'name':     '1D CNN',
        'type':     'dl',
        'member':   1,
        'model_path':  'models/member1/cnn_model.keras',
        'vector_path': 'models/member1/tokenizer.pkl',
    },
    {
        'key':      'rf',
        'name':     'Random Forest',
        'type':     'ml',
        'member':   2,
        'model_path':  'models/member2/rf_model.pkl',
        'vector_path': 'models/member2/tfidf.pkl',
    },
    {
        'key':      'lstm',
        'name':     'LSTM',
        'type':     'dl',
        'member':   2,
        'model_path':  'models/member2/lstm_model.keras',
        'vector_path': 'models/member2/tokenizer.pkl',
    },
    {
        'key':      'xgb',
        'name':     'XGBoost',
        'type':     'ml',
        'member':   3,
        'model_path':  'models/member3/xgb_model.pkl',
        'vector_path': 'models/member3/tfidf.pkl',
    },
    {
        'key':      'transformer',
        'name':     'Transformer',
        'type':     'dl',
        'member':   3,
        'model_path':  'models/member3/transformer_model.keras',
        'vector_path': 'models/member3/tokenizer.pkl',
    },
]

# Leaderboard order for /compare response
COMPARE_ORDER = ['lr', 'rf', 'xgb', 'cnn', 'lstm', 'transformer']


# ── Text Preprocessing ─────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    return re.sub(r'[^a-z0-9\s]', '', text)


# ── Model Loading ──────────────────────────────────────────────
def load_all_models():
    print("\n🔄 Loading models...\n")
    for cfg in MODEL_CONFIG:
        key = cfg['key']
        try:
            if cfg['type'] == 'ml':
                m = pickle.load(open(cfg['model_path'], 'rb'))
                v = pickle.load(open(cfg['vector_path'], 'rb'))
            else:
                from tensorflow.keras.models import load_model
                m = load_model(cfg['model_path'])
                v = pickle.load(open(cfg['vector_path'], 'rb'))

            models[key] = {
                'model':      m,
                'vectorizer': v,
                'type':       cfg['type'],
                'name':       cfg['name'],
                'member':     cfg['member'],
            }
            print(f"  ✅  {cfg['name']} loaded")
        except FileNotFoundError:
            print(f"  ⚠️   {cfg['name']} — model file not found (run training scripts first)")
        except Exception as exc:
            print(f"  ❌  {cfg['name']} — {exc}")

    print(f"\n✅  {len(models)}/6 models ready\n")


# ── Core Inference ─────────────────────────────────────────────
def run_prediction(model_key: str, text: str) -> dict | None:
    if model_key not in models:
        return None

    m      = models[model_key]
    cleaned = clean_text(text)

    if m['type'] == 'ml':
        vec         = m['vectorizer'].transform([cleaned])
        prediction  = int(m['model'].predict(vec)[0])
        proba       = m['model'].predict_proba(vec)[0]
        confidence  = float(proba[prediction]) * 100
    else:                                           # DL
        seq        = m['vectorizer'].texts_to_sequences([cleaned])
        padded     = pad_sequences(seq, maxlen=50)
        raw_score  = float(m['model'].predict(padded, verbose=0)[0][0])
        prediction = 1 if raw_score > 0.5 else 0
        confidence = raw_score * 100 if prediction == 1 else (1 - raw_score) * 100

    return {
        'model_key':  model_key,
        'model_name': m['name'],
        'model_type': m['type'].upper(),
        'member':     m['member'],
        'prediction': prediction,
        'label':      'spam' if prediction == 1 else 'ham',
        'confidence': round(confidence, 2),
    }


# ── Routes ─────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('ui', 'index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':        'ok',
        'loaded_models': list(models.keys()),
        'total_loaded':  len(models),
    })


@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        with open('evaluation_metrics.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({'error': 'evaluation_metrics.json not found'}), 404


@app.route('/predict', methods=['POST'])
def predict():
    payload   = request.get_json(force=True)
    text      = (payload.get('text') or '').strip()
    model_key = (payload.get('model') or 'lr').strip().lower()

    if not text:
        return jsonify({'error': 'text field is required'}), 400
    if model_key not in [c['key'] for c in MODEL_CONFIG]:
        return jsonify({'error': f'Unknown model key: {model_key}'}), 400

    result = run_prediction(model_key, text)
    if result is None:
        return jsonify({
            'error': f'Model "{model_key}" is not loaded. '
                     f'Please run the training scripts first.'
        }), 503

    return jsonify(result)


@app.route('/compare', methods=['POST'])
def compare():
    payload = request.get_json(force=True)
    text    = (payload.get('text') or '').strip()

    if not text:
        return jsonify({'error': 'text field is required'}), 400

    results = []
    for key in COMPARE_ORDER:
        r = run_prediction(key, text)
        if r:
            results.append(r)

    if not results:
        return jsonify({'error': 'No models are loaded. Run training scripts first.'}), 503

    # Mark the highest-confidence result
    best_idx = max(range(len(results)), key=lambda i: results[i]['confidence'])
    results[best_idx]['is_best'] = True

    return jsonify({'text': text, 'results': results})


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({'error': str(e)}), 500



# ── Entry Point ────────────────────────────────────────────────
if __name__ == '__main__':
    load_all_models()
    print("🚀  SpamGuard API running at http://localhost:5000\n")
    app.run(debug=True, port=5000, use_reloader=True)

