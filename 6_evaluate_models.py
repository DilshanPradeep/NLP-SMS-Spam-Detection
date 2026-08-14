import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Try to import plotting libraries, handle gracefully if they are missing
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  matplotlib or seaborn not installed. Skipping confusion matrix plot generation.")

# ── Text Cleaning ────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    return re.sub(r'[^a-z0-9\s]', '', text)

# ── Model Configuration ──────────────────────────────────────────
MODEL_CONFIGS = {
    'lr': {
        'name': 'Logistic Regression',
        'type': 'ML',
        'model_path': 'models/member1/lr_model.pkl',
        'vector_path': 'models/member1/tfidf.pkl',
    },
    'cnn': {
        'name': '1D CNN',
        'type': 'DL',
        'model_path': 'models/member1/cnn_model.keras',
        'vector_path': 'models/member1/tokenizer.pkl',
    },
    'rf': {
        'name': 'Random Forest',
        'type': 'ML',
        'model_path': 'models/member2/rf_model.pkl',
        'vector_path': 'models/member2/tfidf.pkl',
    },
    'lstm': {
        'name': 'LSTM',
        'type': 'DL',
        'model_path': 'models/member2/lstm_model.keras',
        'vector_path': 'models/member2/tokenizer.pkl',
    },
    'xgb': {
        'name': 'XGBoost',
        'type': 'ML',
        'model_path': 'models/member3/xgb_model.pkl',
        'vector_path': 'models/member3/tfidf.pkl',
    },
    'transformer': {
        'name': 'Transformer',
        'type': 'DL',
        'model_path': 'models/member3/transformer_model.keras',
        'vector_path': 'models/member3/tokenizer.pkl',
    }
}

# Static attributes to maintain in the JSON file
STATIC_ATTRIBUTES = {
    'lr': {'short': 'LR', 'member': 1, 'color': '#3b82f6'},
    'rf': {'short': 'RF', 'member': 2, 'color': '#06b6d4'},
    'xgb': {'short': 'XGB', 'member': 3, 'color': '#f59e0b'},
    'cnn': {'short': 'CNN', 'member': 1, 'color': '#8b5cf6'},
    'lstm': {'short': 'LSTM', 'member': 2, 'color': '#ec4899'},
    'transformer': {'short': 'TF', 'member': 3, 'color': '#10b981'},
}

def main():
    print("🏁 Starting Evaluation Pipeline...\n")
    
    # 1. Load test data
    test_path = 'data/test.csv'
    if not os.path.exists(test_path):
        print(f"❌ Error: Test dataset not found at {test_path}. Please run 1_prepare_data.py first.")
        return
        
    df_test = pd.read_csv(test_path).dropna()
    X_test = df_test['text'].astype(str)
    y_test = df_test['label'].astype(int)
    
    print(f"📊 Loaded {len(df_test)} test samples (Spam: {sum(y_test)}, Ham: {len(y_test) - sum(y_test)})\n")
    
    # Load existing metrics as a starting point if available
    metrics_path = 'evaluation_metrics.json'
    results_json = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                results_json = json.load(f)
        except Exception as e:
            print(f"⚠️  Could not read existing evaluation_metrics.json: {e}")
            results_json = {}
            
    os.makedirs('models/plots', exist_ok=True)
    
    # Track performance for console display
    summary_data = []

    # 2. Evaluate each model
    for key, cfg in MODEL_CONFIGS.items():
        print(f"🔄 Evaluating {cfg['name']} ({cfg['type']})...")
        
        # Check if files exist
        if not os.path.exists(cfg['model_path']) or not os.path.exists(cfg['vector_path']):
            print(f"  ⚠️  Files missing for {cfg['name']}. Make sure you run model training first. Skipping.")
            continue
            
        try:
            # Load model and feature extractor
            if cfg['type'] == 'ML':
                with open(cfg['model_path'], 'rb') as f:
                    model = pickle.load(f)
                with open(cfg['vector_path'], 'rb') as f:
                    vectorizer = pickle.load(f)
                
                # Transform text and predict
                X_test_clean = X_test.apply(clean_text)
                X_test_vec = vectorizer.transform(X_test_clean)
                y_pred = model.predict(X_test_vec)
                
            else: # DL Model
                model = load_model(cfg['model_path'])
                with open(cfg['vector_path'], 'rb') as f:
                    tokenizer = pickle.load(f)
                
                # Tokenize & pad sequences
                X_test_clean = X_test.apply(clean_text)
                seqs = tokenizer.texts_to_sequences(X_test_clean)
                padded = pad_sequences(seqs, maxlen=50)
                
                raw_preds = model.predict(padded, verbose=0)
                y_pred = (raw_preds > 0.5).astype(int).flatten()
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred) * 100
            prec = precision_score(y_test, y_pred, zero_division=0) * 100
            rec = recall_score(y_test, y_pred, zero_division=0) * 100
            f1 = f1_score(y_test, y_pred, zero_division=0) * 100
            
            print(f"  🎯 Accuracy: {acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1: {f1:.2f}%")
            
            # Generate and save confusion matrix plot if matplotlib/seaborn are available
            if PLOTTING_AVAILABLE:
                plt.figure(figsize=(5, 4))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
                plt.title(f'Confusion Matrix - {cfg["name"]}')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                plt.tight_layout()
                plot_path = f'models/plots/{key}_confusion_matrix.png'
                plt.savefig(plot_path, dpi=150)
                plt.close()
                print(f"  🖼️  Saved confusion matrix plot to {plot_path}")
            
            # Update metrics JSON structure
            results_json[key] = {
                'name': cfg['name'],
                'short': STATIC_ATTRIBUTES[key]['short'],
                'type': cfg['type'],
                'member': STATIC_ATTRIBUTES[key]['member'],
                'accuracy': round(acc, 2),
                'precision': round(prec, 2),
                'recall': round(rec, 2),
                'f1': round(f1, 2),
                'color': STATIC_ATTRIBUTES[key]['color']
            }
            
            summary_data.append({
                'Model': cfg['name'],
                'Type': cfg['type'],
                'Accuracy': f"{acc:.2f}%",
                'Precision': f"{prec:.2f}%",
                'Recall': f"{rec:.2f}%",
                'F1-Score': f"{f1:.2f}%"
            })
            
        except Exception as e:
            print(f"  ❌ Error evaluating {cfg['name']}: {e}")
            
    # 3. Save updated metrics to JSON
    with open(metrics_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n💾 Saved updated metrics to {metrics_path}")
    
    # 4. Print Summary Table
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        print("\n🏆 Model Comparison Leaderboard:")
        print("=" * 80)
        print(df_summary.to_string(index=False))
        print("=" * 80)
        print("\nAll evaluations complete! Start the Flask API with 'python api.py' to serve these metrics.")
    else:
        print("\n⚠️ No models were evaluated. Please make sure training scripts have run successfully first.")

if __name__ == '__main__':
    main()
