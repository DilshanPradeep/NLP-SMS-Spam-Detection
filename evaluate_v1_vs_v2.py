import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import re
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import keras
from keras import layers, ops
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.preprocessor import preprocess_sms

@keras.saving.register_keras_serializable(package="SMS_Spam")
class PositionalEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        length = ops.shape(x)[-1]
        positions = ops.arange(start=0, stop=length, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def compute_mask(self, inputs, mask=None):
        return ops.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

@keras.saving.register_keras_serializable(package="SMS_Spam")
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ffn_dim, rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=rate)
        self.ffn = keras.Sequential([
            layers.Dense(ffn_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, mask=None):
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, :]
        attn_output = self.att(query=inputs, value=inputs, key=inputs, attention_mask=attn_mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "rate": self.rate,
        })
        return config

def clean_v1(text):
    text = str(text).lower()
    return re.sub(r'[^a-z0-9\s]', '', text)

def main():
    print("\n" + "="*80)
    print(" [BENCHMARK EVALUATION] MEMBER 3 V1 vs V2 MODELS")
    print("="*80)

    if os.path.exists('data/test_raw.csv'):
        df_test = pd.read_csv('data/test_raw.csv').dropna()
        raw_texts = df_test['raw_text'].astype(str)
        y_test = df_test['label'].astype(int).values
    elif os.path.exists('data/test.csv'):
        df_test = pd.read_csv('data/test.csv').dropna()
        raw_texts = df_test['text'].astype(str)
        y_test = df_test['label'].astype(int).values
    else:
        print("[ERROR] Test dataset not found!")
        return

    n_samples = len(y_test)
    n_spam = int(sum(y_test))
    n_ham = n_samples - n_spam
    print(f"[TEST DATA] {n_samples} total samples | Spam: {n_spam} ({(n_spam/n_samples)*100:.1f}%) | Ham: {n_ham} ({(n_ham/n_samples)*100:.1f}%)\n")

    models_to_eval = [
        {
            'id': 'xgb_v1',
            'name': 'XGBoost V1 (Baseline)',
            'family': 'XGBoost',
            'version': 'V1',
            'model_path': 'models/member3/xgb_model.pkl',
            'vec_path': 'models/member3/tfidf.pkl',
            'type': 'ML',
            'preprocess': 'v1',
            'maxlen': None
        },
        {
            'id': 'xgb_v2',
            'name': 'XGBoost V2 (Enhanced)',
            'family': 'XGBoost',
            'version': 'V2',
            'model_path': 'models/member3_v2/xgb_model.pkl',
            'vec_path': 'models/member3_v2/tfidf.pkl',
            'type': 'ML',
            'preprocess': 'v2',
            'maxlen': None
        },
        {
            'id': 'tf_v1',
            'name': 'Transformer V1 (Baseline)',
            'family': 'Transformer',
            'version': 'V1',
            'model_path': 'models/member3/transformer_model.keras',
            'vec_path': 'models/member3/tokenizer.pkl',
            'type': 'DL',
            'preprocess': 'v1',
            'maxlen': 50
        },
        {
            'id': 'tf_v2',
            'name': 'Transformer V2 (Corrected)',
            'family': 'Transformer',
            'version': 'V2',
            'model_path': 'models/member3_v2/transformer_model.keras',
            'vec_path': 'models/member3_v2/tokenizer.pkl',
            'type': 'DL',
            'preprocess': 'v2',
            'maxlen': 60
        }
    ]

    results = {}
    summary_rows = []

    for cfg in models_to_eval:
        print(f"[EVAL] Evaluating {cfg['name']}...")
        if not os.path.exists(cfg['model_path']) or not os.path.exists(cfg['vec_path']):
            print(f"   [WARN] Skipping {cfg['name']} -- model files not found.")
            continue

        try:
            if cfg['preprocess'] == 'v1':
                processed_texts = raw_texts.apply(clean_v1)
            else:
                processed_texts = raw_texts.apply(preprocess_sms)

            if cfg['type'] == 'ML':
                with open(cfg['model_path'], 'rb') as f: model = pickle.load(f)
                with open(cfg['vec_path'], 'rb') as f: vec = pickle.load(f)
                X_vec = vec.transform(processed_texts)
                y_pred = model.predict(X_vec)
            else:
                model = load_model(cfg['model_path'])
                with open(cfg['vec_path'], 'rb') as f: tokenizer = pickle.load(f)
                seqs = tokenizer.texts_to_sequences(processed_texts)
                padded = pad_sequences(seqs, maxlen=cfg['maxlen'], padding='post', truncating='post')
                raw_preds = model.predict(padded, verbose=0)
                y_pred = (raw_preds > 0.5).astype(int).flatten()

            acc = accuracy_score(y_test, y_pred) * 100
            prec = precision_score(y_test, y_pred, zero_division=0) * 100
            rec = recall_score(y_test, y_pred, zero_division=0) * 100
            f1 = f1_score(y_test, y_pred, zero_division=0) * 100
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            results[cfg['id']] = {
                'name': cfg['name'],
                'family': cfg['family'],
                'version': cfg['version'],
                'type': cfg['type'],
                'accuracy': round(acc, 2),
                'precision': round(prec, 2),
                'recall': round(rec, 2),
                'f1': round(f1, 2),
                'cm': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}
            }

            summary_rows.append({
                'Model': cfg['name'],
                'Type': cfg['type'],
                'Accuracy': f"{acc:.2f}%",
                'Precision': f"{prec:.2f}%",
                'Recall': f"{rec:.2f}%",
                'F1-Score': f"{f1:.2f}%",
                'TN': int(tn),
                'FP': int(fp),
                'FN': int(fn),
                'TP': int(tp)
            })

            print(f"   [RESULT] Accuracy: {acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1: {f1:.2f}% | (FP={fp}, FN={fn})")

        except Exception as e:
            print(f"   [ERROR] Error evaluating {cfg['name']}: {e}")

    if summary_rows:
        df_res = pd.DataFrame(summary_rows)
        print("\n" + "="*95)
        print(" FINAL V1 vs V2 COMPARISON TABLE")
        print("="*95)
        print(df_res[['Model', 'Type', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'FP', 'FN']].to_string(index=False))
        print("="*95)

    with open('evaluation_comparison_v1_vs_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n[SAVED] Comparison results saved to evaluation_comparison_v1_vs_v2.json")

if __name__ == '__main__':
    main()
