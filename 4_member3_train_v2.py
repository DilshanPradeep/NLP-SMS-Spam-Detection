import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import pickle
import numpy as np
import keras
from keras import layers, ops
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.makedirs('models/member3_v2', exist_ok=True)

print("[INFO] Loading V2 preprocessed datasets...")
train_df = pd.read_csv('data/train_v2.csv').dropna()
val_df = pd.read_csv('data/val_v2.csv').dropna()

X_train, y_train = train_df['text'].astype(str), train_df['label'].astype(int)
X_val, y_val = val_df['text'].astype(str), val_df['label'].astype(int)

print(f"[DATA] Training samples: {len(X_train)} | Validation samples: {len(X_val)}")

print("\n" + "="*50)
print("[TRAIN] Training Member 3: XGBoost V2 (Enhanced)")
print("="*50)

tfidf_v2 = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    token_pattern=r'(?u)\b\w+\b',
    min_df=2,
    max_df=0.85
)

X_train_ml = tfidf_v2.fit_transform(X_train)
X_val_ml = tfidf_v2.transform(X_val)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"[WEIGHT] Calculated scale_pos_weight: {scale_pos_weight:.2f}")

xgb_model_v2 = XGBClassifier(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.06,
    scale_pos_weight=scale_pos_weight,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric='logloss',
    tree_method='hist',
    random_state=42
)

xgb_model_v2.fit(X_train_ml, y_train, eval_set=[(X_val_ml, y_val)], verbose=False)

with open('models/member3_v2/tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf_v2, f)

with open('models/member3_v2/xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model_v2, f)

print("[SUCCESS] XGBoost V2 trained and saved successfully to models/member3_v2/")

print("\n" + "="*50)
print("[TRAIN] Training Member 3: Custom Transformer V2 (Corrected Architecture)")
print("="*50)

vocab_size = 5000
maxlen = 60
embed_dim = 64
num_heads = 4
ffn_dim = 128
dropout_rate = 0.2

custom_filters = '!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n'
tokenizer_v2 = Tokenizer(num_words=vocab_size, filters=custom_filters, oov_token="<OOV>")
tokenizer_v2.fit_on_texts(X_train)

X_train_dl = pad_sequences(tokenizer_v2.texts_to_sequences(X_train), maxlen=maxlen, padding='post', truncating='post')
X_val_dl = pad_sequences(tokenizer_v2.texts_to_sequences(X_val), maxlen=maxlen, padding='post', truncating='post')

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

inputs = layers.Input(shape=(maxlen,), dtype="int32", name="input_tokens")
x = PositionalEmbedding(maxlen, vocab_size, embed_dim)(inputs)
x = TransformerBlock(embed_dim, num_heads, ffn_dim, rate=dropout_rate)(x)
x = layers.GlobalAveragePooling1D(name="global_pooling")(x)
x = layers.Dense(32, activation="relu", name="dense_head")(x)
x = layers.Dropout(dropout_rate)(x)
outputs = layers.Dense(1, activation="sigmoid", name="output_sigmoid")(x)

transformer_v2 = keras.Model(inputs=inputs, outputs=outputs, name="SMS_Spam_Transformer_V2")

transformer_v2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0006),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
)

class_weights_arr = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: float(class_weights_arr[0]), 1: float(class_weights_arr[1])}

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

transformer_v2.fit(
    X_train_dl,
    y_train,
    validation_data=(X_val_dl, y_val),
    epochs=12,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

with open('models/member3_v2/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer_v2, f)

transformer_v2.save('models/member3_v2/transformer_model.keras')

print("\n" + "="*50)
print("[COMPLETE] All Member 3 V2 Models trained and saved successfully to models/member3_v2/")
print("="*50)
