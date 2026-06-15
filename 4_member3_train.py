import pandas as pd
import os
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.makedirs('models/member3', exist_ok=True)
train_df = pd.read_csv('data/train.csv').dropna()
X_train, y_train = train_df['text'], train_df['label']

# --- 1. Machine Learning Model: XGBoost ---
print("Training Member 3: XGBoost...")
tfidf = TfidfVectorizer(max_features=3000)
X_train_ml = tfidf.fit_transform(X_train)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_ml, y_train)

with open('models/member3/tfidf.pkl', 'wb') as f: pickle.dump(tfidf, f)
with open('models/member3/xgb_model.pkl', 'wb') as f: pickle.dump(xgb_model, f)

# --- 2. Deep Learning Model: Custom Transformer Network ---
print("Training Member 3: Transformer Network...")
vocab_size = 5000
maxlen = 50
embed_dim = 32

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)
X_train_dl = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=maxlen)

# Transformer Architecture
inputs = layers.Input(shape=(maxlen,))
embedding_layer = layers.Embedding(vocab_size, embed_dim)(inputs)

# Simple Positional Encoding
positions = tf.range(start=0, limit=maxlen, delta=1)
pos_embedding = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)(positions)
x = embedding_layer + pos_embedding

# Attention Block
attention_output = layers.MultiHeadAttention(num_heads=2, key_dim=embed_dim)(x, x)
x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

# Feed Forward
ffn_output = layers.Dense(32, activation="relu")(x)
x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

transformer_model = tf.keras.Model(inputs=inputs, outputs=outputs)
transformer_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

transformer_model.fit(X_train_dl, y_train, epochs=3, batch_size=32, verbose=1)

with open('models/member3/tokenizer.pkl', 'wb') as f: pickle.dump(tokenizer, f)
transformer_model.save('models/member3/transformer_model.keras')

print("Member 3 Models Saved successfully!")