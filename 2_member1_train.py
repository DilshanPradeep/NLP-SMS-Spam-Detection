import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

os.makedirs('models/member1', exist_ok=True)
train_df = pd.read_csv('data/train.csv').dropna()
X_train, y_train = train_df['text'], train_df['label']

# --- 1. Machine Learning Model: Logistic Regression ---
print("Training Member 1: Logistic Regression...")
tfidf = TfidfVectorizer(max_features=3000)
X_train_ml = tfidf.fit_transform(X_train)

lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train_ml, y_train)

with open('models/member1/tfidf.pkl', 'wb') as f: pickle.dump(tfidf, f)
with open('models/member1/lr_model.pkl', 'wb') as f: pickle.dump(lr_model, f)

# --- 2. Deep Learning Model: 1D CNN ---
print("Training Member 1: 1D CNN...")
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_dl = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=50)

cnn_model = Sequential([
    Embedding(5000, 32, input_length=50),
    Conv1D(64, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_dl, y_train, epochs=3, batch_size=32, verbose=1)

with open('models/member1/tokenizer.pkl', 'wb') as f: pickle.dump(tokenizer, f)
cnn_model.save('models/member1/cnn_model.keras')

print("Member 1 Models Saved successfully!")