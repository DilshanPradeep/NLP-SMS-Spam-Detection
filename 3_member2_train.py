import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

os.makedirs('models/member2', exist_ok=True)
train_df = pd.read_csv('data/train.csv').dropna()
X_train, y_train = train_df['text'], train_df['label']

# --- 1. Machine Learning Model: Random Forest ---
print("Training Member 2: Random Forest...")
tfidf = TfidfVectorizer(max_features=3000)
X_train_ml = tfidf.fit_transform(X_train)

rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train_ml, y_train)

with open('models/member2/tfidf.pkl', 'wb') as f: pickle.dump(tfidf, f)
with open('models/member2/rf_model.pkl', 'wb') as f: pickle.dump(rf_model, f)

# --- 2. Deep Learning Model: LSTM ---
print("Training Member 2: LSTM...")
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_dl = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=50)

lstm_model = Sequential([
    Embedding(5000, 32, input_length=50),
    LSTM(32),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_dl, y_train, epochs=3, batch_size=32, verbose=1)

with open('models/member2/tokenizer.pkl', 'wb') as f: pickle.dump(tokenizer, f)
lstm_model.save('models/member2/lstm_model.keras')

print("Member 2 Models Saved successfully!")