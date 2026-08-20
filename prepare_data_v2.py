import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.preprocessor import preprocess_sms

os.makedirs('data', exist_ok=True)

print("Loading raw dataset...")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
try:
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])
except Exception as e:
    print(f"Online download failed: {e}. Checking local cache...")
    raise e

df['cleaned_text_v2'] = df['text'].apply(preprocess_sms)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    df['cleaned_text_v2'], df['label'], test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp, y_train_temp, test_size=0.176, random_state=42
)

pd.DataFrame({'text': X_train, 'label': y_train}).to_csv('data/train_v2.csv', index=False)
pd.DataFrame({'text': X_val, 'label': y_val}).to_csv('data/val_v2.csv', index=False)
pd.DataFrame({'text': X_test, 'label': y_test}).to_csv('data/test_v2.csv', index=False)

X_raw_train_temp, X_raw_test, _, _ = train_test_split(
    df['text'], df['label'], test_size=0.15, random_state=42
)
pd.DataFrame({'raw_text': X_raw_test, 'label': y_test}).to_csv('data/test_raw.csv', index=False)

print(f"Data V2 preparation complete! Saved to data/train_v2.csv, data/val_v2.csv, data/test_v2.csv, data/test_raw.csv")
print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
