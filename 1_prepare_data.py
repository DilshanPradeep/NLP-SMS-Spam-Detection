import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

os.makedirs('data', exist_ok=True)

print("Downloading Dataset...")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

df['cleaned_text'] = df['text'].apply(clean_text)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X_train_temp, X_test, y_train_temp, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.176, random_state=42)

pd.DataFrame({'text': X_train, 'label': y_train}).to_csv('data/train.csv', index=False)
pd.DataFrame({'text': X_val, 'label': y_val}).to_csv('data/val.csv', index=False)
pd.DataFrame({'text': X_test, 'label': y_test}).to_csv('data/test.csv', index=False)

print("Data Cleaning & Splitting Success! Saved in 'data/' folder.")