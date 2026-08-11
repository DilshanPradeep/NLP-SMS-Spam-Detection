import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    text = str(text).lower()
    return re.sub(r'[^a-z0-9\s]', '', text)

print("--- Spam Detection Tester ---")
print("Available Models:")
print("1: Member 1 - Logistic Regression (ML)")
print("2: Member 1 - 1D CNN (DL)")
print("3: Member 2 - Random Forest (ML)")
print("4: Member 2 - LSTM (DL)")
print("5: Member 3 - XGBoost (ML)")
print("6: Member 3 - Transformer (DL)")

choice = input("Enter the number of the model you want to test (1-6): ")

try:
    if choice == '1':
        model = pickle.load(open('models/member1/lr_model.pkl', 'rb'))
        vectorizer = pickle.load(open('models/member1/tfidf.pkl', 'rb'))
        m_type = 'ML'
    elif choice == '2':
        model = load_model('models/member1/cnn_model.keras')
        vectorizer = pickle.load(open('models/member1/tokenizer.pkl', 'rb'))
        m_type = 'DL'
    elif choice == '3':
        model = pickle.load(open('models/member2/rf_model.pkl', 'rb'))
        vectorizer = pickle.load(open('models/member2/tfidf.pkl', 'rb'))
        m_type = 'ML'
    elif choice == '4':
        model = load_model('models/member2/lstm_model.keras')
        vectorizer = pickle.load(open('models/member2/tokenizer.pkl', 'rb'))
        m_type = 'DL'
    elif choice == '5':
        model = pickle.load(open('models/member3/xgb_model.pkl', 'rb'))
        vectorizer = pickle.load(open('models/member3/tfidf.pkl', 'rb'))
        m_type = 'ML'
    elif choice == '6':
        model = load_model('models/member3/transformer_model.keras')
        vectorizer = pickle.load(open('models/member3/tokenizer.pkl', 'rb'))
        m_type = 'DL'
    else:
        print("Invalid choice!")
        exit()

    print(f"\nModel Loaded Successfully! (Type 'exit' to stop)")
    
    while True:
        user_input = input("\nEnter a message to test: ")
        if user_input.lower() == 'exit':
            break
            
        cleaned_msg = clean_text(user_input)
        
        if m_type == 'ML':
            processed_msg = vectorizer.transform([cleaned_msg])
            prediction = model.predict(processed_msg)[0]
        else: # DL
            processed_msg = pad_sequences(vectorizer.texts_to_sequences([cleaned_msg]), maxlen=50)
            prediction = (model.predict(processed_msg, verbose=0) > 0.5).astype("int32")[0][0]

        if prediction == 1:
            print("🚨 RESULT: This is a SPAM message!")
        else:
            print("✅ RESULT: This is a SAFE (Ham) message.")
            
except Exception as e:
    print(f"Error loading model! Please make sure you have run the training scripts first. Error: {e}")