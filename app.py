import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import tkinter as tk
from tkinter import messagebox

# Load the ensemble model and TF-IDF vectorizer
ensemble_model = joblib.load('ensemble_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load the LSTM model and tokenizer
lstm_model = load_model('lstm_model.h5')
tokenizer = joblib.load('tokenizer.pkl')

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()                # Convert to lowercase
    return text

def predict(text):
    text = preprocess_text(text)
    
    # TF-IDF prediction
    tfidf_vector = tfidf_vectorizer.transform([text])
    ensemble_pred = ensemble_model.predict(tfidf_vector)
    ensemble_label = 'REAL' if ensemble_pred[0] == 'REAL' else 'FAKE'
    
    # LSTM prediction
    seq = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=200)
    lstm_pred = lstm_model.predict(seq)
    lstm_label = 'REAL' if lstm_pred[0][0] > 0.5 else 'FAKE'
    
    return ensemble_label, lstm_label

# Function to handle prediction and display results
def make_prediction():
    text = news_entry.get("1.0", tk.END).strip()  # Get text from entry
    if not text:
        messagebox.showwarning("Input Error", "Please enter some news content.")
        return
    
    # Get predictions
    ensemble_prediction, lstm_prediction = predict(text)
    
    # Display predictions
    result_text = f"Ensemble Model Prediction: {ensemble_prediction}\nLSTM Model Prediction: {lstm_prediction}"
    result_label.config(text=result_text)

# Set up GUI
root = tk.Tk()
root.title("News Content Predictor")
root.geometry("500x400")

# News content input
news_label = tk.Label(root, text="Enter News Content:", font=("Arial", 12))
news_label.pack(pady=10)

news_entry = tk.Text(root, height=10, width=50, wrap="word")
news_entry.pack(pady=10)

# Predict button
predict_button = tk.Button(root, text="Predict", font=("Arial", 12), command=make_prediction)
predict_button.pack(pady=20)

# Result display
result_label = tk.Label(root, text="", font=("Arial", 12), fg="blue")
result_label.pack(pady=10)

# Run the GUI loop
root.mainloop()
