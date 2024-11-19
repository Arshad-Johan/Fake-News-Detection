import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
import joblib
import nltk

print("Downloading NLTK resources...")
nltk.download('stopwords')
nltk.download('punkt') 
nltk.download('punkt_tab') 
print("Downloaded NLTK resources.")

print("Loading datasets...")

try:
    real_news = pd.read_csv('True.csv', on_bad_lines='skip', engine='python')
    fake_news = pd.read_csv('Fake.csv', on_bad_lines='skip', engine='python')
    print("Datasets loaded.")
except pd.errors.ParserError as e:
    print(f"Error loading datasets: {e}")

print("Datasets loaded.")

print("Adding labels and selecting necessary columns...")
real_news['label'] = 'REAL'
fake_news['label'] = 'FAKE'
real_news = real_news[['text', 'label']]
fake_news = fake_news[['text', 'label']]
print("Labels added.")


print("Merging datasets...")
merged_df = pd.concat([real_news, fake_news], ignore_index=True)
print(f"Datasets merged. Total samples: {len(merged_df)}")

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(merged_df['text'], merged_df['label'], test_size=0.2, random_state=42)
print("Dataset split into training and test sets.")

print("Test Text Samples:")
print(X_test.head(20))
print("Test Labels:")
print(y_test.head(20))


print("Starting TF-IDF vectorization...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
print("TF-IDF vectorization completed.")

print("Training Word2Vec model...")
tokenized_texts = [word_tokenize(text) for text in X_train]
w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
print("Word2Vec model trained.")

def text_to_w2v(text):
    words = word_tokenize(text)
    words = [word for word in words if word in w2v_model.wv]
    if len(words) == 0:
        return np.zeros(100)
    return np.mean(w2v_model.wv[words], axis=0)

print("Converting texts to Word2Vec embeddings...")
w2v_train = np.array([text_to_w2v(text) for text in X_train])
w2v_test = np.array([text_to_w2v(text) for text in X_test])
print("Text conversion to Word2Vec embeddings completed.")

print("Initializing ML models...")
log_reg = LogisticRegression(max_iter=200)
rf = RandomForestClassifier(n_estimators=100)
svm = SVC(kernel='linear', probability=True)
print("ML models initialized.")

print("Training ensemble model...")
ensemble_model = VotingClassifier(estimators=[
    ('lr', log_reg),
    ('rf', rf),
    ('svm', svm)
], voting='soft')
ensemble_model.fit(tfidf_train, y_train)
ensemble_pred = ensemble_model.predict(tfidf_test)
print(f"Ensemble Model Accuracy (TF-IDF): {accuracy_score(y_test, ensemble_pred) * 100:.2f}%")

print("Saving ensemble model and TF-IDF vectorizer...")
joblib.dump(ensemble_model, 'ensemble_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
print("Ensemble model and vectorizer saved.")

print("Preparing data for LSTM model...")
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=200)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=200)
print("Data prepared for LSTM model.")

print("Building and training LSTM model...")
lstm_model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=200),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_seq, y_train.replace({'FAKE': 0, 'REAL': 1}), epochs=5, batch_size=64, validation_data=(X_test_seq, y_test.replace({'FAKE': 0, 'REAL': 1})))
print("LSTM model trained.")

print("Saving LSTM model...")
lstm_model.save("lstm_model.h5")
print("LSTM model saved.")
print("Saving tokenizer...")
joblib.dump(tokenizer, 'tokenizer.pkl')
print("Tokenizer saved.")