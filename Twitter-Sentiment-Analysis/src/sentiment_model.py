import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords

# Stopwords
stop_words = set(stopwords.words('english'))


# Load data
def load_data(file_path="data/processed/cleaned_tweets.csv"):
    df = pd.read_csv(file_path)

    df = df.dropna(subset=["clean_text", "sentiment"])
    df = df[df["clean_text"].str.strip() != ""]

    # 🔥 REMOVE noisy class
    df = df[df["sentiment"] != "Irrelevant"]

    return df


# Advanced cleaning
def advanced_clean(text):
    words = text.split()
    
    # Remove stopwords
    words = [w for w in words if w not in stop_words]
    
    return " ".join(words)


# Train model
def train_model(df):
    # Apply advanced cleaning
    df["clean_text"] = df["clean_text"].apply(advanced_clean)

    X = df["clean_text"]
    y = df["sentiment"]

    # Better TF-IDF (n-grams)
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )

    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )

    # Improved Logistic Regression
    model = LogisticRegression(max_iter=300, C=1.5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return model, vectorizer


if __name__ == "__main__":
    df = load_data()
    model, vectorizer = train_model(df)

    import pickle
    import os

    # Create models folder
    os.makedirs("models", exist_ok=True)

    # Save model
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save vectorizer
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("Model and vectorizer saved successfully!")