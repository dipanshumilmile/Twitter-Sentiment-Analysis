import pandas as pd
import re

# Load dataset
def load_data(file_path="data/raw/tweets.csv"):
    df = pd.read_csv(file_path, header=None)
    
    # Assign column names based on your dataset
    df.columns = ["id", "entity", "sentiment", "text"]
    
    return df


# Clean text function
def clean_text(text):
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    
    # Remove mentions (@user)
    text = re.sub(r"@\w+", "", text)
    
    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


# Apply preprocessing
def preprocess_data(df):
    df["clean_text"] = df["text"].apply(clean_text)
    
    # Keep only required columns
    df = df[["clean_text", "sentiment"]]
    
    return df


# Save cleaned data
def save_data(df, file_path="data/processed/cleaned_tweets.csv"):
    import os
    os.makedirs("data/processed", exist_ok=True)
    
    df.to_csv(file_path, index=False)
    print(f"Cleaned data saved to {file_path}")


if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    save_data(df)
    
    print(df.head())