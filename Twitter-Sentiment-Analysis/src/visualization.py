import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/processed/cleaned_tweets.csv")

# Count sentiments
sentiment_counts = df["sentiment"].value_counts()

print(sentiment_counts)

# Plot bar chart
plt.figure()
sentiment_counts.plot(kind='bar')

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.show()