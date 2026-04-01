import tweepy
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BEARER_TOKEN = os.getenv("BEARER_TOKEN")

def fetch_tweets(query="AI", max_tweets=100):
    """
    Fetch tweets using Twitter API v2
    """
    client = tweepy.Client(bearer_token=BEARER_TOKEN)

    tweets = []

    response = client.search_recent_tweets(
        query=query,
        max_results=100,
        tweet_fields=["created_at", "lang"]
    )

    if response.data is not None:
        for tweet in response.data:
            tweets.append({
                "text": tweet.text,
                "created_at": tweet.created_at,
                "lang": tweet.lang
            })

    return tweets


def save_to_csv(tweets, filename="data/raw/tweets.csv"):
    """
    Save tweets to CSV
    """
    df = pd.DataFrame(tweets)
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} tweets to {filename}")


if __name__ == "__main__":
    query = input("Enter topic to search tweets: ")
    tweets = fetch_tweets(query=query, max_tweets=100)

    if tweets:
        save_to_csv(tweets)
    else:
        print("No tweets found.")