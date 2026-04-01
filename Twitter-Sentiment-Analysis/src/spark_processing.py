from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Create Spark Session
spark = SparkSession.builder \
    .appName("Twitter Sentiment Analysis") \
    .getOrCreate()

# Load dataset
df = spark.read.csv("data/processed/cleaned_tweets.csv", header=True, inferSchema=True)

print("=== Sample Data ===")
df.show(5)

# Count total records
print("Total Records:", df.count())

# Group by sentiment
print("=== Sentiment Distribution ===")
sentiment_count = df.groupBy("sentiment").count()
sentiment_count.show()

# Calculate percentage
total = df.count()

print("=== Sentiment Percentage ===")
sentiment_percentage = sentiment_count.withColumn(
    "percentage",
    (col("count") / total) * 100
)

sentiment_percentage.show()

# Stop Spark session
spark.stop()