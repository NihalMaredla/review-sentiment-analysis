from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Load the English Amazon review dataset
print("Loading dataset...")
dataset = load_dataset("goosmanlei/amazon_reviews_multi", "en")

# Use a manageable sample for initial development
SAMPLE_SIZE = 20000

train_df = dataset["train"].shuffle(seed=42).select(
    range(min(SAMPLE_SIZE, len(dataset["train"])))
).to_pandas()

# Keep only the columns needed for sentiment analysis
df = train_df[["review_title", "review_body", "stars"]].copy()

# Handle missing values
df["review_title"] = df["review_title"].fillna("")
df["review_body"] = df["review_body"].fillna("")

# Combine title and review body
df["text"] = (
    df["review_title"].astype(str).str.strip()
    + " "
    + df["review_body"].astype(str).str.strip()
).str.strip()

# Convert star ratings to numeric values
df["stars"] = pd.to_numeric(df["stars"], errors="coerce")

# Remove unusable reviews
df = df.dropna(subset=["stars"])
df = df[df["text"].str.len() > 0]
df = df[df["stars"].between(1, 5)]

# Create sentiment labels
def get_sentiment(stars):
    if stars <= 2:
        return "Negative"
    elif stars == 3:
        return "Neutral"
    else:
        return "Positive"

df["sentiment"] = df["stars"].apply(get_sentiment)

# Save the processed dataset
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "amazon_reviews_clean.csv"
df[["text", "stars", "sentiment"]].to_csv(output_file, index=False)

# Display a summary
print("\nDataset preparation complete!")
print("Number of reviews:", len(df))
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())
print("\nSample reviews:")
print(df[["text", "stars", "sentiment"]].head(5).to_string(index=False))
print(f"\nSaved to: {output_file}")