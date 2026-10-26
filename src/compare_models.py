from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from tensorflow.keras.preprocessing.sequence import pad_sequences

# File paths
DATA_PATH = Path("data/processed/amazon_reviews_clean.csv")
ML_PATH = Path("models/ml_model/sentiment_pipeline.joblib")
DL_DIR = Path("models/dl_model")

# Load dataset
df = pd.read_csv(DATA_PATH).dropna(subset=["text", "sentiment"])
df["text"] = df["text"].astype(str)
df = df[df["text"].str.strip() != ""]

# Recreate the same test split used during training
_, X_test, _, y_test = train_test_split(
    df["text"],
    df["sentiment"],
    test_size=0.20,
    random_state=42,
    stratify=df["sentiment"]
)

# Load ML model
ml_model = joblib.load(ML_PATH)
ml_predictions = ml_model.predict(X_test)

# Load LSTM artifacts
dl_model = tf.keras.models.load_model(DL_DIR / "sentiment_lstm.keras")
tokenizer = joblib.load(DL_DIR / "tokenizer.joblib")
encoder = joblib.load(DL_DIR / "label_encoder.joblib")

# Prepare text for LSTM
X_test_seq = pad_sequences(
    tokenizer.texts_to_sequences(X_test),
    maxlen=150,
    padding="post",
    truncating="post"
)

probabilities = dl_model.predict(X_test_seq, verbose=0)
dl_predictions = encoder.inverse_transform(
    np.argmax(probabilities, axis=1)
)

# Compare both models on identical reviews
results = []

for name, predictions in [
    ("ML - Logistic Regression", ml_predictions),
    ("DL - LSTM", dl_predictions)
]:
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Macro F1-score": f1_score(
            y_test, predictions,
            average="macro",
            zero_division=0
        )
    })

results_df = pd.DataFrame(results)
print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)
print("\nResults saved to models/model_comparison.csv")