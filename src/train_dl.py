import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
DATA_PATH = Path("data/processed/amazon_reviews_clean.csv")
OUTPUT_DIR = Path("models/dl_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORDS = 20000
MAX_LENGTH = 150
EPOCHS = 5
BATCH_SIZE = 64

# 1. Load dataset
df = pd.read_csv(DATA_PATH).dropna(subset=["text", "sentiment"])
df["text"] = df["text"].astype(str)
df = df[df["text"].str.strip() != ""]

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["sentiment"],
    test_size=0.20,
    random_state=42,
    stratify=df["sentiment"]
)

# 2. Encode sentiment labels
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# 3. Convert text to integer sequences
tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(
    tokenizer.texts_to_sequences(X_train),
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)
X_test_seq = pad_sequences(
    tokenizer.texts_to_sequences(X_test),
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

# 4. Build LSTM model

model = Sequential([
    Input(shape=(MAX_LENGTH,)),
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=64,
        mask_zero=True
    ),
    LSTM(64, dropout=0.2),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 5. Train with early stopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=1,
    restore_best_weights=True
)

print("\nTraining LSTM...")
history = model.fit(
    X_train_seq,
    y_train_encoded,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)

# 6. Evaluate on held-out test set
probabilities = model.predict(X_test_seq, verbose=0)
predictions = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_test_encoded, predictions)
macro_f1 = f1_score(
    y_test_encoded,
    predictions,
    average="macro",
    zero_division=0
)

report = classification_report(
    y_test_encoded,
    predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
)

print("\n===== LSTM EVALUATION =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(report)

# 7. Save model, tokenizer and labels
model.save(OUTPUT_DIR / "sentiment_lstm.keras")
joblib.dump(tokenizer, OUTPUT_DIR / "tokenizer.joblib")
joblib.dump(encoder, OUTPUT_DIR / "label_encoder.joblib")

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)
    file.write(f"\nAccuracy: {accuracy:.4f}\n")
    file.write(f"Macro F1-score: {macro_f1:.4f}\n")

with open(
    OUTPUT_DIR / "training_config.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump({
        "max_words": MAX_WORDS,
        "max_length": MAX_LENGTH,
        "classes": encoder.classes_.tolist(),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1)
    }, file, indent=2)

print("\nLSTM model and preprocessing artifacts saved!")