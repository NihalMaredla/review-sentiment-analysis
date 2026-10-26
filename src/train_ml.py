import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

# 1. Load the prepared dataset
data_path = Path("data/processed/amazon_reviews_clean.csv")
df = pd.read_csv(data_path).dropna(subset=["text", "sentiment"])

X = df["text"].astype(str)
y = df["sentiment"]

print("Dataset loaded:", len(df), "reviews")
print("\nClass distribution:")
print(y.value_counts())

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining reviews:", len(X_train))
print("Testing reviews:", len(X_test))

# 3. Build the ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])

# 4. Train the model
print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

# 5. Evaluate on unseen test data
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)

print("\n===== MODEL EVALUATION =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print("\nClassification report:")
report = classification_report(
    y_test, y_pred, zero_division=0
)
print(report)

# 6. Save the trained model and results
output_dir = Path("models/ml_model")
output_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(model, output_dir / "sentiment_pipeline.joblib")

with open(output_dir / "classification_report.txt", "w",
          encoding="utf-8") as file:
    file.write(report)
    file.write(f"\nAccuracy: {accuracy:.4f}\n")
    file.write(f"Macro F1-score: {macro_f1:.4f}\n")

# 7. Save confusion matrix visualization
labels = ["Negative", "Neutral", "Positive"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)
display.plot(cmap="Blues", values_format="d")
plt.title("Sentiment Analysis - Logistic Regression")
plt.tight_layout()
plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
plt.close()

# 8. Test a new review
sample_review = (
    "The product is excellent and works perfectly."
)
prediction = model.predict([sample_review])[0]

print("\nSample review:", sample_review)
print("Predicted sentiment:", prediction)

print("\nModel saved successfully!")