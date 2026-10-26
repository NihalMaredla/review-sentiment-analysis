import joblib
from pathlib import Path

MODEL_PATH = Path("models/ml_model/sentiment_pipeline.joblib")


def load_model():
    """Load the saved sentiment analysis pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run src/train_ml.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_sentiment(text, model=None):
    """Predict sentiment for a single review."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please provide a non-empty review.")

    if model is None:
        model = load_model()

    sentiment = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    confidence = max(probabilities)

    return {
        "sentiment": sentiment,
        "confidence": float(confidence),
        "probabilities": {
            label: float(probability)
            for label, probability in zip(
                model.classes_, probabilities
            )
        }
    }


if __name__ == "__main__":
    review = input("Enter a product review: ")

    try:
        result = predict_sentiment(review)

        print("\nPredicted sentiment:", result["sentiment"])
        print(
            "Model confidence:",
            f"{result['confidence']:.2%}"
        )

        print("\nClass probabilities:")
        for label, probability in result["probabilities"].items():
            print(f"{label}: {probability:.2%}")

    except (ValueError, FileNotFoundError) as error:
        print("Error:", error)