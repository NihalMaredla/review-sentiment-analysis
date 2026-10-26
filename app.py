from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

ML_MODEL_PATH = BASE_DIR / "models" / "ml_model" / "sentiment_pipeline.joblib"
DL_MODEL_PATH = BASE_DIR / "models" / "dl_model" / "sentiment_lstm.keras"
TOKENIZER_PATH = BASE_DIR / "models" / "dl_model" / "tokenizer.joblib"
ENCODER_PATH = BASE_DIR / "models" / "dl_model" / "label_encoder.joblib"


# --------------------------------------------------
# Load models only when needed
# --------------------------------------------------
@st.cache_resource
def load_ml_model():
    return joblib.load(ML_MODEL_PATH)


@st.cache_resource
def load_dl_artifacts():
    model = tf.keras.models.load_model(DL_MODEL_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, tokenizer, encoder


def predict_ml(text):
    model = load_ml_model()
    probabilities = model.predict_proba([text])[0]
    index = int(np.argmax(probabilities))

    return {
        "sentiment": model.classes_[index],
        "confidence": float(probabilities[index]),
        "probabilities": dict(zip(model.classes_, probabilities))
    }


def predict_dl(text):
    model, tokenizer, encoder = load_dl_artifacts()

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        sequence,
        maxlen=150,
        padding="post",
        truncating="post"
    )

    probabilities = model.predict(padded, verbose=0)[0]
    index = int(np.argmax(probabilities))

    return {
        "sentiment": encoder.classes_[index],
        "confidence": float(probabilities[index]),
        "probabilities": dict(zip(encoder.classes_, probabilities))
    }


# --------------------------------------------------
# User interface
# --------------------------------------------------
st.title("📊 Amazon Review Sentiment Analyzer")
st.write(
    "Analyze Amazon product reviews using Machine Learning "
    "and Deep Learning models."
)

st.info(
    "Current version: text-based sentiment analysis. "
    "Audio and video analysis will be added next."
)

model_choice = st.selectbox(
    "Choose a sentiment model",
    ["Logistic Regression (ML)", "LSTM (Deep Learning)"]
)

review = st.text_area(
    "Enter an Amazon product review",
    placeholder="Example: The product quality is excellent and delivery was fast!",
    height=150
)

if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        try:
            with st.spinner("Analyzing your review..."):
                if model_choice == "Logistic Regression (ML)":
                    result = predict_ml(review)
                else:
                    result = predict_dl(review)

            sentiment = result["sentiment"].capitalize()
            confidence = result["confidence"]

            st.subheader("Analysis Result")

            if sentiment == "Positive":
                st.success(f"😊 Sentiment: {sentiment}")
            elif sentiment == "Negative":
                st.error(f"😞 Sentiment: {sentiment}")
            else:
                st.warning(f"😐 Sentiment: {sentiment}")

            st.metric("Prediction Confidence", f"{confidence:.1%}")

            st.subheader("Class Probabilities")
            probability_df = pd.DataFrame({
                "Sentiment": list(result["probabilities"].keys()),
                "Probability": list(result["probabilities"].values())
            })

            st.bar_chart(
                probability_df.set_index("Sentiment")
            )

        except Exception as error:
            st.error(f"An error occurred: {error}")
            st.caption(
                "Check that the model files exist and that "
                "the .venv312 environment is active."
            )

st.divider()
st.caption("Amazon Review Sentiment Analysis | ML + Deep Learning")