from pathlib import Path
import tempfile
import subprocess
import shutil

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from faster_whisper import WhisperModel

# -------------------- PAGE SETUP --------------------
st.set_page_config(
    page_title="ReviewSense | Sentiment AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent
ML_PATH = BASE / "models" / "ml_model" / "sentiment_pipeline.joblib"
DL_PATH = BASE / "models" / "dl_model" / "sentiment_lstm.keras"
TOKENIZER_PATH = BASE / "models" / "dl_model" / "tokenizer.joblib"
ENCODER_PATH = BASE / "models" / "dl_model" / "label_encoder.joblib"

# -------------------- CUSTOM DESIGN --------------------
st.markdown(
    """
    <style>
    .stApp {background:#0b1020;color:#edf1ff;}
    [data-testid="stSidebar"] {
        background:#11182b;border-right:1px solid #293650;
    }
    .block-container {max-width:1200px;padding-top:2rem;}
    .hero {
        background:linear-gradient(120deg,#18264a,#29234e,#16384b);
        padding:32px;border:1px solid #35436c;border-radius:22px;
        margin-bottom:25px;
    }
    .eyebrow {
        color:#a7b7ff;font-size:.78rem;font-weight:700;
        letter-spacing:2px;text-transform:uppercase;
    }
    .hero h1 {color:white;font-size:2.8rem;margin:.5rem 0;}
    .hero p {color:#c4cde5;font-size:1.05rem;}
    .panel {
        background:#121a2e;border:1px solid #293650;
        border-radius:17px;padding:22px;margin-bottom:15px;
    }
    div[data-testid="stMetric"] {
        background:#121a2e;border:1px solid #293650;
        border-radius:15px;padding:17px;
    }
    div.stButton>button {
        border-radius:10px;min-height:44px;font-weight:600;
    }
    .muted {color:#a5b1cb;font-size:.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_ml():
    return joblib.load(ML_PATH)


@st.cache_resource
def load_dl():
    model = tf.keras.models.load_model(DL_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, tokenizer, encoder


@st.cache_resource
def load_whisper():
    # The model downloads on first use. CPU/int8 keeps memory use lower.
    return WhisperModel("tiny", device="cpu", compute_type="int8")


# -------------------- PREDICTION --------------------
def analyze(text, choice):
    if choice == "Logistic Regression":
        model = load_ml()
        probabilities = model.predict_proba([text])[0]
        classes = model.classes_
    else:
        model, tokenizer, encoder = load_dl()
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(
            sequence, maxlen=150, padding="post", truncating="post"
        )
        probabilities = model.predict(padded, verbose=0)[0]
        classes = encoder.classes_

    index = int(np.argmax(probabilities))
    return {
        "sentiment": str(classes[index]).capitalize(),
        "confidence": float(probabilities[index]),
        "probabilities": {
            str(label).capitalize(): float(value)
            for label, value in zip(classes, probabilities)
        },
    }


def transcribe_audio(audio_bytes, suffix):
    """Transcribe uploaded audio and always remove its temporary file."""
    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix or ".wav", delete=False
        ) as audio_file:
            audio_file.write(audio_bytes)
            audio_path = audio_file.name

        whisper = load_whisper()
        segments, info = whisper.transcribe(audio_path)
        transcript = " ".join(
            segment.text.strip() for segment in segments
            if segment.text.strip()
        ).strip()
        return transcript, info.language
    finally:
        if audio_path:
            Path(audio_path).unlink(missing_ok=True)


def transcribe_video(video_bytes, suffix):
    """Extract a video's audio with FFmpeg, transcribe it, and clean temp files."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and add its bin folder to PATH, "
            "then restart the terminal/Streamlit app."
        )

    video_path = None
    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix or ".mp4", delete=False
        ) as video_file:
            video_file.write(video_bytes)
            video_path = video_file.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
            audio_path = audio_file.name

        command = [
            ffmpeg, "-y", "-i", video_path, "-vn", "-ac", "1",
            "-ar", "16000", "-c:a", "pcm_s16le", audio_path,
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=300
        )
        if completed.returncode != 0:
            details = (completed.stderr or "FFmpeg could not process this video.")[-1500:]
            raise RuntimeError(details)

        whisper = load_whisper()
        segments, info = whisper.transcribe(audio_path)
        transcript = " ".join(
            segment.text.strip() for segment in segments
            if segment.text.strip()
        ).strip()
        return transcript, info.language
    finally:
        if video_path:
            Path(video_path).unlink(missing_ok=True)
        if audio_path:
            Path(audio_path).unlink(missing_ok=True)


def show_result(result, model_name):
    sentiment = result["sentiment"]
    confidence = result["confidence"]

    if sentiment == "Positive":
        st.success("🟢 Predicted sentiment: Positive")
    elif sentiment == "Negative":
        st.error("🔴 Predicted sentiment: Negative")
    else:
        st.warning("🟡 Predicted sentiment: Neutral")

    col_a, col_b = st.columns(2)
    col_a.metric("Prediction confidence", f"{confidence:.1%}")
    col_b.metric("Selected model", model_name)

    if confidence < 0.60:
        st.info(
            "Confidence is relatively low. Treat this prediction cautiously, "
            "especially for sarcasm or mixed opinions."
        )

    st.markdown("### Class probabilities")
    chart_data = pd.DataFrame.from_dict(
        result["probabilities"],
        orient="index",
        columns=["Probability"],
    )
    st.bar_chart(chart_data, horizontal=True)


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("# 🔎 ReviewSense")
    st.caption("SENTIMENT INTELLIGENCE")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Review Analyzer", "Audio Analyzer", "Video Analyzer", "About"],
    )

    st.divider()
    choice = st.selectbox(
        "Prediction model",
        ["Logistic Regression", "LSTM"],
    )

    st.markdown("### Sentiment classes")
    st.markdown("🟢 Positive\n\n🟡 Neutral\n\n🔴 Negative")
    st.divider()
    st.caption("Machine Learning · Deep Learning")


# -------------------- SHARED HEADER --------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI-powered review intelligence</div>
      <h1>Understand the feeling<br>behind every review.</h1>
      <p>Explore customer feedback with machine learning and deep learning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- PAGE: AUDIO ANALYZER --------------------
if page == "Audio Analyzer":
    st.markdown("## 🎙️ Audio sentiment analysis")
    st.markdown(
        '<p class="muted">Upload a recording. ReviewSense transcribes the '
        'spoken words and predicts their sentiment.</p>',
        unsafe_allow_html=True,
    )
    st.info(
        "This feature analyzes the words spoken, not vocal tone or emotion. "
        "The first transcription may take longer while the Whisper model downloads."
    )

    audio = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
        help="Choose a short recording with clear speech.",
    )

    if audio is not None:
        st.audio(audio)

        if st.button("🎙️ Transcribe and analyze", type="primary"):
            try:
                with st.spinner("Transcribing audio... this may take a while the first time."):
                    transcript, language = transcribe_audio(
                        audio.getvalue(),
                        Path(audio.name).suffix or ".wav",
                    )

                if not transcript:
                    st.warning("No speech was detected. Try a clearer recording.")
                else:
                    st.markdown("### 📝 Transcript")
                    st.text_area(
                        "Transcribed speech",
                        value=transcript,
                        height=150,
                        disabled=True,
                    )
                    st.caption(f"Detected language: {language}")

                    with st.spinner("Analyzing transcript sentiment..."):
                        result = analyze(transcript, choice)

                    st.markdown("## Analysis results")
                    show_result(result, choice)

            except Exception as error:
                st.error(
                    "Audio processing failed. Check that the audio format is "
                    "supported and review the app logs for details."
                )
                st.exception(error)

# -------------------- PAGE: VIDEO ANALYZER --------------------
elif page == "Video Analyzer":
    st.markdown("## 🎬 Video sentiment analysis")
    st.markdown(
        '<p class="muted">Upload a video. ReviewSense extracts its audio, '
        'transcribes spoken words, and predicts their sentiment.</p>',
        unsafe_allow_html=True,
    )
    st.info(
        "This feature analyzes spoken words only. It does not interpret facial "
        "expressions, scenes, or visual emotion. Longer videos take more time to process."
    )

    video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "avi", "mkv", "webm", "mpeg", "mpg"],
        help="Choose a video with clear speech. Short videos are recommended.",
        key="video_uploader",
    )

    if video is not None:
        st.video(video)
        if st.button("🎬 Transcribe video and analyze", type="primary"):
            try:
                with st.spinner(
                    "Extracting audio and transcribing speech... "
                    "The first run may take longer while Whisper downloads."
                ):
                    transcript, language = transcribe_video(
                        video.getvalue(), Path(video.name).suffix or ".mp4"
                    )

                if not transcript:
                    st.warning(
                        "No speech was detected in this video. Try a video with clearer speech."
                    )
                else:
                    st.markdown("### 📝 Video transcript")
                    st.text_area(
                        "Transcribed speech", value=transcript,
                        height=180, disabled=True, key="video_transcript",
                    )
                    st.caption(f"Detected language: {language}")
                    with st.spinner("Analyzing transcript sentiment..."):
                        result = analyze(transcript, choice)
                    st.markdown("## Analysis results")
                    show_result(result, choice)
            except subprocess.TimeoutExpired:
                st.error("Video processing took too long. Try a shorter video.")
            except Exception as error:
                st.error(
                    "Video processing failed. Check the video format, confirm FFmpeg "
                    "is installed, and review the app logs for details."
                )
                st.exception(error)

# -------------------- PAGE: ABOUT --------------------
elif page == "About":
    st.markdown("## About ReviewSense")
    st.markdown(
        """
        <div class="panel">
          <h3>Project overview</h3>
          <p class="muted">ReviewSense classifies Amazon product review text
          into Positive, Neutral, or Negative sentiment.</p>
          <h3>Logistic Regression</h3>
          <p class="muted">A traditional machine learning model using TF-IDF
          features.</p>
          <h3>LSTM</h3>
          <p class="muted">A neural network that learns patterns in sequences
          of review tokens.</p>
          <h3>Audio and video analysis</h3>
          <p class="muted">Audio recordings and audio extracted from uploaded videos are transcribed using Whisper, then the transcript is passed to the selected sentiment model. This version does not analyze vocal tone, facial expressions, or visual scenes.</p>
          <h3>Limitations</h3>
          <p class="muted">Predictions are estimates. Sarcasm, mixed opinions,
          transcription errors, and unusual wording can reduce reliability.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------- PAGE: REVIEW ANALYZER --------------------
else:
    st.markdown("## 📝 Review workspace")
    st.markdown(
        '<p class="muted">Enter a review or choose an example below.</p>',
        unsafe_allow_html=True,
    )

    examples = {
        "Choose an example": "",
        "Positive review": (
            "Excellent quality, works perfectly, and worth every penny!"
        ),
        "Negative review": (
            "Very poor quality. It broke after two days and I regret buying it."
        ),
        "Neutral review": (
            "The product arrived on time and matches the description."
        ),
    }
    example = st.selectbox("Try a sample review", list(examples))
    review = st.text_area(
        "Product review",
        value=examples[example],
        placeholder="Paste an Amazon product review here...",
        height=160,
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        submitted = st.button(
            "✨ Analyze review", type="primary", use_container_width=True
        )
    with col2:
        st.caption(f"Current model: {choice}")

    if submitted:
        if not review.strip():
            st.warning("Please enter a review first.")
        else:
            try:
                with st.spinner("Analyzing your review..."):
                    result = analyze(review.strip(), choice)
                st.divider()
                st.markdown("## Analysis results")
                show_result(result, choice)
                with st.expander("View analyzed review"):
                    st.write(review)
            except Exception as error:
                st.error("The analysis failed. Check the app logs for details.")
                st.exception(error)

# -------------------- FOOTER --------------------
st.divider()
st.markdown(
    '<p class="muted" style="text-align:center;">'
    "ReviewSense · Amazon Review Sentiment Analysis · ML + Deep Learning"
    "</p>",
    unsafe_allow_html=True,
)
