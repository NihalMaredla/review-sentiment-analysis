
# ReviewSense — Amazon Review Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Live Demo](https://img.shields.io/badge/Live-Demo-success?logo=streamlit)](https://reviewsense-nihal-demo.streamlit.app/)

**ReviewSense** is a sentiment-analysis web application that classifies text as **Positive**, **Neutral**, or **Negative**. It combines traditional machine learning and deep learning with speech transcription, allowing users to analyze written reviews, audio recordings, and videos containing speech.

**🚀 Live Demo:** [Open ReviewSense](https://reviewsense-nihal-demo.streamlit.app/)

**📂 Repository:** [View the source code on GitHub](https://github.com/NihalMaredla/review-sentiment-analysis)

> **Note:** Audio and video analysis classify the transcribed words. They do not detect emotion from vocal tone, facial expressions, or visual scenes.

## Features

-   **Review Analyzer:** Enter a review or try a sample review to get a
    sentiment prediction.
-   **Audio Analyzer:** Upload an audio file; Whisper transcribes
    speech, then the selected sentiment model analyzes the transcript.
-   **Video Analyzer:** Upload a supported video; FFmpeg
    extracts/decodes its audio for transcription, then the selected
    model analyzes the transcript.
-   **Two prediction models:** Compare Logistic Regression and LSTM
    predictions.
-   **Class probabilities:** View the model's estimated probabilities
    for each sentiment class.
-   **Streamlit interface:** A multi-page style navigation sidebar and
    interactive results.

## Models and dataset

### Logistic Regression (Machine Learning)

The ML pipeline uses **TF-IDF text features** with Logistic Regression
for three-class sentiment classification.

### LSTM (Deep Learning)

The LSTM model converts text into token sequences, pads/truncates
sequences to a fixed length, and predicts one of the three sentiment
classes.

### Dataset

The project dataset was prepared from the English portion of the Hugging
Face dataset `goosmanlei/amazon_reviews_multi`. The prepared subset
contains **20,000 samples**. Ratings were mapped to sentiment as
follows:

-   Ratings **1--2** → Negative
-   Rating **3** → Neutral
-   Ratings **4--5** → Positive

### Recorded evaluation results

  Model                            Accuracy   Macro F1
  ------------------------------ ---------- ----------
  TF-IDF + Logistic Regression       76.72%     0.6962
  LSTM                               70.80%     0.6794

These are evaluation results recorded for this project's prepared
dataset and evaluation split; results can vary with data splits,
preprocessing, and environment. Prediction confidence displayed by the
app is not the same as model accuracy.

## Tech stack

-   Python
-   Streamlit
-   scikit-learn (TF-IDF and Logistic Regression)
-   TensorFlow / Keras (LSTM)
-   Hugging Face Datasets (source dataset)
-   `faster-whisper` (speech transcription)
-   FFmpeg / PyAV (audio/video media processing)
-   NumPy, Pandas, Joblib

## Project structure

``` text
Amazon-Sentiment-Analysis/
├── app.py
├── main.py
├── requirements.txt
├── packages.txt
├── data/
├── models/
│   ├── ml_model/
│   │   └── sentiment_pipeline.joblib
│   └── dl_model/
│       ├── sentiment_lstm.keras
│       ├── tokenizer.joblib
│       └── label_encoder.joblib
├── src/
│   ├── inspect_dataset.py
│   ├── prepare_dataset.py
│   ├── train_ml.py
│   ├── train_dl.py
│   ├── predict.py
│   └── compare_models.py
└── notebooks/
```

Your local folder names may differ slightly depending on the current
repository version.

## Run locally

### 1. Clone the repository

``` bash
git clone https://github.com/NihalMaredla/review-sentiment-analysis.git
cd review-sentiment-analysis
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
```

### 3. Install Python dependencies

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For local video analysis, install FFmpeg and ensure the folder
containing `ffmpeg.exe` is on your Windows `PATH`. Restart the terminal
after changing `PATH`.

### 4. Start the app

``` powershell
streamlit run app.py
```

Open the local URL printed in the terminal, usually
`http://localhost:8501`.

## How to use

1.  Select **Review Analyzer**, **Audio Analyzer**, or **Video
    Analyzer** in the sidebar.
2.  Choose **Logistic Regression** or **LSTM** as the prediction model.
3.  Enter a review or upload a media file supported by the uploader.
4.  Start analysis when prompted.
5.  Review the predicted sentiment, prediction confidence, and class
    probabilities.

For a first media test, use a short clip with clear speech.
Transcription can take longer on the first run while the Whisper model
is downloaded.

## Deployment notes

The app is deployed using Streamlit Community Cloud. `packages.txt` is
included for the system-level FFmpeg dependency required for video
processing in the cloud environment. Keep `requirements.txt` and
`packages.txt` in the repository root.

Live app:
https://review-sentiment-analysis-hvpuc3cc3dg2tpkauyovl.streamlit.app/

## Limitations

-   Sentiment is inferred from text, so sarcasm, context, mixed
    opinions, spelling errors, and transcription mistakes can affect
    predictions.
-   Audio/video predictions reflect the transcript only, not vocal tone,
    facial expression, or visual context.
-   A displayed confidence value is the model's estimated probability,
    not a guarantee that the prediction is correct.
-   Media processing time and memory use depend on file length, file
    format, and the hosting environment.

## Future improvements

-   Add downloadable prediction reports and session history.
-   Add timestamped, segment-level video transcript and sentiment
    results.
-   Evaluate on a separate external test set and document class-wise
    precision/recall.
-   Add automated tests for preprocessing, model loading, and
    media-processing errors.

## Author

**Nihal Maredla**\
GitHub: [@NihalMaredla](https://github.com/NihalMaredla)

------------------------------------------------------------------------

If you find the project useful, consider starring the repository.
