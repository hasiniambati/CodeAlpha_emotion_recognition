import streamlit as st
import librosa
import numpy as np
import joblib

# Page Config
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)

# Load Models
model = joblib.load("models/emotion_model_svm.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# Title
st.title("🎤 Speech Emotion Recognition System")

st.markdown("""
Upload a speech audio file (.wav) and the model will predict the emotion.
""")

st.divider()

# File Upload
uploaded_file = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    try:

        # Load Audio
        audio, sr = librosa.load(
            uploaded_file,
            sr=None
        )

        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )

        mfcc_scaled = np.mean(
            mfcc.T,
            axis=0
        )

        features = mfcc_scaled.reshape(1, -1)

        # Scale Features
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)

        emotion = encoder.inverse_transform(prediction)[0]

        # Confidence
        if hasattr(model, "decision_function"):

            scores = model.decision_function(features_scaled)

            confidence = np.max(scores)

        else:
            confidence = 0

        st.success(
            f"Predicted Emotion: {emotion.upper()}"
        )

        st.info(
            f"Confidence Score: {confidence:.2f}"
        )

    except Exception as e:

        st.error(
            f"Error processing audio: {e}"
        )

st.divider()

st.subheader("Supported Emotions")

st.write("""
😀 Happy

😢 Sad

😠 Angry

😨 Fearful

😐 Neutral

😌 Calm

🤢 Disgust

😲 Surprised
""")

st.divider()

st.caption(
    "Built using Librosa, Scikit-Learn, SVM and Streamlit"
)