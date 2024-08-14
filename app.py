import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
from audio_recorder_streamlit import audio_recorder

# Load the pre-trained model
model = load_model('emotionclassifier.h5')
emotions = ['angry', 'fear', 'happy', 'neutral', 'sad']

def predict_emotion(audio_bytes):
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    x = np.expand_dims([mfccs_mean], -1)
    prediction = model.predict(x)
    predicted_emotion = emotions[prediction.argmax(axis=1)[0]]
    return predicted_emotion

# Set page configuration
st.set_page_config(page_title="Emotion Prediction from Tamil Audio", page_icon="🎵", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: black;
        font-family: Arial, sans-serif;
    }
    .header {
        text-align: center;
        margin-top: 5px;
        margin-bottom: 10px;
    }
    .upload-section {
        text-align: center;
        margin-bottom: 1px;
        color: blue;
    }
    .recording-section {
        text-align: center;
        margin-top: 20px;
        background-color: #333;
        padding: 15px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
    }
    .prediction-section {
        text-align: center;
        font-size: 1.2em;
        margin-top: 30px;
        background-color: #4CAF50;
        padding: 10px;
        border-radius: 40px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    .instruction {
        font-size: 1em;
        margin-bottom: 20px;
        text-align: center;
    }
    .description {
        text-align: center;
        font-size: 1em;
        margin-top: 1px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        transition: background-color 0.3s, color 0.3s;
    }
    .stButton>button:hover {
        background-color: green;
        color: white;
        border: white;
    }
    /* Enhanced audio recorder button styling */
    .stAudioRecorder button {
        background-color: #FF5722;
        color: white;
        border: none;
        padding: 15px 40px;
        font-size: 1.2em;
        border-radius: 50px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        display: inline-block;
        position: relative;
        overflow: hidden;
    }
    .stAudioRecorder button:hover {
        background-color: #E64A19;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .stAudioRecorder button:active {
        transform: scale(0.95);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stAudioRecorder button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 300%;
        height: 300%;
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 50%;
        transform: translate(-50%, -50%) scale(0);
        transition: transform 0.5s ease;
    }
    .stAudioRecorder button:hover::before {
        transform: translate(-50%, -50%) scale(1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App header
st.markdown('<div class="header"><h1>🎵 Emotion Prediction from Tamil Audio</h1></div>', unsafe_allow_html=True)

# Project description
st.markdown(
    """
    <div class="description">
    <p>You can either record your audio or upload an audio file to get started. The prediction process may take a few moments, so please be patient.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Recording section
st.markdown('<div class="recording-section"><h3>Record your audio here:</h3><h6>(Make sure that you should record the audio in a quiet place to get accurate predictions.)</h6></div>', unsafe_allow_html=True)
audio_bytes = audio_recorder()

# File upload section
st.markdown('<div class="upload-section"><h3>Or upload your audio file here:</h3></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["wav"], help="Accepted file formats: .wav.")

# Initialize variables to track prediction state
prediction_made = False

# Handle audio prediction
if audio_bytes is not None and not prediction_made:
    st.audio(audio_bytes, format='audio/wav')
    if st.button("Predict Emotion"):
        with st.spinner('Predicting... Please be patient...'):
            emotion = predict_emotion(audio_bytes)
            emotion = emotion.capitalize()
            st.markdown(
                f'<h3>The Result is:</h3>'
                f'<div class="prediction-section"><h3>Predicted Emotion is {emotion}</h3></div>',
                unsafe_allow_html=True
            )
        # Clear the recorded audio after prediction
        audio_bytes = None
        prediction_made = True

elif uploaded_file is not None and not prediction_made:
    st.write("Uploaded file:", uploaded_file.name)
    st.audio(uploaded_file.read(), format='audio/wav')
    uploaded_file.seek(0)
    if st.button("Predict Emotion"):
        with st.spinner('Predicting... Please be patient...'):
            audio_bytes = uploaded_file.read()
            emotion = predict_emotion(audio_bytes)
            emotion = emotion.capitalize()
            st.markdown(
                f'<h3>The Result is:</h3>'
                f'<div class="prediction-section"><h3>Predicted Emotion is {emotion}</h></div>',
                unsafe_allow_html=True
            )
        # Clear the uploaded file after prediction
        uploaded_file = None
        prediction_made = True

else:
    if not prediction_made:
        st.info("Please record your audio or upload a file to get started.")
