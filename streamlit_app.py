import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import io

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
        margin-bottom: 50px;
    }
    .upload-section {
        text-align: center;
        margin-bottom: 1px;
        color: blue;
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
        margin-top: 5px;
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
    <p>Upload an audio file in .wav format to get started. The prediction process may take a few moments, so please be patient.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Create a file uploader widget
st.markdown('<div class="upload-section"><h3>Upload your audio file here:</h3></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["wav"], help="Accepted file formats: .wav.")

# Prediction button and result display
if uploaded_file is not None:
    st.write("Uploaded file:", uploaded_file.name)
    
    # Play the uploaded audio file
    st.audio(uploaded_file.read(), format='audio/wav')
    
    # Reset the file pointer to the beginning after playing
    uploaded_file.seek(0)
    
    # Create a button to trigger the prediction
    if st.button("Predict Emotion"):
        with st.spinner('Predicting...Please be patient...'):
            # Read the file bytes
            audio_bytes = uploaded_file.read()
            
            # Predict the emotion
            emotion = predict_emotion(audio_bytes)
            emotion = emotion.capitalize()
            st.markdown(
                f'<h3>The Result is:</h3>'
                f'<div class="prediction-section"><h3>Predicted Emotion is {emotion}</h3></div>',
                unsafe_allow_html=True
            )
else:
    st.info("Please upload an audio file to get started.")
