import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained model and label mapping
model = load_model('emotion_recognition_model.h5')
index_to_label = np.load('label_mapping.npy', allow_pickle=True).item()

# Function to extract features from audio
def extract_features(audio, sr=22050):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    if mfccs.shape[1] < 100:
        mfccs = np.pad(mfccs, ((0, 0), (0, 100 - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :100]
    return mfccs.T

# Function to record audio from the microphone
def record_audio(duration=3, sample_rate=22050):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until the recording is finished
    audio = audio.flatten()  # Flatten the audio array
    print("Recording complete.")
    return audio, sample_rate

# Function to predict emotion from audio
def predict_emotion():
    audio, sample_rate = record_audio()
    features = extract_features(audio, sample_rate)
    features = np.expand_dims(features, axis=0)  # Reshape for model input

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.reshape(-1, features.shape[2])).reshape(features.shape)

    # Predict emotion
    predicted = model.predict(features_scaled)
    predicted_class = np.argmax(predicted, axis=1)
    predicted_emotion = index_to_label[predicted_class[0]]
    print("Predicted emotion:", predicted_emotion)

# Run the real-time emotion prediction
predict_emotion()
