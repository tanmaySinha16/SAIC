import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('emotion_recognition_model.h5')

# Load the label-to-emotion mapping
index_to_label = np.load('label_mapping.npy', allow_pickle=True).item()

# Function to extract features from the audio file
def extract_features(audio_file):
    audio, sr = librosa.load(audio_file, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    if mfccs.shape[1] < 100:
        mfccs = np.pad(mfccs, ((0, 0), (0, 100 - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :100]
    return mfccs.T

# Extract features for a test audio file
audio_file = r"C:\Users\sinha\OneDrive\Desktop\Py\SAIC\dataset\angry\03-01-05-01-01-01-01.wav"
features = extract_features(audio_file)
features = np.expand_dims(features, axis=0)  # Reshape for model input

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.reshape(-1, features.shape[2])).reshape(features.shape)

# Predict emotion
predicted = model.predict(features_scaled)
predicted_class = np.argmax(predicted, axis=1)

# Get the emotion name from the mapping
predicted_emotion = index_to_label[predicted_class[0]]
print("Predicted emotion:", predicted_emotion)
