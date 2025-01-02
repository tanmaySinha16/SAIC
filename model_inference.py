import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('emotion_recognition_model.h5')

# Load the label-to-emotion mapping
index_to_label = np.load('label_mapping.npy', allow_pickle=True).item()

# Ensure index-to-label mapping reflects the correct emotion names
emotion_map = { 
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
}

# Update index_to_label mapping
index_to_label = emotion_map

# Function to extract features from the audio file
def extract_features(audio_file):
    audio, sr = librosa.load(audio_file, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Pad or trim to ensure the same feature size (100 timesteps)
    if mfccs.shape[1] < 100:
        mfccs = np.pad(mfccs, ((0, 0), (0, 100 - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :100]
        
    return mfccs.T  # Transpose to (time_steps, features)

# Extract features for a test audio file
audio_file = r"C:\Users\sinha\OneDrive\Desktop\Py\SAIC\dataset\angry\03-01-05-01-01-01-01.wav"
features = extract_features(audio_file)
features = np.expand_dims(features, axis=0)  # Reshape to (1, time_steps, features)

# Load a pre-fitted scaler (or fit a scaler to X_train during training)
scaler = StandardScaler()
scaler.fit(features.reshape(-1, features.shape[2]))  # Fit to features of the current input
features_scaled = scaler.transform(features.reshape(-1, features.shape[2])).reshape(features.shape)

# Predict emotion
predicted = model.predict(features_scaled)
predicted_class = np.argmax(predicted, axis=1)

# Get the emotion name from the updated mapping
predicted_emotion = index_to_label[predicted_class[0]]
print(f"Predicted class: {predicted_class[0]}")
print(f"Predicted emotion: {predicted_emotion}")
