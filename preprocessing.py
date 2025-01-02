import os
import librosa
import numpy as np

# Define the path to your organized dataset (six classes)
dataset_folder = r"C:\Users\sinha\OneDrive\Desktop\Py - Copy\SAIC\dataset"

# Prepare lists to hold features and labels
features = []
labels = []

# Mapping of emotion labels to numeric indices (6 classes)
emotion_map = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
}

# Loop through each emotion folder and extract features
for emotion in os.listdir(dataset_folder):
    emotion_folder = os.path.join(dataset_folder, emotion)
    
    if os.path.isdir(emotion_folder):
        for file_name in os.listdir(emotion_folder):
            if file_name.endswith(".wav"):
                file_path = os.path.join(emotion_folder, file_name)
                
                try:
                    # Load the audio file
                    audio, sample_rate = librosa.load(file_path, sr=22050)  # Standard sampling rate
                    
                    # Extract MFCC features
                    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
                    if mfcc.shape[1] < 100:
                        mfcc = np.pad(mfcc, ((0, 0), (0, 100 - mfcc.shape[1])), mode='constant')
                    else:
                        mfcc = mfcc[:, :100]
                    features.append(mfcc.T)
                    labels.append(emotion_map.get(emotion, -1))  # Use -1 for invalid emotion (if any)
                
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Check unique labels in y_train to ensure they are valid
print("Unique labels in labels:", np.unique(labels))

# --- Remove invalid labels (those marked as -1) ---
valid_indices = labels != -1  # Get indices where the label is valid
features = features[valid_indices]  # Filter features
labels = labels[valid_indices]  # Filter labels

# --- Split Data into Train and Test (70% Train, 30% Test) ---
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Reshape the data for LSTM input (samples, time_steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# --- Save the Data ---
np.save('X_train_reshaped.npy', X_train)
np.save('X_test_reshaped.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("Feature extraction completed successfully!")
