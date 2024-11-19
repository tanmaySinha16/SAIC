import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

# Define the path to your organized dataset
dataset_folder = r"C:\Users\sinha\OneDrive\Desktop\Py\SAIC\dataset"

# Prepare lists to hold features and labels
features = []
labels = []

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
                    labels.append(emotion)
                
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Create a mapping of emotion labels to indices
unique_emotions = np.unique(labels)
label_to_index = {emotion: index for index, emotion in enumerate(unique_emotions)}
index_to_label = {index: emotion for emotion, index in label_to_index.items()}

# Convert labels to numerical indices
numerical_labels = np.array([label_to_index[label] for label in labels])

# Save features, numerical labels, and the mapping for future use
np.save('features.npy', features)
np.save('numerical_labels.npy', numerical_labels)
np.save('label_mapping.npy', index_to_label)  # Save the mapping as well

print("Feature extraction completed successfully!")
