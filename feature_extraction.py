import os
import librosa
import numpy as np
import pandas as pd

# Define the path to your organized dataset
dataset_folder = r"C:\Users\sinha\OneDrive\Desktop\Py\SAIC\dataset"  # Replace with your organized data path

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
                    audio, sample_rate = librosa.load(file_path, sr=None)
                    
                    # Extract MFCC features
                    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
                    mfcc_mean = np.mean(mfcc.T, axis=0)  # Take the mean of the MFCC features
                    
                    # Append features and corresponding emotion label
                    features.append(mfcc_mean)
                    labels.append(emotion)
                
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)
print(labels)
# Save features and labels for future use
np.save('features.npy', features)
np.save('labels.npy', labels)

print("Feature extraction completed successfully!")
