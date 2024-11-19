import os
import shutil

# Define the source and destination paths
source_folder = r"C:\Users\sinha\Downloads\archive (1)\audio_speech_actors_01-24"  
destination_folder = r"C:\Users\sinha\OneDrive\Desktop\Py\SAIC\dataset"  

# Mapping of emotion codes to emotion names
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Create folders for each emotion
for emotion in emotion_map.values():
    os.makedirs(os.path.join(destination_folder, emotion), exist_ok=True)

# Iterate through each actor's folder
for actor_folder in os.listdir(source_folder):
    actor_path = os.path.join(source_folder, actor_folder)
    
    # Check if the path is a directory (actor folder)
    if os.path.isdir(actor_path):
        # Loop through each .wav file in the actor's folder
        for file_name in os.listdir(actor_path):
            if file_name.endswith(".wav"):  # Check if the file is a .wav file
                parts = file_name.split("-")
                emotion_code = parts[2]  # Extract the emotion code from the filename
                emotion_folder = emotion_map.get(emotion_code, "unknown")
                
                if emotion_folder != "unknown":
                    # Move the file to the corresponding emotion folder
                    shutil.move(os.path.join(actor_path, file_name), os.path.join(destination_folder, emotion_folder, file_name))

print("Files organized successfully!")
