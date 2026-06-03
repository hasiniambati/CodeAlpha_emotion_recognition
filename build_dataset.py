import os
import librosa
import numpy as np
import pandas as pd

# Emotion mapping
emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

features = []
emotions = []

dataset_path = "dataset"

for actor in os.listdir(dataset_path):

    actor_path = os.path.join(dataset_path, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if file.endswith(".wav"):

            file_path = os.path.join(actor_path, file)

            # Emotion code is third field
            emotion_code = file.split("-")[2]

            emotion = emotion_dict[emotion_code]

            try:
                audio, sr = librosa.load(file_path)

                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=sr,
                    n_mfcc=40
                )

                mfcc_scaled = np.mean(mfcc.T, axis=0)

                features.append(mfcc_scaled)
                emotions.append(emotion)

            except Exception as e:
                print("Error:", file_path)
                print(e)

# Create dataframe
X = pd.DataFrame(features)

X["emotion"] = emotions

print(X.head())
print("\nShape:", X.shape)

# Save dataset
X.to_csv("emotion_dataset.csv", index=False)

print("\nDataset saved successfully!")