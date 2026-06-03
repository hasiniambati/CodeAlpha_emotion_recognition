import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_path = "dataset/Actor_01/03-01-01-01-01-01-01.wav"

audio, sr = librosa.load(audio_path)

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=40
)

plt.figure(figsize=(10,4))

librosa.display.specshow(
    mfcc,
    x_axis="time"
)

plt.colorbar()
plt.title("MFCC Features")

plt.tight_layout()
plt.show()