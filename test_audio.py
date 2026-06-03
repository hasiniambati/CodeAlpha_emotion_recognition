import librosa

audio_path = "dataset/Actor_01/03-01-01-01-01-01-01.wav"

audio, sample_rate = librosa.load(audio_path)

print("Sample Rate:", sample_rate)
print("Audio Length:", len(audio))
print("First 10 Samples:")
print(audio[:10])