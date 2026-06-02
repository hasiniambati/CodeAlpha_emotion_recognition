<img width="1920" height="1140" alt="Home_page" src="https://github.com/user-attachments/assets/132400f4-ac9b-4308-b5a0-7b3001d6a2c4" />

# 🎤 Speech Emotion Recognition System

## Overview

This project is a Machine Learning-based Speech Emotion Recognition System that predicts human emotions from speech audio files using MFCC feature extraction and an SVM classifier.

Supported emotions:

* Angry
* Calm
* Disgust
* Fearful
* Happy
* Neutral
* Sad
* Surprised

---

## Features

* Upload WAV audio files
* Automatic MFCC feature extraction
* Emotion prediction using Support Vector Machine (SVM)
* Interactive Streamlit web application
* Real-time emotion classification

---

## Dataset

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

Total Samples: 1440

Emotions:

* Neutral
* Calm
* Happy
* Sad
* Angry
* Fearful
* Disgust
* Surprised

---

## Technologies Used

* Python
* Streamlit
* Scikit-Learn
* Librosa
* NumPy
* Pandas

---

## Model Performance

### Random Forest

Accuracy: 57.99%

### SVM

Accuracy: 68.40%

SVM was selected as the final model due to better performance.

---

## Project Structure

```text
emotion_recognition/
│
├── app.py
├── build_dataset.py
├── train.py
├── train_svm.py
├── test_audio.py
│
├── emotion_dataset.csv
├── requirements.txt
├── README.md
│
├── models/
│   ├── emotion_model.pkl
│   ├── emotion_model_svm.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── dataset/
│   ├── Actor_01/
│   ├── Actor_02/
│   ├── ...
│   └── Actor_24/
│
├── screenshots/
│   ├── home_page.png
│   ├── prediction_happy.png
│   ├── prediction_sad.png
│   └── prediction_disgust.png
│
└── .gitignore
---

## Installation

Clone the repository:

```bash
git clone https://github.com/hasiniambati23/CodeAlpha_emotion_recognition.git
cd CodeAlpha_emotion_recognition
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Application

```bash
streamlit run emotion_app.py
```

---

## Screenshots

### Home Page
<img width="1920" height="1140" alt="Home_page" src="https://github.com/user-attachments/assets/431479e0-4347-4859-ac70-1c53a3843b89" />

### Prediction_disgust
<img width="1920" height="1140" alt="prediction_disgust" src="https://github.com/user-attachments/assets/11216310-b45b-4c19-b107-0fd23a4b7401" />

### Prediction_Happy
<img width="1920" height="1140" alt="prediction_happy" src="https://github.com/user-attachments/assets/2d3e3891-2282-458e-a124-a93bab982c01" />

### Prediction_Sad
<img width="1920" height="1140" alt="prediction_sad" src="https://github.com/user-attachments/assets/3cf9a6ee-ea72-4c59-bc19-bcfdf9d6320e" />

---
## Live Demo
[Open Application]()

## Author

Hasini Reddy

Machine Learning Internship Project
