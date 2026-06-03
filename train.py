import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("emotion_dataset.csv")

print("Dataset Shape:", df.shape)

# Features and target
X = df.drop("emotion", axis=1)
y = df["emotion"]

# Encode emotions
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)

# Save model
joblib.dump(model, "emotion_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("\nModel Saved Successfully!")