import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Check if dataset exists
file_path = "gesture_data.csv"
if not os.path.exists(file_path):
    print("❌ Error: 'gesture_data.csv' not found. Please collect gesture data first.")
    exit()

# Load dataset
df = pd.read_csv(file_path)

# Ensure 'gesture' column exists
if "gesture" not in df.columns:
    print("❌ Error: 'gesture' column missing in dataset. Ensure the data is correctly formatted.")
    exit()

# Prepare data
X = df.drop(columns=["gesture"], errors='ignore').values
y = df["gesture"].values

# Check for minimum required gestures
unique_gestures = np.unique(y)
if len(unique_gestures) < 2:
    print(f"❌ Error: Found only {len(unique_gestures)} unique gesture(s). SVM requires at least 2.")
    print("📌 Solution: Collect more gesture samples before training.")
    exit()

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

# Save model and encoder
joblib.dump(model, "gesture_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("✅ Model trained and saved successfully!")