from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hand_model = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

CSV_FILE = "gesture_data.csv"

# Create CSV if not exists
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["gesture"] + [f"p{i}" for i in range(63)])
    df.to_csv(CSV_FILE, index=False)

def get_landmark_list(hand_landmarks):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    min_vals = np.min(landmarks, axis=0)
    max_vals = np.max(landmarks, axis=0)
    normalized = (landmarks - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized.flatten()

@app.route("/", methods=["GET"])
def index():
    return render_template("dashboard.html", collecting=False)

@app.route("/start", methods=["POST"])
def start_collection():
    gesture_name = request.form["gesture"].strip()
    cap = cv2.VideoCapture(0)
    collected_samples = 0
    data = []

    while collected_samples < 50:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_model.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                vector = get_landmark_list(hand_landmarks)
                data.append([gesture_name] + vector.tolist())
                collected_samples += 1
                print(f"Captured sample {collected_samples}/50")

        cv2.imshow("Collecting Gesture", frame)
        if cv2.waitKey(50) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    df_existing = pd.read_csv(CSV_FILE)
    df_new = pd.DataFrame(data, columns=df_existing.columns)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(CSV_FILE, index=False)

    return render_template("dashboard.html", status=f"✅ '{gesture_name}' captured!", collecting=False, show_train=True)

@app.route("/train", methods=["POST"])
def train_model():
    if not os.path.exists(CSV_FILE):
        return jsonify({"status": "error", "message": "CSV not found."})

    df = pd.read_csv(CSV_FILE)

    if "gesture" not in df.columns:
        return jsonify({"status": "error", "message": "'gesture' column missing."})

    X = df.drop(columns=["gesture"], errors='ignore').values
    y = df["gesture"].values
    unique_gestures = np.unique(y)

    if len(unique_gestures) < 2:
        return jsonify({"status": "error", "message": "At least 2 gestures required."})

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC(kernel="rbf", probability=True)
    model.fit(X_train, y_train)

    joblib.dump(model, "gesture_model.pkl")
    joblib.dump(encoder, "label_encoder.pkl")

    return jsonify({"status": "success", "message": "Model trained and saved!"})

if __name__ == "__main__":
    app.run(debug=True)
