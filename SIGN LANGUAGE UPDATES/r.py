from flask import Flask, render_template, Response, request, jsonify
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
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["gesture"] + [f"p{i}" for i in range(63)])
    df.to_csv(CSV_FILE, index=False)

collecting_data = []
current_gesture = ""

def get_landmark_list(hand_landmarks):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    min_vals = np.min(landmarks, axis=0)
    max_vals = np.max(landmarks, axis=0)
    normalized = (landmarks - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized.flatten()

@app.route("/")
def index():
    return render_template("Dcopy.html")

@app.route("/start", methods=["POST"])
def start_collection():
    global collecting_data, current_gesture
    current_gesture = request.form["gesture"].strip()
    collecting_data = []
    return jsonify({"status": "started"})

@app.route("/collect_status")
def collect_status():
    def status_stream():
        global collecting_data
        while len(collecting_data) < 50:
            yield f"data: {len(collecting_data)}\n\n"
        yield "data: done\n\n"
    return Response(status_stream(), mimetype='text/event-stream')

@app.route("/finalize_data", methods=["POST"])
def finalize_data():
    global collecting_data, current_gesture
    data = request.get_json()
    if data.get('save'):
        df_existing = pd.read_csv(CSV_FILE)
        df_new = pd.DataFrame([[current_gesture] + d for d in collecting_data], columns=df_existing.columns)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(CSV_FILE, index=False)

        collecting_data = []
        return jsonify({"status": "success", "message": f"'{current_gesture}' saved!"})
    else:
        collecting_data = []
        return jsonify({"status": "success", "message": "Data discarded."})

@app.route("/video_feed")
def video_feed():
    def gen():
        global collecting_data, current_gesture
        cap = cv2.VideoCapture(0)
        sample_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hand_model.process(rgb)

            if results.multi_hand_landmarks and sample_count < 50:
                # Loop through all detected hands
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    vector = get_landmark_list(hand_landmarks)
                    collecting_data.append(vector.tolist())
                    sample_count += 1
                    if sample_count >= 50:  # Stop once 50 samples are collected
                        break

            cv2.putText(frame, f"Samples: {min(sample_count,50)}/50", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if sample_count >= 50:
                cap.release()
                break

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/train", methods=["POST"])
def train_model():
    if not os.path.exists(CSV_FILE):
        return jsonify({"status": "error", "message": "CSV not found."})

    df = pd.read_csv(CSV_FILE)
    if "gesture" not in df.columns:
        return jsonify({"status": "error", "message": "'gesture' column missing."})

    # Ensure 'gesture' labels are strings
    y = np.array(df["gesture"].values, dtype=str)

    X = df.drop(columns=["gesture"], errors='ignore').values

    if len(np.unique(y)) < 2:
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
