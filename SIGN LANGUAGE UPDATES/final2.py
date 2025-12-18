from flask import Flask, render_template, Response, jsonify,request,redirect, url_for
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import joblib
import pygame
from gtts import gTTS
import os
from googletrans import Translator
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import signal
import threading
import time
import shutil
import datetime


app = Flask(__name__)

# Load trained SVM model and encoder
model = joblib.load("gesture_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hand_model = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

# Initialize translator
translator = Translator()

# Stability Parameters
stable_gesture = None
stable_count = 0
stable_frames_threshold = 10  # Number of consistent frames needed to confirm a gesture
confirmed_gesture = None  # Store the final confirmed gesture

# Initialize pygame mixer
pygame.mixer.init()

CSV_FILE = "gesture_data.csv"
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["gesture"] + [f"p{i}" for i in range(63)])
    df.to_csv(CSV_FILE, index=False)

# Function to extract and normalize landmarks
def get_landmark_list(hand_landmarks):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    min_vals = np.min(landmarks, axis=0)
    max_vals = np.max(landmarks, axis=0)
    normalized_landmarks = (landmarks - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized_landmarks.flatten()


# Function to translate text to Tamil and speak
def speak_in_tamil(text):
    audio_path = "output.mp3"
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        pygame.mixer.quit()
        pygame.mixer.init()
        if os.path.exists(audio_path):
            os.remove(audio_path)

        tts = gTTS(text=text, lang='ta')
        tts.save(audio_path)
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            continue
    except Exception as e:
        print("Error in gTTS:", e)
        
        
# Video capture function
def generate_frames():
    global stable_gesture, stable_count, confirmed_gesture
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_model.process(rgb_frame)

        detected_gesture = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmark_vector = get_landmark_list(hand_landmarks).reshape(1, -1)

                prediction = model.predict(landmark_vector)[0]
                gesture_name = encoder.inverse_transform([prediction])[0]

                if gesture_name == stable_gesture:
                    stable_count += 1
                else:
                    stable_gesture = gesture_name
                    stable_count = 0

                if stable_count >= stable_frames_threshold:
                    if confirmed_gesture != stable_gesture:
                        confirmed_gesture = stable_gesture
                        detected_gesture = confirmed_gesture
                        speak_in_tamil(gesture_name)

                    cv2.putText(frame, f"Gesture: {gesture_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Waiting for stable action...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/restart')
# def restart():
#     # Save the currently running script before restarting
#     current_file = __file__
#     backup_file = current_file.replace(".py", "_backup.py")

#     try:
#         shutil.copy(current_file, backup_file)
#         print(f"✅ Current file saved as backup: {backup_file}")
#     except Exception as e:
#         print(f"❌ Error saving file: {e}")

#     shutdown_func = request.environ.get('werkzeug.server.shutdown')

#     def shutdown():
#         time.sleep(1)
#         if shutdown_func:
#             shutdown_func()

#     threading.Thread(target=shutdown).start()
#     return redirect(url_for('index'))  # Or render_template('index.html')



@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/dashboard", methods=["GET"])
def dashboard():
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



    
    
@app.route('/get_gesture')
def get_gesture():
    global confirmed_gesture
    is_speaking = pygame.mixer.music.get_busy()
    return jsonify({
        "gesture": confirmed_gesture,
        "speaking": is_speaking
    })



if __name__ == '__main__':
    app.run(debug=True, port=5001)
