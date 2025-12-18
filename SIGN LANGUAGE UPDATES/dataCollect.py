import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
hand_model = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

# Function to extract and normalize landmarks
def get_landmark_list(hand_landmarks):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    min_vals = np.min(landmarks, axis=0)
    max_vals = np.max(landmarks, axis=0)
    
    # Normalize landmarks
    normalized_landmarks = (landmarks - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized_landmarks.flatten()

# Check if dataset exists, else create it
csv_file = "gesture_data.csv"
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=["gesture"] + [f"p{i}" for i in range(63)])
    df.to_csv(csv_file, index=False)

gesture_name = input("Enter the gesture name: ").strip()
print("Collecting 50 samples automatically... Please hold your gesture steady!")

landmark_data = []
collected_samples = 0

while collected_samples < 50:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_model.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmark_vector = get_landmark_list(hand_landmarks)

            # Store data
            landmark_data.append([gesture_name] + landmark_vector.tolist())
            collected_samples += 1
            print(f"📸 Sample {collected_samples}/50 captured!")

    cv2.imshow("Gesture Training", frame)
    cv2.waitKey(50)  # Slight delay

# Save to CSV
df = pd.read_csv(csv_file)
df_new = pd.DataFrame(landmark_data, columns=df.columns)
df = pd.concat([df, df_new], ignore_index=True)
df.to_csv(csv_file, index=False)

print(f"✅ Gesture '{gesture_name}' successfully saved with 50 samples!")
cap.release()
cv2.destroyAllWindows()
