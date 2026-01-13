import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from collections import deque
from dotenv import load_dotenv
import os


load_dotenv()  # loads .env into environment variables


# -------------------------
# Load Fine-tuned ResNet50 Model (.keras)
# -------------------------
MODEL_TYPE = "vgg"   # "resnet" | "vgg" | "efficientnet"
MODEL_PATH = "vgg19_best.keras"

if MODEL_TYPE == "resnet":
    from tensorflow.keras.applications.resnet50 import preprocess_input
elif MODEL_TYPE == "vgg":
    from tensorflow.keras.applications.vgg19 import preprocess_input
elif MODEL_TYPE == "efficientnet":
    from tensorflow.keras.applications.efficientnet import preprocess_input

model = load_model(MODEL_PATH)

emotion_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

# -------------------------
# Spotify Authentication
# -------------------------
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    redirect_uri="http://127.0.0.1:8000/callback",
    scope="user-modify-playback-state user-read-playback-state"
))

# Emotion → Playlist Mapping
emotion_to_playlist = {
    "Angry": "37i9dQZF1DX76Wlfdnj7AP",   # Rock Hard
    "Sad": "37i9dQZF1DWVrtsSlLKzro",     # Sad Songs
    "Happy": "37i9dQZF1DXdPec7aLTmlC",   # Happy Hits
    "Surprise": "37i9dQZF1DX2SK4ytI2KAZ",# Fresh Finds
    "Neutral": "37i9dQZF1DXcCnTAt8CfNe"  # Chill Lofi Beats
}

def play_music(emotion):
    playlist_id = emotion_to_playlist.get(emotion)
    if playlist_id:
        devices = sp.devices()
        if devices['devices']:
            device_id = devices['devices'][0]['id']  # use first device
            sp.start_playback(device_id=device_id, context_uri=f"spotify:playlist:{playlist_id}")
            print(f"🎵 Playing {emotion} playlist!")
        else:
            print("No active Spotify device found. Please open Spotify app.")

# -------------------------
# OpenCV Webcam Feed
# -------------------------
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- Emotion Smoothing ---
emotion_history = deque(maxlen=5)
last_played_emotion = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]   # keep ROI in color
        roi_color = cv2.resize(roi_color, (224, 224))  # ResNet50 input size
        roi_array = np.expand_dims(roi_color, axis=0)
        roi_array = preprocess_input(roi_array)  # preprocess for ResNet50

        # Predict emotion
        prediction = model.predict(roi_array, verbose=0)[0]
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]
        confidence = prediction[emotion_idx]

        # Draw rectangle and emotion text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # --- Smoothing: Majority Voting ---
        if confidence > 0.6:
            emotion_history.append(emotion)

            if len(emotion_history) == emotion_history.maxlen:
                most_common = max(set(emotion_history), key=emotion_history.count)

                if most_common != last_played_emotion:
                    play_music(most_common)
                    last_played_emotion = most_common

    cv2.imshow("Emotion + Spotify", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


