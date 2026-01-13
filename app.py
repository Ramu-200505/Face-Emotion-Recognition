import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import preprocess_input

# --------------------
# CONFIG
# --------------------
MODEL_PATH = "vgg19_best.keras"
emotion_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

st.set_page_config(page_title="Emotion Music Recommender")
st.title("🎵 Emotion-Based Music Recommendation")

# --------------------
# Load Model
# --------------------
@st.cache_resource
def load_emotion_model():
    return load_model(MODEL_PATH)

model = load_emotion_model()

# --------------------
# Face Detector
# --------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --------------------
# UI Controls
# --------------------
run = st.checkbox("Start Camera")
frame_placeholder = st.empty()

cap = None

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not accessible")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224,224))
            face = np.expand_dims(face, axis=0)
            face = preprocess_input(face)

            pred = model.predict(face, verbose=0)
            emotion = emotion_labels[np.argmax(pred)]

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        frame_placeholder.image(frame, channels="BGR")

else:
    if cap:
        cap.release()
