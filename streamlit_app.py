import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import os
import pandas as pd
from tensorflow.keras.models import load_model
import base64

# ======== BACKGROUND IMAGE SETUP ==========
def set_bg_image(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

set_bg_image("background.jpg")

# ======== LOGIN BOX CSS ==========
st.markdown("""
    <style>
    .login-box {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 10px;
        max-width: 400px;
        margin: auto;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ======== SESSION STATE SETUP ==========
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "student_name" not in st.session_state:
    st.session_state.student_name = ""
if "student_id" not in st.session_state:
    st.session_state.student_id = ""

# ======== LOAD MODELS & SETUP ========
model = load_model("model/model.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade = cv2.CascadeClassifier("haarcascade.xml")
os.makedirs("data/captured", exist_ok=True)
log_path = "data/emotion_log.csv"

# ======== LOGIN PAGE ==========
def login_page():
    st.title("LOGIN")
    st.markdown("<br>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="login-box">', unsafe_allow_html=True)

        name = st.text_input("Enter your Name")
        sid = st.text_input("Enter your ID")

        if st.button("Login"):
            if name.strip() == "" or sid.strip() == "":
                st.warning("Please enter both name and ID.")
            else:
                st.session_state.student_name = name
                st.session_state.student_id = sid
                st.session_state.logged_in = True
                st.success("Login successful!")

        st.markdown('</div>', unsafe_allow_html=True)

# ======== LIVE VIDEO EMOTION DETECTION ==========
def capture_video_emotions():
    st.subheader("🎥 Live Emotion Detection")
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])

    cap = None

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi, (48, 48)) / 255.0
                roi_reshaped = roi_resized.reshape(1, 48, 48, 1)

                prediction = model.predict(roi_reshaped)
                emotion = emotion_labels[np.argmax(prediction)]

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Log data (one entry per detection session)
                timestamp = datetime.now().isoformat()
                log_entry = pd.DataFrame([[timestamp, st.session_state.student_name, st.session_state.student_id, emotion]],
                                          columns=["Timestamp", "Name", "ID", "Emotion"])
                if os.path.exists(log_path):
                    log_entry.to_csv(log_path, mode="a", header=False, index=False)
                else:
                    log_entry.to_csv(log_path, index=False)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

        cap.release()

# ======== MAIN PAGE ==========
def main_page():
    st.title(f"Welcome, {st.session_state.student_name} 👋")
    capture_video_emotions()
    if st.button("Logout"):
        st.session_state.logged_in = False

# ======== ROUTING ==========
if not st.session_state.logged_in:
    login_page()
else:
    main_page()




