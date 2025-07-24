# Final Streamlit App with All UI Enhancements

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from tensorflow.keras.models import load_model
import os
import base64

# Configuration
st.set_page_config(layout="wide")

# Load model and face cascade
model = load_model("model/model.keras")  # Ensure this was saved with keras_v3 format
face_cascade = cv2.CascadeClassifier("haarcascade.xml")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
log_path = "data/emotion_log.csv"
os.makedirs("data", exist_ok=True)

# Background blur function
def set_blurred_bg(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(f'''
        <style>
        .stApp {{
            background: linear-gradient(rgba(255,255,255,0.5), rgba(255,255,255,0.5)),
                        url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
    ''', unsafe_allow_html=True)

# Right-aligned login layout
def login_screen():
    st.markdown("""
    <style>
    .login-container {
        display: flex;
        height: 100vh;
        overflow: hidden;
    }
    .left {
        flex: 1;
        background-image: url('background.jpg');
        background-size: cover;
        background-position: center;
    }
    .right {
        flex: 1;
        background-color: #00274d;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .login-box {
        width: 80%;
        max-width: 400px;
        background-color: #00274d;
        padding: 2rem;
        color: white;
        border-radius: 10px;
        text-align: center;
    }
    .login-box input {
        margin-top: 10px;
        padding: 10px;
        border-radius: 5px;
        width: 100%;
        border: none;
    }
    .login-box button {
        margin-top: 15px;
        padding: 10px;
        background-color: #ffffff;
        color: #00274d;
        font-weight: bold;
        width: 100%;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    </style>
    <div class="login-container">
        <div class="left"></div>
        <div class="center">
            <div class="login-box">
                <h2>🔐 STUDENT LOGIN</h2>
                <form action="#">
                    <input id="name" name="name" type="text" placeholder="Enter your Name" class="stTextInput">
                    <input id="sid" name="sid" type="text" placeholder="Enter your ID" class="stTextInput">
                </form>
    """, unsafe_allow_html=True)

    login_btn = st.button("LOGIN")

    st.markdown("""
                <script>
                    const loginBox = document.querySelector('.login-box');
                    const inputs = loginBox.querySelectorAll('input');
                    inputs[0].value = window.parent.document.querySelector('[data-testid="stTextInput"] input[name=name_input]').value;
                    inputs[1].value = window.parent.document.querySelector('[data-testid="stTextInput"] input[name=sid_input]').value;
                </script>
            </div></div></div>
    """, unsafe_allow_html=True)

    if login_btn:
        if name.strip() and sid.strip():
            st.session_state.logged_in = True
            st.session_state.name = name
            st.session_state.sid = sid
            st.experimental_rerun()
        else:
            st.warning("Please enter both name and ID.")

# Navigation popup
def nav_bar():
    st.markdown("""
    <style>
    .nav-popup {
        text-align: center;
        margin-top: 10px;
    }
    .nav-selectbox label, .css-2trqyj, .css-1wa3eu0 {
        text-transform: uppercase;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("## MENU")
    option = st.selectbox("Select Page", ["Dashboard", "Data Log", "Logout"], index=0)
    return option

# Emotion Detection
def detect_emotion():
    st.subheader("📷 Capture Image")
    image = st.camera_input("Take a picture")

    if image:
        img = Image.open(image)
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48)) / 255.0
            roi = roi.reshape(1, 48, 48, 1)

            pred = model.predict(roi)
            emotion = emotion_labels[np.argmax(pred)]

            st.image(img_np, caption=f"Detected Emotion: {emotion}", use_column_width=True)

            if emotion in ['Angry', 'Sad', 'Disgust']:
                st.error(f"⚠️ Alert: {emotion} emotion detected.")

            timestamp = datetime.now().isoformat()
            entry = pd.DataFrame([[timestamp, st.session_state.name, st.session_state.sid, emotion]],
                                 columns=["Timestamp", "Name", "ID", "Emotion"])
            if os.path.exists(log_path):
                entry.to_csv(log_path, mode="a", header=False, index=False)
            else:
                entry.to_csv(log_path, index=False)
            break
        else:
            st.warning("No face detected.")

# Dashboard
def show_dashboard():
    st.subheader("📊 Emotion Dashboard")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        st.bar_chart(df['Emotion'].value_counts())

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        trend = df.groupby(df['Timestamp'].dt.date)['Emotion'].apply(lambda x: x.mode()[0])
        st.line_chart(trend.value_counts().sort_index())

        st.markdown("### 📌 Suggestions to Enhance Teaching:")
        st.info("""
        1. Track emotion trends to adjust teaching strategies.
        2. Identify frequent negative emotions to improve engagement.
        3. Use visual feedback to motivate students.
        """)
    else:
        st.warning("No data yet.")

# Data Log
def show_log():
    st.subheader("📄 Logged Emotion Data")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        st.dataframe(df)
    else:
        st.info("No logs available.")

# Main logic
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_screen()
else:
    set_blurred_bg("background.jpg")
    page = nav_bar()
    with st.container():
        if page == "Dashboard":
            detect_emotion()
            show_dashboard()
        elif page == "Data Log":
            show_log()
        elif page == "Logout":
            st.session_state.logged_in = False
            st.experimental_rerun()
