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
model = load_model("model/model.keras")
face_cascade = cv2.CascadeClassifier("haarcascade.xml")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
log_path = "data/emotion_log.csv"
os.makedirs("data", exist_ok=True)

# Normal background image (non-blurred)
def set_plain_bg(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(f'''
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
    ''', unsafe_allow_html=True)

# Login screen

def login_screen():
    set_plain_bg("background.png")
    st.markdown("### 🔐 STUDENT LOGIN", unsafe_allow_html=True)
    name = st.text_input("Enter your Name")
    sid = st.text_input("Enter your ID")
    login_btn = st.button("LOGIN")

    if login_btn:
        if name.strip() and sid.strip():
            st.session_state.logged_in = True
            st.session_state.name = name
            st.session_state.sid = sid
            st.experimental_rerun()
        else:
            st.warning("Please enter both name and ID.")

# Navigation taskbar

def nav_bar():
    st.markdown("""
    <style>
    .navbar {
        background-color: #006d77;
        overflow: hidden;
        display: flex;
        justify-content: center;
        padding: 10px 0;
    }
    .navbar a {
        text-decoration: none;
        color: white;
        padding: 12px 20px;
        text-align: center;
        font-weight: bold;
        text-transform: uppercase;
    }
    .navbar a:hover {
        background-color: #004c52;
        border-radius: 5px;
    }
    </style>
    <div class="navbar">
        <a href="/?nav=Emotion Capture">Emotion Capture</a>
        <a href="/?nav=Dashboard">Dashboard</a>
        <a href="/?nav=Data Log">Data Log</a>
        <a href="/?nav=Logout">Logout</a>
    </div>
    """, unsafe_allow_html=True)

    query_params = st.experimental_get_query_params()
    nav_page = query_params.get("nav", ["Emotion Capture"])[0]
    return nav_page

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
    set_plain_bg("background.png")
    page = nav_bar()
    with st.container():
        if page == "Emotion Capture":
            detect_emotion()
        elif page == "Dashboard":
            show_dashboard()
        elif page == "Data Log":
            show_log()
        elif page == "Logout":
            st.session_state.logged_in = False
            st.experimental_rerun()
