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
   st.markdown("""
    <h2 style='text-align: center; color: #006d77; font-weight: bold; margin-top: 30px;'>
        🔐 STUDENT LOGIN
    </h2>
""", unsafe_allow_html=True)

    name = st.text_input("Enter your Name")
    sid = st.text_input("Enter your ID")
    login_btn = st.button("LOGIN")

    if login_btn:
        if name.strip() and sid.strip():
            st.session_state.logged_in = True
            st.session_state.name = name
            st.session_state.sid = sid
            st.session_state.page = "Emotion Capture"
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
        justify-content: right;
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
        <div style='position: absolute; left: 35px; color: black; font-weight: bold; font-size: 35px;'>VIRTUAL EMODASH</div>
        <a href="#" onclick="window.location.search='?nav=Emotion Capture'">Emotion Capture</a>
        <a href="#" onclick="window.location.search='?nav=Dashboard'">Dashboard</a>
        <a href="#" onclick="window.location.search='?nav=Data Log'">Data Log</a>
        <a href="#" onclick="window.location.search='?nav=Logout'">Logout</a>
    </div>
    """, unsafe_allow_html=True)

    query_params = st.experimental_get_query_params()
    nav_page = query_params.get("nav", [st.session_state.get("page", "Emotion Capture")])[0]
    st.session_state.page = nav_page
    return nav_page

# Emotion Detection

def detect_emotion():
    st.subheader("📷 Capture Image")
    col1, col2 = st.columns([1, 2])

    with col1: 
        image = st.camera_input("Take a picture")
       

    with col2:
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

                st.image(img_np, use_column_width=True)
                st.markdown(f"""
                    <h1 style='text-align:center; color:red; animation: popIn 1s ease-in-out;'>DETECTED EMOTION: {emotion.upper()}</h1>
                    <style>
                    @keyframes popIn {{
                        0% {{ transform: scale(0.8); opacity: 0; }}
                        100% {{ transform: scale(1); opacity: 1; }}
                    }}
                    </style>
                """, unsafe_allow_html=True)
               


# Main logic
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_screen()
else:
    set_plain_bg("background.png")
    page = nav_bar()
    st.markdown(f"""<h3 style='text-align:center; color:#006d77;'>👋 Welcome, {st.session_state.name}!</h3>""", unsafe_allow_html=True)

    with st.container():
        if page == "Emotion Capture":
            detect_emotion()
        elif page == "Dashboard":
            show_dashboard()
        elif page == "Data Log":
            show_log()
        elif page == "Logout":
            st.session_state.logged_in = False
            st.session_state.page = "Emotion Capture"
            st.experimental_set_query_params()
            st.experimental_rerun()
