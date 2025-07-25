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
        <h2 style='text-align: center; color: #006d77;'>
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
            st.experimental_set_query_params(page="Emotion Capture")
            st.experimental_rerun()
        else:
            st.warning("Please enter both name and ID.")
# Navigation taskbar

def nav_bar():
    selected = st.session_state.get("page", "Emotion Capture")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    with col1:
        if st.button("EMOTION CAPTURE"):
            st.session_state.page = "Emotion Capture"
    with col2:
        if st.button("DASHBOARD"):
            st.session_state.page = "Dashboard"
    with col3:
        if st.button("DATA LOG"):
            st.session_state.page = "Data Log"
    with col4:
        if st.button("LOGOUT"):
            st.session_state.logged_in = False
            st.session_state.page = "Emotion Capture"
            st.success("You have been logged out.")
            st.experimental_rerun()

    return st.session_state.page
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

                # Log the emotion
                log_entry = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Name": st.session_state.name,
                    "ID": st.session_state.sid,
                    "Emotion": emotion
                }
                log_df = pd.DataFrame([log_entry])
                if os.path.exists(log_path):
                    log_df.to_csv(log_path, mode='a', header=False, index=False)
                else:
                    log_df.to_csv(log_path, mode='w', header=True, index=False)

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

# Dashboard page

def show_dashboard():
    st.subheader("📊 Emotion Trend Dashboard")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        emotion_counts = df['Emotion'].value_counts().reindex(emotion_labels, fill_value=0)
        st.line_chart(emotion_counts)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full CSV Log", data=csv, file_name='emotion_log.csv', mime='text/csv')
    else:
        st.info("No emotion data available to display.")

# Data Log page

def show_log():
    st.subheader("🧾 Emotion Detection Log")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        st.dataframe(df.tail(20))
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV Log", data=csv, file_name='emotion_log.csv', mime='text/csv')
    else:
        st.info("No logs found.")

#Main routing logic
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'page' not in st.session_state:
    st.session_state.page = "Emotion Capture"

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
