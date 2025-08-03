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

# Background setter
def set_plain_bg(image_path):
    if os.path.exists(image_path):
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
    else:
        # fallback solid background
        st.markdown("""
            <style>
            .stApp { background: #e8f6f9; }
            </style>
        """, unsafe_allow_html=True)

# Login screen
def login_screen():
    set_plain_bg("background.png")
    st.markdown("""
        <div style="max-width:700px; margin:auto; padding:40px; background:#0f4c75; border-radius:12px; color:white;">
            <h2 style='text-align: center; margin-bottom:5px;'>🔐 STUDENT LOGIN</h2>
            <p style='text-align:center; margin-top:0;'>Welcome to Virtual EmoDash. Please enter your details to continue.</p>
            <div style="display:flex; gap:20px; justify-content:center; flex-wrap:wrap;">
                <div style="flex:1; min-width:250px;">
                    <label style="color:white; font-weight:bold;">Name</label>
                    <input type="text" id="name_input" style="width:100%; padding:8px; border-radius:6px; border:none;" placeholder="Enter your Name">
                </div>
                <div style="flex:1; min-width:250px;">
                    <label style="color:white; font-weight:bold;">Student ID</label>
                    <input type="text" id="id_input" style="width:100%; padding:8px; border-radius:6px; border:none;" placeholder="Enter your ID">
                </div>
            </div>
            <div style="text-align:center; margin-top:20px;">
                <button id="login_btn" style="background:#00b7c2; color:white; padding:12px 30px; border:none; border-radius:8px; font-weight:bold; cursor:pointer;">
                    LOGIN
                </button>
            </div>
        </div>
        <script>
        const loginBtn = window.parent.document.querySelector("#login_btn");
        </script>
    """, unsafe_allow_html=True)

    # fallback to regular inputs if custom styling doesn't capture
    name = st.text_input("Enter your Name")
    sid = st.text_input("Enter your ID")
    login_btn = st.button("LOGIN", use_container_width=True)

    if login_btn:
        if name.strip() and sid.strip():
            st.session_state.logged_in = True
            st.session_state.name = name
            st.session_state.sid = sid
            st.session_state.page = "Home"
            st.experimental_set_query_params(page="Home")
            st.experimental_rerun()
        else:
            st.warning("Please enter both name and ID.")

# Navigation bar with Home
def nav_bar():
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    # styling bar
    st.markdown("""
        <style>
        .taskbar {
            background-color: #006d77;
            padding: 8px 20px;
            display: flex;
            align-items: center;
            gap: 15px;
            border-radius: 8px;
        }
        .taskbar .title {
            font-size: 22px;
            font-weight: bold;
            color: white;
            margin-right: auto;
        }
        .taskbar button {
            background: transparent;
            border: none;
            color: white;
            padding: 10px 16px;
            font-weight: bold;
            text-transform: uppercase;
            cursor: pointer;
            border-radius: 6px;
        }
        .taskbar button:hover {
            background: #014f57;
        }
        .active {
            background: #00b7c2;
        }
        </style>
    """, unsafe_allow_html=True)

    cols = st.columns([1, 1, 1, 1, 1])
    with cols[0]:
        st.markdown("<div class='taskbar'><div class='title'>🎓 VIRTUAL EMODASH</div>", unsafe_allow_html=True)
    # buttons
    def nav_button(label, page_key):
        is_active = st.session_state.get("page", "") == page_key
        style = "active" if is_active else ""
        if st.button(label, key=label):
            st.session_state.page = page_key

    with cols[1]:
        nav_button("Home", "Home")
    with cols[2]:
        nav_button("Emotion Capture", "Emotion Capture")
    with cols[3]:
        nav_button("Dashboard", "Dashboard")
    with cols[4]:
        nav_button("Data Log", "Data Log")
    # logout separate
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "Emotion Capture"
        st.success("You have been logged out.")
        st.experimental_set_query_params(page="Emotion Capture")
        st.experimental_rerun()
    # close title div
    st.markdown("</div>", unsafe_allow_html=True)
    return st.session_state.get("page", "Home")

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
        st.markdown("**Emotion Distribution Over All Captures**")
        st.line_chart(emotion_counts)
        st.markdown("**Summary Statistics**")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Entries", len(df))
        with c2:
            st.metric("Unique Students", df["ID"].nunique())
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

# Home page
def show_home():
    st.markdown("""
        <div style="background: rgba(255,255,255,0.9); padding:25px; border-radius:12px; max-width:1000px; margin:auto;">
            <h1 style="color:#006d77; text-align:center; margin:5px;">WELCOME TO VIRTUAL EMODASH</h1>
            <p style="text-align:center; font-size:14px; margin-top:0;">
                Real-time emotion-aware learning monitoring system for improving engagement and teaching effectiveness.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Quick Overview")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        total_entries = len(df)
        unique_students = df["ID"].nunique() if "ID" in df.columns else 0
        emotion_counts = df['Emotion'].value_counts().reindex(emotion_labels, fill_value=0)
        recent = df.tail(5)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Captures", total_entries)
        with c2:
            st.metric("Unique Students", unique_students)
        with c3:
            top_emotion = emotion_counts.idxmax() if not emotion_counts.empty else "N/A"
            st.metric("Top Emotion", top_emotion)

        st.markdown("**Emotion Distribution**")
        st.bar_chart(emotion_counts)

        st.markdown("**Recent Activity**")
        st.table(recent)

        st.markdown("""
            <div style="border:1px solid #006d77; padding:12px; border-radius:8px; background:#e8f6f9;">
                <strong>Faculty Tips:</strong>
                <ul>
                    <li>Monitor spikes in negative emotions and reach out proactively.</li>
                    <li>Use trend data to adjust pacing or offer breaks.</li>
                    <li>Provide positive reinforcement when neutral/happy trends dominate.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No data captured yet. Go to Emotion Capture to begin.")

# Main routing logic
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'page' not in st.session_state:
    st.session_state.page = "Home"

if not st.session_state.logged_in:
    login_screen()
else:
    set_plain_bg("background.png")
    page = nav_bar()

    with st.container():
        if page == "Home":
            show_home()
        elif page == "Emotion Capture":
            detect_emotion()
        elif page == "Dashboard":
            show_dashboard()
        elif page == "Data Log":
            show_log()
