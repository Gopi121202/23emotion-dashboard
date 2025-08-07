import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from tensorflow.keras.models import load_model
import os
import base64
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Paths and setup
MODEL_PATH = "model/model.keras"
CASCADE_PATH = "haarcascade.xml"
LOG_PATH = "data/emotion_log.csv"
os.makedirs("data", exist_ok=True)

# Load model and cascade
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
alert_emotions = {"Angry", "Sad", "Fear"}

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
        st.markdown("""
            <style>
            .stApp { background: #e8f6f9; }
            </style>
        """, unsafe_allow_html=True)

# JavaScript alert sound + animation trigger
def play_alert_script():
    st.markdown("""
        <script>
        (function() {
            try {
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const o = ctx.createOscillator();
                const g = ctx.createGain();
                o.type = 'square';
                o.frequency.setValueAtTime(600, ctx.currentTime);
                g.gain.setValueAtTime(0.1, ctx.currentTime);
                o.connect(g);
                g.connect(ctx.destination);
                o.start();
                setTimeout(() => { o.stop(); }, 120);
            } catch(e) {
                console.log("Audio API unavailable", e);
            }
        })();
        </script>
    """, unsafe_allow_html=True)

# Login form
def login_form():
    st.markdown("""
        <div style="max-width:600px; margin:40px auto; padding:30px; background:#0f4c75; border-radius:12px; color:white;">
            <h2 style='text-align: center; margin-bottom:5px;'>🔐 LOGIN</h2>
            <p style='text-align:center; margin-top:0;'>Enter your name and ID to proceed.</p>
        </div>
    """, unsafe_allow_html=True)
    with st.form("student_login_form", clear_on_submit=False):
        name = st.text_input("Name", key="login_name")
        sid = st.text_input("ID", key="login_sid")
        submitted = st.form_submit_button("LOGIN")
        if submitted:
            if name.strip() and sid.strip():
                st.session_state.logged_in = True
                st.session_state.name = name.strip()
                st.session_state.sid = sid.strip()
                st.session_state.page = "Home"
                st.experimental_set_query_params(page="Home")
                st.success(f"WELCOME, {st.session_state.name}!")
                st.experimental_rerun()
            else:
                st.warning("Please enter both name and ID.")

# Navigation bar
def nav_bar():
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    st.markdown("""
        <style>
        .taskbar {
            background-color: #006d77;
            padding: 8px 16px;
            display: flex;
            align-items: center;
            gap: 10px;
            border-radius: 8px;
            margin-bottom:10px;
        }
        .taskbar .title {
            font-size: 20px;
            font-weight: bold;
            color: white;
            margin-right: auto;
        }
        .taskbar button {
            background: transparent;
            border: none;
            color: white;
            padding: 8px 14px;
            font-weight: bold;
            text-transform: uppercase;
            cursor: pointer;
            border-radius: 6px;
        }
        .taskbar button.active {
            background: #00b7c2;
        }
        .taskbar button:hover {
            background: #014f57;
        }
        </style>
    """, unsafe_allow_html=True)

    cols = st.columns([1, 0.7, 0.7, 0.7, 0.7, 0.7])
    with cols[0]:
        st.markdown("<div class='taskbar'><div class='title'>🎓VIRTUAL EMODASH</div>", unsafe_allow_html=True)

    def make_btn(label, page_key, col):
        is_active = st.session_state.get("page", "") == page_key
        with col:
            btn = st.button(label, key=f"nav_{page_key}")
            if btn:
                st.session_state.page = page_key

    make_btn("Home", "Home", cols[1])
    make_btn("Emotion Capture", "Emotion Capture", cols[2])
    make_btn("Dashboard", "Dashboard", cols[3])
    make_btn("Data Log", "Data Log", cols[4])
    with cols[5]:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.page = "Home"
            st.experimental_set_query_params(page="Home")
            st.success("You have been logged out.")
            st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    return st.session_state.get("page", "Home")

# Emotion capture
def detect_emotion():
    st.subheader("CAPTURE IMAGE")
    col1, col2 = st.columns([1, 2])

    with col1:
        image = st.camera_input("Take a picture")
    with col2:
        if image:
            img = Image.open(image)
            img_np = np.array(img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            detected_emotion = None
            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (48, 48)) / 255.0
                roi = roi.reshape(1, 48, 48, 1)

                pred = model.predict(roi)
                emotion = emotion_labels[np.argmax(pred)]
                detected_emotion = emotion

                # Log
                log_entry = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Name": st.session_state.name,
                    "ID": st.session_state.sid,
                    "Emotion": emotion
                }
                log_df = pd.DataFrame([log_entry])
                if os.path.exists(LOG_PATH):
                    log_df.to_csv(LOG_PATH, mode='a', header=False, index=False)
                else:
                    log_df.to_csv(LOG_PATH, mode='w', header=True, index=False)

            st.image(img_np, use_column_width=True)
            if detected_emotion:
                if detected_emotion in alert_emotions:
                    play_alert_script()
                    st.markdown(f"""
                        <div style="background:#ffe6e6; padding:15px; border-radius:8px; margin:10px 0; border:2px solid #d62828;">
                            <strong style="color:#a80000;">ALERT: Detected emotion is <strong>{detected_emotion.upper()}</strong>. Consider intervention.</strong> 
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div style="text-align:center;">
                        <h1 style='color:#d62828; animation: popIn 1s ease-in-out; font-size:50px margin:0;'>DETECTED EMOTION:</h1>
                        <h2 style='color:#d62828; animation: popIn 1s ease-in-out; font-size:50px; margin:5px;'><strong>{detected_emotion.upper()}</strong></h2>
                    </div>
                    <style>
                    @keyframes popIn {{
                        0% {{ transform: scale(0.8); opacity: 0; }}
                        100% {{ transform: scale(1); opacity: 1; }}
                    }}
                    </style>
                """, unsafe_allow_html=True)

# Dashboard
def show_dashboard():
    st.subheader("📊 EMOTION TREND DASHBOARD")
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        emotion_counts = df['Emotion'].value_counts().reindex(emotion_labels, fill_value=0)

        st.markdown("EMOTION DISTRIBUTION OVER ALL CAPTURES")
        st.line_chart(emotion_counts)

        st.markdown("SUMMARY STATISTICS")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Entries", len(df))
        with c2:
            st.metric("Unique Students", df["ID"].nunique() if "ID" in df.columns else 0)

        st.markdown("BREAKDOWN BY EMOTION PERCENTAGE")
        total = emotion_counts.sum()
        if total > 0:
            percentages = (emotion_counts / total * 100).round(1)
        else:
            percentages = emotion_counts

        # Smaller pie chart
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        wedges, texts, autotexts = ax.pie(
            emotion_counts.values,
            autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
            startangle=90,
            pctdistance=0.65,
            labeldistance=1.0,
            wedgeprops={'linewidth': 0.5, 'edgecolor': 'black'}
        )
        ax.axis('equal')
        legend_labels = [f"{emo}: {percentages[emo]}%" for emo in emotion_counts.index if emotion_counts[emo] > 0]
        ax.legend(wedges, legend_labels, title="Emotions", loc="center left", bbox_to_anchor=(1.1, 0.5), fontsize="x-small", frameon=False)
        st.pyplot(fig)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full CSV Log", data=csv, file_name='emotion_log.csv', mime='text/csv')
    else:
        st.info("No data yet. Capture emotions first.")

# Data log
def show_log():
    st.subheader("EMOTION DETECTION LOG")
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        st.dataframe(df.tail(150))
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV Log", data=csv, file_name='emotion_log.csv', mime='text/csv')
    else:
        st.info("No logs found.")

# Home (with overview)
def show_home():
    st.markdown("""
        <div style="background: rgba(255,255,255,0.9); padding:25px; border-radius:12px; max-width:1000px; margin:auto;">
            <h1 style="color:#006d77; text-align:center; margin:5px;">WELCOME TO VIRTUAL EMODASH</h1>
            <p style="color:#006d77; text-align:center; font-size:14px; margin-top:0;">
                Real-time emotion-aware learning monitoring system for improving engagement and teaching effectiveness.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Quick Overview")
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        total_entries = len(df)
        unique_students = df["ID"].nunique() if "ID" in df.columns else 0
        emotion_counts = df['Emotion'].value_counts().reindex(emotion_labels, fill_value=0)
        recent = df.tail(10)

        # check top emotion alert
        top_emotion = emotion_counts.idxmax() if not emotion_counts.empty else None
        if top_emotion in alert_emotions:
            play_alert_script()
            st.markdown(f"""
                <div style="background:#ffe6e6; padding:15px; border-radius:8px; margin:10px 0; border:2px solid #d62828;">
                    <strong style="color:#d62828;">ALERT: Top emotion is <strong>{top_emotion.upper()}</strong>. Immediate attention suggested.</strong>
                </div>
            """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Captures", total_entries)
        with c2:
            st.metric("Unique Students", unique_students)
        with c3:
            top_em = top_emotion if top_emotion else "N/A"
            st.metric("Top Emotion", top_em.upper())

        st.markdown("Emotion Distribution")
        st.bar_chart(emotion_counts)

        st.markdown("""
        <div style="border:1px solid #E29578; padding:6px; border-radius:8px; background:#006d77;">
                <strong style="color:white;">Recent Activity:</strong>
            </div>
        """, unsafe_allow_html=True)
        st.table(recent)

        st.markdown("""
            <div style="border:1px solid #E29578; padding:12px; border-radius:8px; background:#006d77;">
                <strong style="color:white;">Faculty Tips:</strong>
                <ul>
                    <li style="color:white;">Monitor spikes in negative emotions and reach out proactively.</li>
                    <li style="color:white;">Use trend data to adjust pacing or offer breaks.</li>
                    <li style="color:white;">Provide positive reinforcement when neutral/happy trends dominate.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No data captured yet. Go to Emotion Capture to begin.")

# Home basic (no overview) for pre-login
def show_home_basic():
    st.markdown("""
        <div style="background: rgba(255,255,255,0.9); padding:25px; border-radius:12px; max-width:1000px; margin:auto;">
            <h1 style="color:#006d77; text-align:center; margin:5px;">WELCOME TO VIRTUAL EMODASH</h1>
            <p style="color:#006d77;  text-align:center; font-size:14px; margin-top:0;">
                Real-time emotion-aware learning monitoring system for improving engagement and teaching effectiveness.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Main
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'page' not in st.session_state:
    st.session_state.page = "Home"

set_plain_bg("background.png")

if not st.session_state.logged_in:
    show_home_basic()
    login_form()
else:
    current = nav_bar()
    if current == "Home":
        st.markdown(f"<h3 style='text-align:center; color:white;'>👋 WELCOME, {st.session_state.name}!</h3>", unsafe_allow_html=True)
        show_home()
    elif current == "Emotion Capture":
        detect_emotion()
    elif current == "Dashboard":
        show_dashboard()
    elif current == "Data Log":
        show_log()
