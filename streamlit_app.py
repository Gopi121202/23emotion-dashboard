# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import os
import pandas as pd
from tensorflow.keras.models import load_model
from io import BytesIO  # For download fix

# Load model and labels
model = load_model("model/model.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Ensure folders exist
os.makedirs("data/captured", exist_ok=True)
log_path = "data/emotion_log.csv"

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade.xml")

st.title("📊 Student Emotion Monitoring Dashboard")

# Step 1: Student form
with st.form("student_form"):
    student_name = st.text_input("Enter Student Name")
    student_id = st.text_input("Enter Student ID")
    submit = st.form_submit_button("Login")

# Store in session state
if submit:
    st.session_state["student_name"] = student_name
    st.session_state["student_id"] = student_id

# Step 2: Webcam & Prediction
if "student_name" in st.session_state and "student_id" in st.session_state:
    st.header(f"Welcome, {st.session_state['student_name']}")
    camera_image = st.camera_input("📸 Take a picture")

    if camera_image is not None:
        st.success("Image captured successfully!")

        # Convert to grayscale
        img = Image.open(camera_image).convert("L")
        gray = np.array(img)

        # Detect face
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        predicted_emotion = "Unknown"

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48)) / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face)
            emotion_idx = np.argmax(prediction)
            predicted_emotion = emotion_labels[emotion_idx]

        # Save captured image
        img_path = f"data/captured/{st.session_state['student_name']}_{st.session_state['student_id']}.jpg"
        with open(img_path, "wb") as f:
            f.write(camera_image.getbuffer())

        # Save log
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()},{st.session_state['student_name']},{st.session_state['student_id']},{predicted_emotion}\n")

        # Display result
        st.subheader(f"Prediction: {predicted_emotion}")
        st.image(camera_image, caption=f"{st.session_state['student_name']} - {predicted_emotion}", width=300)

        # Emotion suggestions
        if predicted_emotion == "Sad":
            st.warning("The student seems sad. Consider the following improvements:")
            st.markdown("- Offer personal support or mentorship")
            st.markdown("- Encourage breaks or lighter activities")
            st.markdown("- Provide positive feedback or recognition")

        # Fixed download button
        buffered = BytesIO(camera_image.getbuffer())
        st.download_button(
            label="Download Captured Image",
            data=buffered,
            file_name=f"{st.session_state['student_name']}_{st.session_state['student_id']}.jpg",
            mime="image/jpeg")
        # Emotion history plots
        if os.path.exists("data/emotion_log.csv"):
            df = pd.read_csv("data/emotion_log.csv", names=["Time", "Name", "ID", "Emotion"], parse_dates=["Time"])
            st.write(df)
            student_df = df[df["ID"] == student_id]  # Define student_df here
            st.subheader("📈 Emotion Trend for Student")
            if not student_df.empty:
                st.line_chart(student_df["Emotion"].value_counts())
            else:
                st.info("No previous emotion data found for this student.")

            st.subheader("📊 Overall Emotion Distribution")
            st.bar_chart(df["Emotion"].value_counts())



