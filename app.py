import streamlit as st
import cv2
import torch
from PIL import Image

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Update path to your model

# Streamlit UI
st.title("Helmet Detection with YOLO")
st.markdown("Upload an image or video to detect helmets.")

# File uploader
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
        # Process the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Convert image for YOLO inference
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = model(image_np)  # Inference
        st.image(results.render()[0], caption='Detected Image', use_column_width=True)

    elif uploaded_file.type == "video/mp4":
        # Process the uploaded video
        video_path = uploaded_file.name
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Perform detection on each frame
            results = model(frame)  # Inference
            st.image(results.render()[0], caption="Detected Frame", use_column_width=True)

        cap.release()