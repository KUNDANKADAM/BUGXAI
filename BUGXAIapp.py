import streamlit as st
import cv2
import pandas as pd
import numpy as np
import time
import tempfile
import os
from ultralytics import YOLO
import uuid
from datetime import datetime
import csv
import io


MODEL_PATH = r"C:\Users\dhruv\Documents\RM Sem 2\Codes\dataset\prem\best.pt"  # Update with your actual YOLO model path if needed

# Class names for your model
class_names = ["Physics Glitch", "Visual Glitch", "Frame Drop", "Texture Glitch", "Screen Freeze"]

# Adaptive screen freeze variables (global for demonstration)
INITIAL_FREEZE_THRESH = 100000
FREEZE_FRAME_THRESH = INITIAL_FREEZE_THRESH
MIN_FREEZE_DURATION = 5
MAX_FREEZE_DURATION = 50
freeze_counter = 0
freeze_detected = False
frame_diffs = []

# Create session states for controlling the process
if "stop_process" not in st.session_state:
    st.session_state.stop_process = False
if "log_text" not in st.session_state:
    st.session_state.log_text = ""  # Will accumulate logs
if "processed_video_path" not in st.session_state:
    st.session_state.processed_video_path = None

# ---------------------------
# Helper Functions
# ---------------------------
st.set_page_config(layout="wide")
def detect_frame_drop(prev_frame, current_frame, threshold=30):
    """Detects frame drops based on pixel differences."""
    if prev_frame is None:
        return False
    diff = cv2.absdiff(prev_frame, current_frame)
    non_zero_count = np.count_nonzero(diff)
    return non_zero_count < threshold

def frame_to_timestamp(frame_number, fps):
    """Convert frame number to MM:SS format"""
    total_seconds = frame_number / fps
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02}:{seconds:02}"


def process_video(input_path, output_path, image_placeholder, log_placeholder):
    """Process video using YOLO model and glitch detection with overlay metrics."""

    global FREEZE_FRAME_THRESH, freeze_counter, freeze_detected, frame_diffs

    FREEZE_FRAME_THRESH = INITIAL_FREEZE_THRESH
    freeze_counter = 0
    freeze_detected = False
    frame_diffs = []
    st.session_state.log_text = ""
    csv_records = []

    model = YOLO(MODEL_PATH).to("cuda")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 FPS if unavailable
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    prev_frame = None
    frame_count = 0

    while cap.isOpened():
        if st.session_state.get("stop_process", False):
            break

        start_time = time.time()  # For FPS calculation

        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        video_timestamp = frame_to_timestamp(frame_count, fps)  # MM:SS format
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run YOLO inference
        results = model.predict(frame, batch=8)

        detections_text = []
        csv_detections = []  # Store CSV records
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                class_label = class_names[cls]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{class_label} ({conf:.2f})"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                detection_info = f"({video_timestamp}) Frame {frame_count}: {class_label} at ({x1},{y1},{x2},{y2}) conf={conf:.2f}"
                detections_text.append(detection_info)
                csv_detections.append([video_timestamp, frame_count, class_label, f"({x1},{y1},{x2},{y2})", conf])


        # Frame Drop Detection
        frame_drop_detected = detect_frame_drop(prev_frame, frame_gray) if prev_frame is not None else False
        if frame_drop_detected:
            detections_text.append(f"({video_timestamp}) Frame {frame_count}: Frame Drop Detected")
            csv_detections.append([video_timestamp, frame_count, "Frame Drop", "Detected", ""])


        # Screen Freeze Detection
        if prev_frame is not None:
            frame_diff = cv2.absdiff(frame_gray, prev_frame)
            diff_sum = np.sum(frame_diff)
            frame_diffs.append(diff_sum)
            if len(frame_diffs) > 50:
                frame_diffs.pop(0)

            avg_diff = np.mean(frame_diffs)
            FREEZE_FRAME_THRESH = max(INITIAL_FREEZE_THRESH, avg_diff * 0.2)

            if diff_sum < FREEZE_FRAME_THRESH:
                freeze_counter += 1
                if MIN_FREEZE_DURATION <= freeze_counter <= MAX_FREEZE_DURATION:
                    detections_text.append(f"({video_timestamp}) Frame {frame_count}: Screen Freeze ({freeze_counter} frames)")
                    csv_detections.append([video_timestamp, frame_count, "Screen Freeze", f"{freeze_counter} frames", ""])
                    freeze_detected = True
            else:
                freeze_counter = 0
                freeze_detected = False

        prev_frame = frame_gray.copy()

        # Calculate FPS
        elapsed_time = time.time() - start_time
        current_fps = round(1.0 / elapsed_time, 2) if elapsed_time > 0 else 0.0

        # Overlay metrics on frame
        overlay_text = [
            f"Frame: {frame_count}",
            f"Time: {video_timestamp}",
            f"FPS: {current_fps}",
        ]
        if freeze_detected:
            overlay_text.append(f"Screen Freeze ({freeze_counter} frames)")
        if frame_drop_detected:
            overlay_text.append("Frame Drop Detected")

        # Display text on frame
        for i, text in enumerate(overlay_text):
            cv2.putText(frame, text, (50, 50 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Write frame to output video
        out.write(frame)

        # Update Streamlit UI
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_placeholder.image(frame_rgb, caption="Live Glitch Detection", use_container_width=True)

        if detections_text:  # Append only if detection occurs
            st.session_state.log_text += "\n".join(detections_text) + "\n"

        log_placeholder.text_area(
            "",
            st.session_state.log_text.strip(),
            height=300,
            key=f"log_text_area_{uuid.uuid4()}"
        )
        csv_records.extend(csv_detections)


        time.sleep(0.01)

    cap.release()
    out.release()
    if csv_records:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Timestamp", "Frame", "Detection", "Location", "Confidence"])  # Header row
        writer.writerows(csv_records)
        st.session_state.csv_data = output.getvalue()

def run_test_tab():
    """
    The code for the 'Test' tab:
      - Upload video
      - Predict button
      - Stop button
      - Download log
      - Download processed video
    """

    # File uploader
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

    # Columns layout
    video_col, _, log_col = st.columns([0.45, .00001, 0.45])

    with video_col:
        st.subheader("Live Glitch Detection")
        image_placeholder = st.empty()

    with log_col:
        st.subheader("Detection Log")
        log_placeholder = st.empty()

    # Button row
    predict_col, stop_col, download_log_col, download_vid_col = st.columns([1, 1, 1, 1])

    with predict_col:
        if st.button("Predict", use_container_width=True):
            if uploaded_file:
                st.session_state.stop_process = False  # Reset stop flag

                # Create temporary paths for input and output
                temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_input_path = temp_input.name
                temp_output_path = temp_output.name
                temp_input.write(uploaded_file.read())
                temp_input.close()
                temp_output.close()

                # Run processing
                process_video(temp_input_path, temp_output_path, image_placeholder, log_placeholder)

                # Save final processed video path to session
                st.session_state.processed_video_path = temp_output_path
            else:
                st.warning("Please upload a video before clicking Predict.")

    with stop_col:
        if st.button("Stop Process", use_container_width=True):
            st.session_state.stop_process = True
            st.warning("Process stopped.")

    with download_log_col:

        # Ensure CSV session state is initialized
        if "csv_data" not in st.session_state:
            st.session_state.csv_data = None

        # Download TXT Log
        st.download_button(
            label="📜 Download Log (TXT)",
            data=st.session_state.log_text if st.session_state.log_text else "No logs available.",
            file_name="detection_log.txt",
            mime="text/plain",
            use_container_width=True,
            disabled=(not st.session_state.log_text)
        )

        # Download CSV Log (only if new values exist)
        if st.session_state.csv_data:
            st.download_button(
                label="📜 Download Log (CSV)",
                data=st.session_state.csv_data,
                file_name="glitch_log.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.button("📜 Download Log (CSV)", disabled=True, use_container_width=True)


    with download_vid_col:
        if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
            with open(st.session_state.processed_video_path, "rb") as f:
                st.download_button(
                    label="📹 Download Video",
                    data=f,
                    file_name="processed_video.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
        else:
            st.button("📹 Download Video", disabled=True, use_container_width=True)



def run_model_tab():
    st.header("Model Analysis 🧠")
    img_width = 600  # Adjust as needed for smaller images
    # Why YOLOv8?
    st.subheader("🤖 Why YOLOv8 for Glitch Detection?")
    st.write("""
    - **Fast & Efficient**: Optimized for real-time detection with high FPS.  
    - **High Accuracy**: Balances speed and accuracy, crucial for game testing.  
    - **Pre-trained & Transfer Learning**: Improves detection without massive datasets.  
    """)
    st.write("")


     # Why Tuning?
    st.subheader("🛠️ Why Model Tuning?")
    st.write("""
    - **Improve Class 1 Detection**: Class 1 had lower recall initially.  
    - **Adjust Confidence Thresholds**: Balancing precision and recall.  
    - **Data Augmentation**: Improved generalization with varied training data.  
    """)
    st.write("")

    # Training Time
    st.subheader("⏳ Training Time & Performance")
    st.write("""
    - **Training Time**: ~29 hours on an Intel i3.  
    - **Dataset Size**: ~5500 labeled glitch images.  
    - **Final Model Size**: ~6MB, lightweight for real-time applications.  
    """)
    st.write("")

    cols = st.columns([0.6, 0.01, 1])
    with cols[0]:
        st.image(r"C:\Users\dhruv\Pictures\Screenshots\Screenshot 2025-03-11 215817.png",  
                caption="Model Train", use_container_width =True)
    with cols[2]:
        st.image(r"C:\Users\dhruv\Pictures\Screenshots\Screenshot 2025-03-11 215832.png", 
                caption="Train Completion", use_container_width =True)

    st.header("Metrics 📊")

    # Confusion Matrix
    col9, col10 = st.columns([1, 2])
    with col9:
        st.image(r"C:\Users\dhruv\Downloads\train3\train3\confusion_matrix_normalized.png", use_container_width =True)
    with col10:
        st.subheader("1. Confusion Matrix")
        st.write("""
        - **Class 2** has the highest accuracy (0.66) among predicted classes.  
        - **Class 1** is often misclassified as **background (0.43)**.  
        - **Background** class is confused with **Class 1 (0.43)** and **Class 2 (0.21)**. 
        """)


    # Precision-Recall Curve
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("graphs/PR_curve.png", use_container_width =True)
    with col2:
        st.subheader("2. Precision-Recall Curve")
        st.write("""
        - Class 2 performs better than Class 1.  
        - Overall **mAP@0.5 = 0.674** ✅  
        """)

    # Recall-Confidence Curve
    col3, col4 = st.columns([1, 2])
    with col3:
        st.image("graphs/R_curve.png", use_container_width =True)
    with col4:
        st.subheader("3. Recall-Confidence Curve")
        st.write("""
        - Class 1 recall drops fast at higher confidence.  
        - Class 2 stays stable. Overall recall **0.79 at 0.0 confidence**.  
        """)

    # Precision-Confidence Curve
    col5, col6 = st.columns([1, 2])
    with col5:
        st.image("graphs/P_curve.png", use_container_width =True)
    with col6:
        st.subheader("4. Precision-Confidence Curve")
        st.write("""
        - Lower confidence → High recall but more false positives.  
        - Higher confidence → Better precision but fewer detections.  
        """)

    # F1 Score
    col7, col8 = st.columns([1, 2])
    with col7:
        st.image("graphs/F1_curve.png", use_container_width =True)
    with col8:
        st.subheader("5. F1 Score")
        st.write("""
        - **Class 1:** 0.65 | **Class 2:** 0.82  
        - **Overall F1 Score:** 0.74 🔥  
        """)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Conclusion
    st.subheader("⚡ Final Thoughts")
    st.write("""
    ✅ Class 2 outperforms Class 1.  
    ✅ Model performs well, but Class 1 recall needs improvement.  
    ✅ Future scope: Better training, confidence tuning & real-time deployment!  
    """)

def run_dataset_tab():
    st.header("📂 Dataset Overview")

    # Dataset Sources
    st.subheader("🔹 Data Collection Sources")
    
    st.write("""
    - **GameXPhysics** 🎮: A dataset of glitch videos from multiple games.  
    [🔗 GameXPhysics Dataset](https://drive.google.com/drive/folders/1pn5rpixj8KCRWsEGO8bvFagjZwOIt-My)  
    - **YouTube** ▶️: Collected various glitch videos from different YouTube sources.  
    - **Own Dataset** 🏗️: Custom dataset created for additional training.  
    """)
    st.write("")  # Spacing

    # Preprocessing Steps
    st.subheader("⚙️ Data Preprocessing")
    
    st.write("""
    - **Grayscale Conversion**: Reduces complexity & focuses on structure.  
    - **Resize**: Stretched to **640x640** for uniformity.  
    - **Flip**: Random **horizontal flipping** for augmentation.  
    - **Brightness Adjustment**: Random variation between **-25% and +25%**.  
    - **Noise Injection**: Up to **1.09% of pixels** modified for robustness.  
    """)
    st.write("")  # Spacing

    # Annotation using Roboflow
    st.subheader("📝 Annotation Process")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(r"C:\Users\dhruv\Pictures\Screenshots\Screenshot 2025-03-11 192454.png", use_container_width =True)
    with col2:
        st.write("""
        - Annotation done using **Roboflow** for labeling game glitches.  
        - Bounding boxes were manually verified for accuracy.  
        - Ensured balanced class distribution across categories.  
        """)

def main():
    st.title("🎮 BUGXAI")

    # Create a tab-based navigation
    tabs = st.tabs(["Test", "Model", "Dataset"])

    with tabs[0]:
        run_test_tab()
    with tabs[1]:
        run_model_tab()
    with tabs[2]:
        run_dataset_tab()

if __name__ == "__main__":
    main()
