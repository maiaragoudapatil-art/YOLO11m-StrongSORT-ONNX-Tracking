import cv2
import pandas as pd
import numpy as np
import os
import time
#pip show boxmot
from ultralytics import YOLO
from boxmot import StrongSort

# -----------------------------
# SETTINGS
# -----------------------------
VIDEO_PATH = r"C:\Users\VijaySegunasi\strongsort_yolo\videos\PNNL_Parking_LOT(1).avi"
MODEL_PATH = "yolo11m.onnx"

os.makedirs("outputs", exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading ONNX Model...")
model = YOLO(MODEL_PATH)

print("Loading StrongSORT...")
tracker = StrongSort()

# -----------------------------
# OPEN VIDEO
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video opened successfully")
print(f"Total Frames: {frame_count_total}")
print(f"FPS: {fps}")

# -----------------------------
# OUTPUT VIDEO
# -----------------------------
out = cv2.VideoWriter(
    "outputs/tracked_video_onnx.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# -----------------------------
# TRACKING STORAGE
# -----------------------------
tracking_data = []

frame_id = 0

start_time = time.time()

# -----------------------------
# PROCESS VIDEO
# -----------------------------
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    if frame_id % 100 == 0:
        print(f"Processed {frame_id}/{frame_count_total} frames")

    # YOLO Detection
    results = model(frame, verbose=False)

    detections = []

    for box in results[0].boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())

        detections.append(
            [x1, y1, x2, y2, conf, cls]
        )

    # StrongSORT
    if len(detections) > 0:

        detections = np.array(
            detections,
            dtype=np.float32
        )

        tracks = tracker.update(
            detections,
            frame
        )

        for track in tracks:

            x1 = float(track[0])
            y1 = float(track[1])
            x2 = float(track[2])
            y2 = float(track[3])

            track_id = int(track[4])

            tracking_data.append([
                frame_id,
                track_id,
                x1,
                y1,
                x2,
                y2
            ])

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID {track_id}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    out.write(frame)

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
out.release()

total_time = time.time() - start_time

# -----------------------------
# SAVE CSV
# -----------------------------
df = pd.DataFrame(
    tracking_data,
    columns=[
        "frame",
        "id",
        "x1",
        "y1",
        "x2",
        "y2"
    ]
)

df.to_csv(
    "outputs/tracking_results_onnx.csv",
    index=False
)

# -----------------------------
# RESULTS
# -----------------------------
print("\n===== FINISHED =====")
print(f"Frames Processed : {frame_id}")
print(f"Total Time       : {total_time:.2f} sec")

if total_time > 0:
    print(f"FPS              : {frame_id / total_time:.2f}")

print("Video Saved      : outputs/tracked_video_onnx.mp4")
print("CSV Saved        : outputs/tracking_results_onnx.csv")