import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from boxmot import StrongSort
import os

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load YOLO model
model = YOLO("yolo11m.pt")

# Load StrongSORT
tracker = StrongSort()

# Input video
video_path = r"C:\Users\VijaySegunasi\strongsort_yolo\PNNL_Parking_LOT(1).avi"

cap = cv2.VideoCapture(video_path)

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video
out = cv2.VideoWriter(
    "outputs/tracked_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

tracking_data = []
frame_id = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    # YOLO inference
    results = model(frame, verbose=False)

    detections = []

    for box in results[0].boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())

        detections.append(
            [x1, y1, x2, y2, conf, cls]
        )

    # Run StrongSORT only if detections exist
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

            # Debug print to understand output format
            print(track)

            x1 = track[0]
            y1 = track[1]
            x2 = track[2]
            y2 = track[3]

            track_id = int(track[4])

            confidence = 0.0
            if len(track) > 5:
                confidence = float(track[5])

            tracking_data.append([
                frame_id,
                track_id,
                x1,
                y1,
                x2,
                y2,
                confidence
            ])

            # Draw box
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

cap.release()
out.release()

# Save CSV
df = pd.DataFrame(
    tracking_data,
    columns=[
        "frame",
        "id",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence"
    ]
)

df.to_csv(
    "outputs/tracking_results.csv",
    index=False
)

print("Tracking Complete!")
print("Video saved: outputs/tracked_video.mp4")
print("CSV saved: outputs/tracking_results.csv")