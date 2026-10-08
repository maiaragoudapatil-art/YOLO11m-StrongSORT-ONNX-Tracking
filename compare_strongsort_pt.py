import cv2
import pandas as pd
import numpy as np
import os
import time
import json
from boxmot import StrongSort, OCSORT, DeepOCSORT
from ultralytics import YOLO
from boxmot import StrongSort

import sys

VIDEO_PATH = sys.argv[1]
TRACKER_NAME = sys.argv[2]
MODEL_PATH = "yolo11m.pt"

OUTPUT_VIDEO = "Outputs/tracked_video_pt.mp4"
OUTPUT_CSV = "Outputs/tracking_results_pt.csv"
OUTPUT_JSON = "Outputs/benchmark_pt.json"

os.makedirs("Outputs", exist_ok=True)

print("Loading PT Model...")
model = YOLO(MODEL_PATH)

print("Loading " + TRACKER_NAME + "...")
tracker = get_tracker(TRACKER_NAME)


def get_tracker(name):

    if name.lower() == "strongsort":
        return StrongSort()

    elif name.lower() == "ocsort":
        return OCSORT()

    elif name.lower() == "deepocsort":
        return DeepOCSORT()

    else:
        raise ValueError(
            f"Unsupported tracker: {name}"
        )

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise Exception("Cannot open video")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = float(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video opened successfully")
print(f"Total Frames: {total_frames}")
print(f"FPS: {fps}")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

tracking_data = []

frame_id = 0

start_time = time.time()

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    if frame_id % 100 == 0:
        print(f"Processed {frame_id}/{total_frames} frames")

    results = model(frame, verbose=False)

    detections = []

    for box in results[0].boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())

        detections.append(
            [x1, y1, x2, y2, conf, cls]
        )

    tracks = []

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

cap.release()
out.release()

total_time = time.time() - start_time

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
    OUTPUT_CSV,
    index=False
)

unique_track_ids = (
    df["id"].nunique()
    if not df.empty else 0
)

benchmark = {
    "project": {
        "model": "YOLO11m",
        "tracker": TRACKER_NAME,
        "format": "PyTorch"
    },

    "video": {
        "path": VIDEO_PATH,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": frame_id
    },

    "performance": {
        "total_processing_time_seconds": round(total_time, 2),
        "average_fps": round(frame_id / total_time, 2),
        "average_latency_ms": round(
            (total_time / frame_id) * 1000,
            2
        )
    },

    "tracking": {
        "total_detections": len(df),
        "unique_track_ids": int(unique_track_ids)
    },

    "outputs": {
        "video": os.path.abspath(OUTPUT_VIDEO),
        "csv": os.path.abspath(OUTPUT_CSV)
    }
}

with open(
    OUTPUT_JSON,
    "w"
) as f:
    json.dump(
        benchmark,
        f,
        indent=4
    )

print("\n===== FINISHED =====")
print(f"Frames Processed : {frame_id}")
print(f"Total Time       : {total_time:.2f} sec")
print(f"FPS              : {frame_id / total_time:.2f}")
print(f"Detections       : {len(df)}")
print(f"Unique Track IDs : {unique_track_ids}")

print("\nOutputs")
print(f"Video : {OUTPUT_VIDEO}")
print(f"CSV   : {OUTPUT_CSV}")
print(f"JSON  : {OUTPUT_JSON}")