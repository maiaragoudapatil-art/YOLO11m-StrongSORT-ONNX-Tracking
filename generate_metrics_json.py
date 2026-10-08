import pandas as pd
import cv2
import json
import os

#configuration

PROJECT_NAME = "YOLO11m + StrongSORT Object Tracking"
TRACKER_NAME = "StrongSORT"

MODEL_NAME = "YOLO11m"
MODEL_FORMAT = "onnx"

MODEL_PATH = r"C:\Users\VijaySegunasi\strongsort_yolo\yolo11m.onnx"

VIDEO_PATH = r"C:\Users\VijaySegunasi\strongsort_yolo\videos\PNNL_Parking_LOT(1).avi"

CSV_PATH = r"outputs\tracking_results_onnx.csv"

OUTPUT_VIDEO = r"outputs\tracked_video_onnx.mp4"

OUTPUT_JSON = r"outputs\tracking_metrics.json"

OUTPUT_LOG = r"outputs\strongsort.log"

# values obtained from tracking execution
TOTAL_PROCESSING_TIME = 1011.32
AVERAGE_FPS = 0.99

# Optional - update if measured
AVERAGE_LATENCY_MS = None
P95_LATENCY_MS = None
AVERAGE_CPU_PERCENT = None
PEAK_CPU_PERCENT = None
AVERAGE_RAM_MB = None
PEAK_RAM_MB = None
SYSTEM_AVG_CPU = None
SYSTEM_PEAK_CPU = None

#Video info
cap = cv2.VideoCapture(VIDEO_PATH)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = float(cap.get(cv2.CAP_PROP_FPS))
video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()

# TRACKING METRICS
df = pd.read_csv(CSV_PATH)

total_detections = len(df)

unique_track_ids = int(
    df["id"].nunique()
)

active_tracks = (
    df.groupby("frame")["id"]
    .nunique()
)

average_active_tracks = float(
    active_tracks.mean()
)

maximum_active_tracks = int(
    active_tracks.max()
)

track_lifetimes = (
    df.groupby("id")["frame"]
    .count()
)

average_track_lifetime = float(
    track_lifetimes.mean()
)

longest_track_lifetime = int(
    track_lifetimes.max()
)
#JSON FORMATE

metrics = {

    "project": {
        "name": PROJECT_NAME,
        "tracker": TRACKER_NAME
    },

    "model": {
        "name": MODEL_NAME,
        "format": MODEL_FORMAT,
        "path": MODEL_PATH
    },

    "configuration": {
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "tracker_config": "BoxMOT.StrongSORT",
        "device": "CPU"
    },

    "video": {
        "input": VIDEO_PATH,
        "width": video_width,
        "height": video_height,
        "fps": video_fps,
        "total_frames": video_frames
    },

    "tracking_statistics": {

        "frame_count": video_frames,

        "total_detections": total_detections,

        "unique_track_ids": unique_track_ids,

        "average_active_tracks":
            average_active_tracks,

        "maximum_active_tracks":
            maximum_active_tracks,

        "average_track_lifetime":
            average_track_lifetime,

        "longest_track_lifetime":
            longest_track_lifetime,

        "average_fps":
            AVERAGE_FPS,

        "average_latency_ms":
            AVERAGE_LATENCY_MS,

        "p95_latency_ms":
            P95_LATENCY_MS,

        "total_processing_time_seconds":
            TOTAL_PROCESSING_TIME
    },

    "resources": {

        "process": {

            "average_cpu_percent":
                AVERAGE_CPU_PERCENT,

            "peak_cpu_percent":
                PEAK_CPU_PERCENT,

            "average_ram_mb":
                AVERAGE_RAM_MB,

            "peak_ram_mb":
                PEAK_RAM_MB
        },

        "system": {

            "average_cpu_percent":
                SYSTEM_AVG_CPU,

            "peak_cpu_percent":
                SYSTEM_PEAK_CPU
        }
    },

    "outputs": {

        "video": OUTPUT_VIDEO,

        "json": OUTPUT_JSON,

        "log": OUTPUT_LOG
    }
}


with open(
    OUTPUT_JSON,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

print(
    f"Metrics JSON saved to:\n{OUTPUT_JSON}"
)