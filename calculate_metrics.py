import pandas as pd
import cv2
import os
import json

csv_file = "outputs/tracking_results_onnx.csv"
video_file = "outputs/tracked_video_onnx.mp4"

df = pd.read_csv(csv_file)

# Total detections
total_detections = len(df)

# Unique track IDs
unique_track_ids = df["id"].nunique()

# Total frames
total_frames = df["frame"].max()

# Active tracks per frame
active_tracks = df.groupby("frame")["id"].nunique()

avg_active_tracks = active_tracks.mean()
max_active_tracks = active_tracks.max()

# Track lifetimes
lifetimes = df.groupby("id")["frame"].count()

avg_lifetime = lifetimes.mean()
max_lifetime = lifetimes.max()

# Video size
video_size_mb = (
    os.path.getsize(video_file)
    / (1024 * 1024)
)

results = {
    "Total Frames": int(total_frames),
    "Total Detections": int(total_detections),
    "Unique Track IDs": int(unique_track_ids),
    "Average Active Tracks": round(avg_active_tracks, 2),
    "Maximum Active Tracks": int(max_active_tracks),
    "Average Track Lifetime": round(avg_lifetime, 2),
    "Longest Track Lifetime": int(max_lifetime),
    "Output Video Size (MB)": round(video_size_mb, 2)
}

print(json.dumps(results, indent=4))

with open(
    "outputs/tracking_metrics.json",
    "w"
) as f:
    json.dump(results, f, indent=4)

print("\nMetrics saved to outputs/tracking_metrics.json")