import os
import sys
import json
import subprocess

print("=" * 60)
print("YOLO11m + StrongSORT Evaluation Pipeline")
print("=" * 60)

print("\nPython Environment:")
print(sys.executable)

video_path = input(
    "\nEnter Video Path:\n"
).strip()

if not os.path.exists(video_path):
    raise FileNotFoundError(
        f"Video not found:\n{video_path}"
    )

print("\nStarting PyTorch Evaluation...\n")

subprocess.run(
    [
        sys.executable,
        "compare_strongsort_pt.py",
        video_path
    ],
    check=True
)

print("\nPyTorch Evaluation Completed")

print("\nStarting ONNX Evaluation...\n")

subprocess.run(
    [
        sys.executable,
        "compare_strongsort_onnx.py",
        video_path
    ],
    check=True
)

print("\nONNX Evaluation Completed")

PT_JSON = "Outputs/benchmark_pt.json"
ONNX_JSON = "Outputs/benchmark_onnx.json"

with open(PT_JSON, "r") as f:
    pt = json.load(f)

with open(ONNX_JSON, "r") as f:
    onnx = json.load(f)

pt_time = pt["performance"]["total_processing_time_seconds"]
onnx_time = onnx["performance"]["total_processing_time_seconds"]

pt_fps = pt["performance"]["average_fps"]
onnx_fps = onnx["performance"]["average_fps"]

pt_detections = pt["tracking"]["total_detections"]
onnx_detections = onnx["tracking"]["total_detections"]

pt_tracks = pt["tracking"]["unique_track_ids"]
onnx_tracks = onnx["tracking"]["unique_track_ids"]

speed_winner = (
    "PyTorch"
    if pt_time < onnx_time
    else "ONNX"
)

fps_winner = (
    "PyTorch"
    if pt_fps > onnx_fps
    else "ONNX"
)

overall_winner = "PyTorch"

if speed_winner == "ONNX" and fps_winner == "ONNX":
    overall_winner = "ONNX"

comparison = {

    "project": {
        "model": "YOLO11m",
        "tracker": "StrongSORT"
    },

    "video": {
        "input": video_path
    },

    "pytorch": {

        "processing_time_seconds":
            pt_time,

        "average_fps":
            pt_fps,

        "total_detections":
            pt_detections,

        "unique_track_ids":
            pt_tracks
    },

    "onnx": {

        "processing_time_seconds":
            onnx_time,

        "average_fps":
            onnx_fps,

        "total_detections":
            onnx_detections,

        "unique_track_ids":
            onnx_tracks
    },

    "comparison": {

        "time_difference_seconds":
            round(
                abs(pt_time - onnx_time),
                2
            ),

        "fps_difference":
            round(
                abs(pt_fps - onnx_fps),
                2
            ),

        "speedup_factor":
            round(
                max(pt_time, onnx_time)
                /
                min(pt_time, onnx_time),
                2
            )
    },

    "winner": {

        "fastest_model":
            speed_winner,

        "highest_fps":
            fps_winner,

        "overall_best":
            overall_winner
    }
}

with open(
    "Outputs/final_comparison.json",
    "w"
) as f:

    json.dump(
        comparison,
        f,
        indent=4
    )

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"\nPT Time          : {pt_time:.2f} sec")
print(f"ONNX Time        : {onnx_time:.2f} sec")

print(f"\nPT FPS           : {pt_fps:.2f}")
print(f"ONNX FPS         : {onnx_fps:.2f}")

print(f"\nPT Detections    : {pt_detections}")
print(f"ONNX Detections  : {onnx_detections}")

print(f"\nPT Track IDs     : {pt_tracks}")
print(f"ONNX Track IDs   : {onnx_tracks}")

print("\n" + "=" * 60)
print("WINNER")
print("=" * 60)

print(f"Fastest Model : {speed_winner}")
print(f"Highest FPS   : {fps_winner}")
print(f"Overall Best  : {overall_winner}")

print("\nFinal JSON Report Saved:")
print("Outputs/final_comparison.json")