import json
import sys
import os
import subprocess

TRACKERS = {
    "1": "StrongSORT",
    "2": "OCSORT",
    "3": "DeepOCSORT"
}

print("=" * 60)
print("YOLO11m + BoxMOT Evaluation Framework")
print("=" * 60)

video_path = input(
    "\nEnter video path:\n"
).strip()

if not os.path.exists(video_path):
    raise FileNotFoundError(
        f"Video not found:\n{video_path}"
    )

print("\nAvailable Trackers")

for key, value in TRACKERS.items():
    print(f"{key}. {value}")

tracker_choice = input(
    "\nSelect Tracker: "
).strip()

if tracker_choice not in TRACKERS:
    raise ValueError("Invalid Tracker")

tracker_name = TRACKERS[tracker_choice]

print("\nModes")
print("1. PT")
print("2. ONNX")
print("3. BOTH")

mode = input(
    "\nSelect Mode: "
).strip()

if mode == "1":

    subprocess.run(
        [
            sys.executable,
            "compare_strongsort_pt.py",
            video_path,
            tracker_name
        ],
        check=True
    )

elif mode == "2":

    subprocess.run(
        [
            sys.executable,
            "compare_strongsort_onnx.py",
            video_path,
            tracker_name
        ],
        check=True
    )

elif mode == "3":

    print("\nRunning PT Pipeline...\n")

    subprocess.run(
        [
            sys.executable,
            "compare_strongsort_pt.py",
            video_path,
            tracker_name
        ],
        check=True
    )

    print("\nRunning ONNX Pipeline...\n")

    subprocess.run(
        [
            sys.executable,
            "compare_strongsort_onnx.py",
            video_path,
            tracker_name
        ],
        check=True
    )

else:
    raise ValueError("Invalid Mode")

if mode == "3":

    with open(
        "outputs/benchmark_pt.json"
    ) as f:

        pt = json.load(f)

    with open(
        "outputs/benchmark_onnx.json"
    ) as f:

        onnx = json.load(f)

    pt_time = pt["performance"]["total_processing_time_seconds"]
    onnx_time = onnx["performance"]["total_processing_time_seconds"]

    pt_fps = pt["performance"]["average_fps"]
    onnx_fps = onnx["performance"]["average_fps"]

    winner_speed = (
        "PyTorch"
        if pt_time < onnx_time
        else "ONNX"
    )

    winner_fps = (
        "PyTorch"
        if pt_fps > onnx_fps
        else "ONNX"
    )

    overall = (
        "PyTorch"
        if (
            pt_time < onnx_time
            and
            pt_fps > onnx_fps
        )
        else "ONNX"
    )

    comparison = {

        "project": {
            "model": "YOLO11m",
            "tracker": tracker_name
        },

        "video": {
            "path": video_path
        },

        "pytorch": pt,

        "onnx": onnx,

        "comparison": {

            "processing_time_difference_sec":
                round(
                    abs(
                        pt_time - onnx_time
                    ),
                    2
                ),

            "fps_difference":
                round(
                    abs(
                        pt_fps - onnx_fps
                    ),
                    2
                ),

            "fastest_model":
                winner_speed,

            "highest_fps":
                winner_fps,

            "overall_best":
                overall
        }
    }

    comparison_file = (
        f"outputs/{tracker_name}_comparison.json"
    )

    with open(
        comparison_file,
        "w"
    ) as f:

        json.dump(
            comparison,
            f,
            indent=4
        )

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print(f"\nTracker Used : {tracker_name}")

    print(
        f"\nPT Time   : {pt_time:.2f}"
    )

    print(
        f"ONNX Time : {onnx_time:.2f}"
    )

    print(
        f"\nPT FPS    : {pt_fps:.2f}"
    )

    print(
        f"ONNX FPS  : {onnx_fps:.2f}"
    )

    print(
        f"\nBest Model : {overall}"
    )

    print(
        f"\nComparison Saved:"
    )

    print(
        comparison_file
    )