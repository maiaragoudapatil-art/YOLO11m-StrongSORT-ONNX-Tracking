from ultralytics import YOLO
import time
import json

video = r"C:\Users\VijaySegunasi\strongsort_yolo\videos\PNNL_Parking_LOT(1).avi"

# PT MODEL
print("Loading PT model...")
pt_model = YOLO("yolo11m.pt")
start = time.time()
count = 0

for _ in pt_model.predict(video, stream=True, verbose=False):
    count += 1
    if count % 100 == 0:
        print(f"PT Processed {count} frames")

pt_time = time.time() - start
print(f"PT Finished. Time = {pt_time:.2f}s")

# ONNX MODEL
print("Loading ONNX model...")

onnx_model = YOLO("yolo11m.onnx")
start = time.time()
count = 0

for _ in onnx_model.predict(video, stream=True, verbose=False):
    count += 1
    if count % 100 == 0:
        print(f"ONNX Processed {count} frames")
onnx_time = time.time() - start

print(f"ONNX Finished. Time = {onnx_time:.2f}s")

# FPS
total_frames = count

pt_fps = total_frames / pt_time
onnx_fps = total_frames / onnx_time

# JSON OUTPUT
results = {
    "project": {
        "model": "YOLO11m",
        "comparison": "PyTorch vs ONNX"
    },
    "video": {
        "input": video,
        "total_frames": total_frames
    },
    "pytorch": {
        "model": "yolo11m.pt",
        "processing_time_seconds": pt_time,
        "average_fps": pt_fps
    },
    "onnx": {
        "model": "yolo11m.onnx",
        "processing_time_seconds": onnx_time,
        "average_fps": onnx_fps
    }
}

with open(
    "outputs/pt_vs_onnx_metrics.json",
    "w"
) as f:
    json.dump(results, f, indent=4)

print("\n===== RESULTS =====")
print(f"PT Time   : {pt_time:.2f} sec")
print(f"PT FPS    : {pt_fps:.2f}")
print(f"ONNX Time : {onnx_time:.2f} sec")
print(f"ONNX FPS  : {onnx_fps:.2f}")
print("\nJSON Saved:")
print("outputs/pt_vs_onnx_metrics.json")
