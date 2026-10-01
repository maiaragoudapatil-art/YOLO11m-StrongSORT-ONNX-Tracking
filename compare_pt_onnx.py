from ultralytics import YOLO
import time

video = r"C:\Users\VijaySegunasi\strongsort_yolo\videos\PNNL_Parking_LOT(1).avi"

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

print("\n===== RESULTS =====")
print(f"PT Time   : {pt_time:.2f} sec")
print(f"ONNX Time : {onnx_time:.2f} sec")
