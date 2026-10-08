from ultralytics import YOLO
#from boxmot import StrongSort

model = YOLO("yolo11m.pt")

model.predict(
    source="C:\\Users\\VijaySegunasi\\strongsort_yolo\\PNNL_Parking_LOT(1).avi",
    save=True
)