from ultralytics import YOLO
import os

STAIRS_WEIGHTS = r"C:\Users\User\Desktop\strider_vision\runs\detect\train3\weights\best.pt"

if not os.path.exists(STAIRS_WEIGHTS):
    raise FileNotFoundError(STAIRS_WEIGHTS)

model = YOLO(STAIRS_WEIGHTS)

model.export(
    format="onnx",
    opset=12,
    imgsz=(192, 320),
    simplify=True,
    dynamic=False,
    verbose=True
)
print("Exported: stairs.onnx")
