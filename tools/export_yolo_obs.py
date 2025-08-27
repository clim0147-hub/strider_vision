from ultralytics import YOLO

# Load pretrained YOLOv8n on COCO
model = YOLO("yolov8n.pt")

# Export to ONNX
model.export(
    format="onnx",
    opset=12,
    imgsz=(192, 320),   # match your pipeline size (HxW)
    simplify=True,
    dynamic=False,
    verbose=True
)
print("Exported: yolov8n.onnx")