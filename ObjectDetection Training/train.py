from ultralytics import YOLO

# pip install ultralytics==8.2.59

# Load a YOLOv8 model
model = YOLO('yolov8n.pt')  # choose a model variant like yolov8s.pt, yolov8m.pt, etc.

# Train the model
model.train(data='data.yaml', epochs=100, imgsz=640, batch=16, amp=False)

# Save the trained model to a .pt file
model_path = 'standard_errors_objdetection2.pt'
model.save(model_path)
print(f"Model saved to {model_path}")