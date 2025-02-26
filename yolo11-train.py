from ultralytics import YOLO
import os

# Define the models directory path
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo-models")

# YOLO model options:
# - "yolo11n.pt": Nano model - Smallest and fastest, lower accuracy but best for real-time on limited hardware
# - "yolo11s.pt": Small model - Good balance of speed and accuracy for resource-constrained environments
# - "yolo11m.pt": Medium model - Balanced performance, recommended for most general applications
# - "yolo11l.pt": Large model - Higher accuracy, suitable for applications where precision is important
# - "yolo11x.pt": Extra Large model - Highest accuracy but slowest, best for offline processing where accuracy is critical

# Model path construction
model_name = "yolo11l.pt"
model_path = os.path.join(MODELS_DIR, model_name)

# Load a COCO-pretrained yolo11l model
model = YOLO(model_path)

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the yolo11l model on the 'bus.jpg' image
results = model("photos/bus.jpg")