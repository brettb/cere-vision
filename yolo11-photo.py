from ultralytics import YOLO
import cv2
import random
import os


def predict(chosen_model, img, classes=[], conf=0.5):
    """
    Perform object detection on an image using the YOLO model.
    
    Args:
        chosen_model: The YOLO model to use for prediction
        img: Input image for detection
        classes: List of class names to detect (empty list means detect all classes)
        conf: Confidence threshold (0.0-1.0) - higher values mean more confident detections
    
    Returns:
        Detection results
    """
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=2):
    """
    Perform object detection and draw bounding boxes on the image.
    
    Args:
        chosen_model: The YOLO model to use for prediction
        img: Input image for detection
        classes: List of class names to detect (empty list means detect all classes)
        conf: Confidence threshold (0.0-1.0) - higher values mean more confident detections
        rectangle_thickness: Thickness of bounding box lines (adjustable)
        text_thickness: Thickness of class label text (adjustable)
    
    Returns:
        Tuple of (annotated image, detection results)
    """
    results = predict(chosen_model, img, classes, conf=conf)
    yolo_classes = list(model.names.values())
    classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
    colors = [random.choices(range(256), k=3) for _ in classes_ids]

    for result in results:
        for box in result.boxes:
            color_number = classes_ids.index(int(box.cls[0]))
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), colors[color_number], rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, colors[color_number], text_thickness)
    return img, results


# Define the models directory path
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo-models")

# ADJUSTABLE: Change the model type based on your needs:
# - "yolo11n.pt": Nano model - Smallest and fastest, lower accuracy but best for real-time on limited hardware
# - "yolo11s.pt": Small model - Good balance of speed and accuracy for resource-constrained environments
# - "yolo11m.pt": Medium model - Balanced performance, recommended for most general applications
# - "yolo11l.pt": Large model - Higher accuracy, suitable for applications where precision is important
# - "yolo11x.pt": Extra Large model - Highest accuracy but slowest, best for offline processing where accuracy is critical

# Model path construction
model_name = "yolo11m.pt"  # Choose the model you want to use
model_path = os.path.join(MODELS_DIR, model_name)

# Load the model
model = YOLO(model_path)

# ADJUSTABLE: Change the input image path as needed
image = cv2.imread(r"photos/bus.jpg")

# ADJUSTABLE PARAMETERS:
# classes: List of specific classes to detect (empty list = detect all classes)
#          Example: classes=["person", "car"] will only detect people and cars
# conf: Confidence threshold (0.0-1.0) - higher values = more confident detections
#       Example: conf=0.7 will only show detections with 70% or higher confidence
# Additional parameters that can be adjusted:
# - rectangle_thickness: Thickness of bounding box lines (default=5)
# - text_thickness: Thickness of class label text (default=2)
result_img, _ = predict_and_detect(model, image, classes=[], conf=0.5)

# Display the result
cv2.imshow("Image", result_img)

# ADJUSTABLE: Change the output image path and filename
cv2.imwrite("YourSavePath.png", result_img)

# Wait for a key press to close the window
# ADJUSTABLE: Change 0 to a positive number to automatically close after that many milliseconds
cv2.waitKey(0)