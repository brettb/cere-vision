import cv2
from ultralytics import YOLO
import random
import os
import sys
import argparse


def predict(chosen_model, img, classes=[], conf=0.5, task='detect'):
    """
    Perform object detection or pose estimation on an image using the YOLO model.
    
    Args:
        chosen_model: The YOLO model to use for prediction
        img: Input image for detection
        classes: List of class names to detect (empty list means detect all classes)
        conf: Confidence threshold (0.0-1.0) - higher values mean more confident detections
        task: Type of task ('detect' or 'pose')
    
    Returns:
        Detection results
    """
    if task == 'detect' and classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=2, task='detect'):
    """
    Perform object detection or pose estimation and draw bounding boxes/keypoints on the image.
    
    Args:
        chosen_model: The YOLO model to use for prediction
        img: Input image for detection
        classes: List of class names to detect (empty list means detect all classes)
        conf: Confidence threshold (0.0-1.0) - higher values mean more confident detections
        rectangle_thickness: Thickness of bounding box lines (adjustable)
        text_thickness: Thickness of class label text (adjustable)
        task: Type of task ('detect' or 'pose')
    
    Returns:
        Tuple of (annotated image, detection results)
    """
    results = predict(chosen_model, img, classes, conf, task)
    
    # Create a copy of the image to draw on
    annotated_img = img.copy()
    
    if task == 'pose':
        # For pose estimation, use the built-in plot method which draws keypoints and connections
        annotated_img = results[0].plot()
    else:
        # For object detection, draw custom bounding boxes
        colors = {}
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Generate a random color for each class if not already generated
                class_id = int(box.cls[0])
                if class_id not in colors:
                    colors[class_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
                # Draw bounding box
                cv2.rectangle(annotated_img, 
                              (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), 
                              colors[class_id], rectangle_thickness)
                
                # Draw class label
                cv2.putText(annotated_img, f"{result.names[class_id]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, colors[class_id], text_thickness)
    
    return annotated_img, results


def create_video_writer(video_cap, output_filename):
    """
    Create a VideoWriter object for saving processed video.
    
    Args:
        video_cap: OpenCV VideoCapture object
        output_filename: Path to save the output video
        
    Returns:
        OpenCV VideoWriter object
    """
    # Get video properties
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))
    return writer


def process_video(input_path, output_path, model_name, task='detect', conf=0.5, classes=None):
    """
    Process a video with YOLO model for detection or pose estimation.
    
    Args:
        input_path: Path to input video
        output_path: Path to save output video
        model_name: Name of the YOLO model to use
        task: Type of task ('detect' or 'pose')
        conf: Confidence threshold
        classes: List of classes to detect (None means all classes)
    """
    # Define the models directory path
    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo-models")
    
    # Model path construction
    model_path = os.path.join(MODELS_DIR, model_name)
    
    print(f"Input video: {input_path}")
    print(f"Output video: {output_path}")
    print(f"Model path: {model_path}")
    print(f"Task: {task}")
    
    # Check if input video exists
    if not os.path.exists(input_path):
        print(f"ERROR: Input video file not found: {input_path}")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load the model
    model = YOLO(model_path)
    
    # Open the video
    cap = cv2.VideoCapture(input_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"ERROR: Failed to open video: {input_path}")
        return False
    
    print(f"Video opened successfully. Width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}, Height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # Create video writer
    writer = create_video_writer(cap, output_path)
    
    # Process the video
    frame_count = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        
        if not ret:
            print(f"End of video or error reading frame. Processed {frame_count} frames.")
            break
        
        frame_count += 1
        if frame_count % 10 == 0:  # Print status every 10 frames
            print(f"Processing frame {frame_count}")
        
        # Run YOLO inference on the frame
        result_img, _ = predict_and_detect(model, frame, classes=classes, conf=conf, task=task)
        
        # Write the frame to the output video
        writer.write(result_img)
        cv2.imshow("Image", result_img)
        
        # ADJUSTABLE: Change the wait time to control playback speed
        # Lower values = faster playback, higher values = slower playback
        wait_time = 1
        
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    
    print("Releasing video writer and capture...")
    # Release the video capture and writer objects
    cap.release()
    writer.release()
    print(f"Video processing complete. Output saved to: {output_path}")
    cv2.destroyAllWindows()
    return True


def main():
    parser = argparse.ArgumentParser(description="Process video with YOLO for detection or pose estimation")
    parser.add_argument("--input", type=str, default="video/boxes.mp4", help="Input video path")
    parser.add_argument("--output", type=str, default="runs/output.mp4", help="Output video path")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Model name (e.g., yolo11m.pt, yolo11n-pose.pt)")
    parser.add_argument("--task", type=str, choices=["detect", "pose"], default="detect", help="Task type: detect or pose")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--classes", type=str, nargs="*", help="Classes to detect (space-separated)")
    
    args = parser.parse_args()
    
    # Process the video
    process_video(
        args.input,
        args.output,
        args.model,
        args.task,
        args.conf,
        args.classes
    )


if __name__ == "__main__":
    main()