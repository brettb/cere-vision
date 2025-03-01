# Cere Vision: YOLO-based Computer Vision Toolkit

A comprehensive toolkit for computer vision tasks using YOLO (You Only Look Once) models, including object detection, pose estimation, segmentation, and more.

## Features
- **Object Detection**: Detect and classify objects in images and videos
- **Pose Estimation**: Track human body keypoints and poses
- **Segmentation**: Perform pixel-level segmentation of objects
- **Classification**: Classify images into categories
- **Training**: Train custom YOLO models on your own datasets

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/brettb/cere-vision.git
   cd cere-vision
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Model Organization

All YOLO model files (.pt) are stored in the `yolo-models` directory. The scripts automatically reference this directory when loading models.

### Downloading Models

Use the provided `download_yolo_model.py` script to download models directly into the correct directory:

```bash
# List available models
python download_yolo_model.py --list

# Download a specific model
python download_yolo_model.py --model yolo11n.pt
```

## Usage

### Photo Processing

The `yolo11-photo.py` script allows you to perform object detection on images:

```bash
# Basic usage with default settings
python3 yolo11-photo.py

# Adjust parameters in the script:
# - Change the model (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)
# - Modify confidence threshold
# - Filter specific classes
# - Adjust bounding box thickness and text thickness
```

### Video Processing

The `yolo11-video.py` script processes videos with object detection or pose estimation:

```bash
# Basic object detection with default settings
python3 yolo11-video.py --input video/your_video.mp4 --output runs/output.mp4

# Pose estimation
python3 yolo11-video.py --input video/your_video.mp4 --output runs/pose_output.mp4 --model yolo11n-pose.pt --task pose

# With custom confidence threshold
python3 yolo11-video.py --input video/your_video.mp4 --output runs/output.mp4 --conf 0.7

# To detect only specific classes
python3 yolo11-video.py --input video/your_video.mp4 --output runs/output.mp4 --classes person car
```

### Training

The `yolo11-train.py` script allows you to train YOLO models on custom datasets:

```bash
# Basic usage with default settings
python3 yolo11-train.py

# The script includes parameters you can modify:
# - Model selection (nano, small, medium, large, extra-large)
# - Dataset configuration
# - Number of epochs
# - Image size
```

## YOLO Model Types

The toolkit supports various YOLO model types:

| Model Size | Description | Best Use Case |
|------------|-------------|---------------|
| Nano (n)   | Smallest and fastest, lower accuracy | Real-time applications on limited hardware |
| Small (s)  | Good balance of speed and accuracy | Resource-constrained environments |
| Medium (m) | Balanced performance | Most general applications |
| Large (l)  | Higher accuracy, slower | Applications where precision is important |
| Extra Large (x) | Highest accuracy, slowest | Offline processing where accuracy is critical |

### Task Types

Each model can be specialized for different tasks:

| Task Type | Suffix | Description |
|-----------|--------|-------------|
| Detection | (none) | Standard object detection |
| Segmentation | -seg | Pixel-level object segmentation |
| Pose | -pose | Human pose estimation |
| Classification | -cls | Image classification |
| Oriented Bounding Box | -obb | Rotated bounding boxes |

Example: `yolo11n-pose.pt` is a nano-sized model specialized for pose estimation.

## Directory Structure

```
cere-vision/
├── yolo-models/         # YOLO model files (.pt)
├── photos/              # Sample images for testing
├── video/               # Sample videos for testing
├── runs/                # Output directory for processed videos
├── docs/                # Documentation
├── yolo11-photo.py      # Script for image processing
├── yolo11-video.py      # Script for video processing
├── yolo11-train.py      # Script for model training
├── download_yolo_model.py # Utility for downloading models
└── requirements.txt     # Project dependencies
```

## Requirements

- Python 3.10+
- OpenCV 4.9.0+
- Ultralytics 8.1.27+
- PyTorch 2.5.1+
- Additional dependencies listed in requirements.txt

## License

[Specify your license here]

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO implementation
- [OpenCV](https://opencv.org/) for computer vision utilities