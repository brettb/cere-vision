To download your first model and determine the results, you can quickly run the default commands for both photos and videos, Just change the file names and paths for use with your personal photos and videos.

First clone the cere-vision repository:
```bash
git clone https://github.com/brettb/cere-vision.git
cd cere-vision
```

Next, run the default command to process a photo from the cere-vision directory to verify everything is working for you:

```bash
python3 yolo11-photo.py
```

Now run the optional command to process a video from the cere-vision directory:

```bash
python3 yolo11-video.py
```

I've added a very basic Demo UI to the project as an example of how the class selection works from a live camera feed.
```bash
python3 demo-gui.py
```

Now modify the prompts with your personal media to check the results.

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
