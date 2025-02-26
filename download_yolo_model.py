#!/usr/bin/env python3
"""
YOLO Model Downloader

This script downloads YOLO models and saves them to the yolo-models directory.
It ensures consistent organization of model files in the project.

Usage:
    python download_yolo_model.py --model yolo11n.pt
    python download_yolo_model.py --model yolo11n-seg.pt
    python download_yolo_model.py --model yolo11x.pt
"""

import os
import argparse
from ultralytics import YOLO
import torch

def download_model(model_name):
    """
    Download a YOLO model and save it to the yolo-models directory.
    
    Args:
        model_name: Name of the model to download (e.g., 'yolo11n.pt')
    """
    # Define the models directory path
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo-models")
    
    # Create the models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Full path where the model will be saved
    model_path = os.path.join(models_dir, model_name)
    
    print(f"Downloading {model_name} to {model_path}...")
    
    try:
        # Download the model
        model = YOLO(model_name)
        
        # Check if the model was downloaded successfully
        if os.path.exists(model.ckpt_path):
            # Copy the model to our models directory
            if model.ckpt_path != model_path:
                torch.save(model.model.state_dict(), model_path)
                print(f"Model saved to: {model_path}")
            else:
                print(f"Model already exists at: {model_path}")
        else:
            print(f"Error: Failed to download model {model_name}")
    
    except Exception as e:
        print(f"Error downloading model: {e}")

def list_available_models():
    """List commonly available YOLO models"""
    print("\nAvailable YOLO models:")
    print("Detection models:")
    print("  - yolo11n.pt (Nano - Smallest and fastest)")
    print("  - yolo11s.pt (Small - Good balance of speed and accuracy)")
    print("  - yolo11m.pt (Medium - Balanced performance)")
    print("  - yolo11l.pt (Large - Higher accuracy)")
    print("  - yolo11x.pt (Extra Large - Highest accuracy)")
    
    print("\nSegmentation models:")
    print("  - yolo11n-seg.pt (Nano - Segmentation)")
    print("  - yolo11s-seg.pt (Small - Segmentation)")
    print("  - yolo11m-seg.pt (Medium - Segmentation)")
    print("  - yolo11l-seg.pt (Large - Segmentation)")
    print("  - yolo11x-seg.pt (Extra Large - Segmentation)")
    
    print("\nPose models:")
    print("  - yolo11n-pose.pt (Nano - Pose estimation)")
    print("  - yolo11s-pose.pt (Small - Pose estimation)")
    print("  - yolo11m-pose.pt (Medium - Pose estimation)")
    print("  - yolo11l-pose.pt (Large - Pose estimation)")
    print("  - yolo11x-pose.pt (Extra Large - Pose estimation)")
    
    print("\nClassification models:")
    print("  - yolo11n-cls.pt (Nano - Classification)")
    print("  - yolo11s-cls.pt (Small - Classification)")
    print("  - yolo11m-cls.pt (Medium - Classification)")
    print("  - yolo11l-cls.pt (Large - Classification)")
    print("  - yolo11x-cls.pt (Extra Large - Classification)")

def main():
    parser = argparse.ArgumentParser(description="Download YOLO models to the yolo-models directory")
    parser.add_argument("--model", type=str, help="Model name to download (e.g., yolo11n.pt)")
    parser.add_argument("--list", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    if not args.model:
        print("Error: Please specify a model name with --model or use --list to see available models")
        return
    
    download_model(args.model)

if __name__ == "__main__":
    main()
