""" 
Runs detections

usage: 
predict.py [-h] [MODEL_PATH] [VIDEO_DIR] [SAVE_DIR]

arguments:
-h, --help            
                Show this help message and exit.
MODEL_PATH
                Path to object detection model
VIDEO_DIR
                Path to directory containing videos
SAVE_PATH
                Path to save detections
"""

from ultralytics import YOLO
import argparse
import os
import glob

# Initiate argument parser
parser = argparse.ArgumentParser(description="Generate detections")
parser.add_argument(
    "model_path",
    help="Path to object detection model",
    type=str,
)
parser.add_argument(
    "video_dir",
    help="Path to directory containing videos",
    type=str,
)
parser.add_argument(
    "save_dir",
    help="Path to directory to save detections",
    type=str,
)

def predict(model_path, video_dir, save_dir):
    model_name = model_path.split("/")[-3]
    # set up data dirs
    detection_data_path = os.path.join(save_dir, model_name)
    if not os.path.exists(detection_data_path):
        os.makedirs(detection_data_path)
    
    model = YOLO(model_path)
    # run predict for each video
    # NOTE: you can actually just call predict on the entire directory but all results will be saved in a single folder. Which is ridiculous
    file_extensions = ["*.MOV", "*.mov", "*.avi"]
    video_paths = []
    for ext in file_extensions:
        video_paths += glob.glob(os.path.join(video_dir, ext))
    for video in video_paths:
        print(f"Running detection on {video} ...")
        video_name = os.path.basename(video).split(".")[0]
        model.predict(source=video, device=[0], project=detection_data_path, name=video_name, conf=0.0, save_txt=True, save_conf=True, verbose=False)

if __name__ == "__main__":
    args = parser.parse_args()
    model_path = args.model_path
    video_dir = args.video_dir
    save_dir = args.save_dir

    predict(model_path, video_dir, save_dir)
