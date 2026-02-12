""""
Visualization functions for debugging and interpretation of MOT

usage: 
visualize_mot.py [-h] [MOT_ANNOTATION_DIR] [VIDEO_DIR] [SAVE_DIR]

arguments:
    -h, --help            
                    Show this help message and exit.
    MOT_ANNOTATION_DIR
                    Path to directory containing MOT annotations
    VIDEO_DIR
                    Path to directory containing videos
    SAVE_DIR
                    Path to directory where MOT annotations will be saved
"""

import cv2
import os
import argparse

# Initiate argument parser
parser = argparse.ArgumentParser(description="Visualizes MOT annotations")
parser.add_argument(
    "mot_annotation_dir",
    help="Path to directory containing MOT annotations",
    type=str,
)
parser.add_argument(
    "video_dir",
    help="Path to directory containing videos",
    type=str,
)
parser.add_argument(
    "save_dir",
    help="Path to directory where visualizations will be saved",
    type=str,
)
args = parser.parse_args()

def load_data(file_path):
    """
    Reads text file containing data in the form:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z> 
    and returns dict

    Args:
        file_path (str): path to MOT txt file
    Returns:
        data (dict): dict of data of form {<frame>: [[<id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>], ...], ...}
    """
    data = {}
    with open(file_path) as f:
        lines = f.readlines()
    for l in lines:
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, *rest = l.split(",")
        if int(frame) not in data:
            data[int(frame)] = [[int(id), int(float(bb_left)), int(float(bb_top)), int(float(bb_width)), int(float(bb_height)), float(conf)]]
        else:
            data[int(frame)].append([int(id), int(float(bb_left)), int(float(bb_top)), int(float(bb_width)), int(float(bb_height)), float(conf)])
    return data

def visualize_MOT(
    video_file,
    gt_data,
    save_path
):
    """
    stuff and things
    visualize single video, gets paths for everything it needs

    Args:
        video_file (str): path to video file
        gt_data (dict): dict of ground truth data of form {<frame>: [[<id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>], ...], ...}
        save_path (str): path to save video
    Returns:
        None
    """
    
    # load tracks and stuff
    
    frame_num = 1
    vid = cv2.VideoCapture(video_file)
    success, frame = vid.read()
    vid_fps = vid.get(cv2.CAP_PROP_FPS)
    vid_height, vid_width, _ = frame.shape
    output_video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"jpeg"), vid_fps, (vid_width, vid_height))
    

    while success:
        draw_frame = visualize_MOT_boxes(frame, frame_num, gt_data)
        output_video.write(draw_frame)
        success, frame = vid.read()
        frame_num += 1
    vid.release()
    output_video.release()

def visualize_MOT_boxes(frame, frame_num, gt_full_data):
    """
    Draws stuff and things for the given frame
    data has all data but some frames don't exist

    Args:
        frame (np.array): frame to draw on
        frame_num (int): frame number
        gt_full_data (dict): dict of ground truth data of form {<frame>: [[<id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>], ...], ...}
    Returns:
        frame (np.array): frame with stuff drawn on it
    """

    if frame_num not in gt_full_data:
        gt_frame_data = []
    else:
        gt_frame_data = gt_full_data[frame_num]

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    gt_color = (0, 255, 0)  # in BGR

    for gt_data in gt_frame_data:
        id, bb_left, bb_top, bb_width, bb_height, conf = gt_data
        frame = cv2.rectangle(
            frame,
            (bb_left, bb_top),
            (bb_width+bb_left, bb_height+bb_top),
            gt_color,
            5,
        )
        # put id in center of bbox
        # Get the size of the text
        text_size, _ = cv2.getTextSize(str(id), cv2.FONT_HERSHEY_SIMPLEX, 1, 5)

        # Calculate the position of the text
        text_x = bb_left + int((bb_width - text_size[0]) / 2)
        text_y = bb_top + int((bb_height + text_size[1]) / 2)

        # Add the text to the image
        cv2.putText(frame, str(id), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)

    frame = cv2.putText(
        frame,
        "Ground truth",
        (10, 30),
        font,
        fontScale,
        gt_color,
        2,
        cv2.LINE_AA)

    return frame

def visualize_all(annotation_dir, video_dir, save_dir):
    """
    Visualizes all MOT annotations in a directory
    
    Args:
        annotation_dir (str): path to directory containing MOT annotations
        save_dir (str): path to directory where visualizations will be saved
    """

    for name in os.listdir(annotation_dir):
        clip_path = os.path.join(annotation_dir, name)
        if os.path.isdir(clip_path):
            mot_path = os.path.join(clip_path, "gt", "gt.txt")
            video_path = os.path.join(video_dir, f"{name}.avi")
            save_path = os.path.join(save_dir, f"{name}.avi")
            print(f"Visualizing sequence: {name}")
            visualize_MOT(video_path, load_data(mot_path), save_path)
    print("Done")

if __name__ == "__main__":
    # parse arguments
    annotation_dir = args.mot_annotation_dir
    video_dir = args.video_dir
    save_dir = args.save_dir
    visualize_all(annotation_dir, video_dir, save_dir)