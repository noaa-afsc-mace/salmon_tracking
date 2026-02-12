"""
Makes MOT visualizations
"""

import os
import sys
from pathlib import Path
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracking.utils.visualize_mot import load_data, visualize_MOT, visualize_MOT_quad, visualize_MOT_dual

# ---- hey look at this, it's important: ----

TRACKERS = ["botsort", "bytetrack", "ioutrack", "centroidtrack"]
# TRACKERS = ["botsort"]
FPS = 30.0
DETECTIONS_SOURCE = "model_type-yolo12x_data_source-clip_based_2_class_2_11_25_training_epochs-500_hyperparameter_source-default"

# ---- (told you it was important) ----

MODE = "test"
CHALLENGE_NAME = f"MOTFish_{FPS}"

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

# Expand user paths and format any placeholders
video_path = os.path.expanduser(config.get('video_clips', '')).replace("{FPS}", str(FPS))
detection_data_path = os.path.expanduser(config.get('detection_data', ''))
experiment_visualization_path = os.path.expanduser(config.get('experiment_visualization_path', ''))
experiment_path_root = "final_results_2026/mot_challenge"
gt_data_path = os.path.expanduser(config.get('gt_data_path', ''))

# handle frame rate stuff
if FPS != 30.0:
    DETECTIONS_SOURCE = f"fps-{FPS}_{DETECTIONS_SOURCE}"

def visualize_trackers_quad():
    """
    Visualizes all four trackers in a single frame for all clips
    """

    detections_dir = os.path.join(detection_data_path, DETECTIONS_SOURCE)
    seqmap_path = os.path.join(gt_data_path, "seqmaps", f"{CHALLENGE_NAME}-{MODE}.txt")
    with open(seqmap_path, "r") as seqmap:
        # skip first line
        seqmap.readline()
        vids = [line.strip() for line in seqmap.readlines()]
    seqmap.close()

    # loop through videos in detection source
    all_detection_dirs = [entry.name for entry in os.scandir(detections_dir) if entry.is_dir()]
    vids_to_process = list(set(all_detection_dirs).intersection(vids))
    
    print(f"Visualizing results. Saving to {experiment_visualization_path}")
    for i in tqdm(range(len(vids_to_process))):
        clip = vids_to_process[i]
        vid_path = os.path.join(video_path, clip + ".avi")
        gt_data = load_data(os.path.join(gt_data_path, f"{CHALLENGE_NAME}-{MODE}",clip, "gt/gt.txt"))
        save_path = os.path.join(experiment_visualization_path, f"quad_{clip}_visualization.mov")
        tracker_data = []
        for tracker in TRACKERS:
            # get data
            experiment_name = f"{tracker}-{DETECTIONS_SOURCE}"
            experiment_path = os.path.join(experiment_path_root, f"{CHALLENGE_NAME}-{MODE}", experiment_name)
            experiment_data_path = os.path.join(experiment_path, "data")

            tracker_data.append(load_data(os.path.join(experiment_data_path, clip + ".txt"))) 
    
        visualize_MOT_quad(vid_path, gt_data, tracker_data[0], TRACKERS[0], tracker_data[1], TRACKERS[1], \
                            tracker_data[2], TRACKERS[2], tracker_data[3], TRACKERS[3], save_path)

def visualize_trackers_dual(experiment1, exp1_title, experiment2, exp2_title):
    """
    Visualizes two trackers in a single frame for all clips

    Args:
    experiment1: name of experiment, from tracking/trackers/mot_challenge/MOTFish_<fps>-test
    exp1_title: title to use for first experiment in video
    experiment2: 2nd experiment
    exp2_title: Title for 2nd experiment
    """

    detections_dir = os.path.join(detection_data_path, DETECTIONS_SOURCE)
    seqmap_path = os.path.join(gt_data_path, "seqmaps", f"{CHALLENGE_NAME}-{MODE}.txt")
    with open(seqmap_path, "r") as seqmap:
        # skip first line
        seqmap.readline()
        vids = [line.strip() for line in seqmap.readlines()]
    seqmap.close()

    # loop through videos in detection source
    all_detection_dirs = [entry.name for entry in os.scandir(detections_dir) if entry.is_dir()]
    vids_to_process = list(set(all_detection_dirs).intersection(vids))
    
    print(f"Visualizing results. Saving to {experiment_visualization_path}")
    for i in tqdm(range(len(vids_to_process))):
        clip = vids_to_process[i]
        vid_path = os.path.join(video_path, clip + ".avi")
        gt_data = load_data(os.path.join(gt_data_path, f"{CHALLENGE_NAME}-{MODE}",clip, "gt/gt.txt"))
        save_path = os.path.join(experiment_visualization_path, f"dual_{clip}_visualization_{exp1_title}_{exp2_title}.mov")

        exp1_path = os.path.join(experiment_path_root, f"{CHALLENGE_NAME}-{MODE}", experiment1)
        exp2_path = os.path.join(experiment_path_root, f"{CHALLENGE_NAME}-{MODE}", experiment2)
        exp1_data_path = os.path.join(exp1_path, "data")
        exp2_data_path = os.path.join(exp2_path, "data")

        exp1_data = load_data(os.path.join(exp1_data_path, clip + ".txt"))
        exp2_data = load_data(os.path.join(exp2_data_path, clip + ".txt"))
    
        visualize_MOT_dual(vid_path, gt_data, exp1_data, exp1_title, exp2_data, exp2_title, save_path)

def visualize_trackers_single(default_settings):
    """
    For each tracker, visualizes results for the tracker for all clips
    """

    detections_dir = os.path.join(detection_data_path, DETECTIONS_SOURCE)
    seqmap_path = os.path.join(gt_data_path, "seqmaps", f"{CHALLENGE_NAME}-{MODE}.txt")

    with open(seqmap_path, "r") as seqmap:
        # skip first line
        seqmap.readline()
        vids = [line.strip() for line in seqmap.readlines()]
    seqmap.close()

    # loop through videos in detection source
    all_detection_dirs = [entry.name for entry in os.scandir(detections_dir) if entry.is_dir()]
    vids_to_process = list(set(all_detection_dirs).intersection(vids))
    
    print(f"Visualizing results. Saving to {experiment_visualization_path}")

    for tracker in TRACKERS:
        experiment_name = f"{tracker}-{DETECTIONS_SOURCE}"
        if default_settings:
            experiment_name = f"default_settings-{experiment_name}"
        experiment_path = os.path.join(experiment_path_root, f"{CHALLENGE_NAME}-{MODE}", experiment_name)
        experiment_data_path = os.path.join(experiment_path, "data")
        vis_save_path = os.path.join(experiment_visualization_path, experiment_name)

        print(f"Visualizing results. Saving to {vis_save_path}")
        if os.path.exists(vis_save_path):
            print(f"Experiment {vis_save_path} already exists. Overwriting")
            pass
        else:
            os.mkdir(vis_save_path)

        for i in tqdm(range(len(vids_to_process))):
            clip = vids_to_process[i]
            vid_path = os.path.join(video_path, clip + ".avi")
            gt_data = load_data(os.path.join(gt_data_path, f"{CHALLENGE_NAME}-{MODE}",clip, "gt/gt.txt"))
            save_path = os.path.join(vis_save_path, f"{clip}_{tracker}_visualization.mov")

            track_data = load_data(os.path.join(experiment_data_path, clip + ".txt"))
            visualize_MOT(vid_path, gt_data, track_data, save_path)

if __name__ == "__main__":
    visualize_trackers_quad()
    # visualize_trackers_dual("bytetrack-model_type-yolo12x_data_source-clip_based_2_class_2_11_25_training_epochs-500_hyperparameter_source-default",
    #                         "bytetrack optimized",
    #                    "default_settings-bytetrack-model_type-yolo12x_data_source-clip_based_2_class_2_11_25_training_epochs-500_hyperparameter_source-default",
    #                    "bytetrack default")
    visualize_trackers_single(default_settings=False)
    