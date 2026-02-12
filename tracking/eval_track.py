"""
Runs full tracker evaluation using params saved in `optimal_params.yaml`.
Results are saved in tracking/trackers/mot_challenge.

Usage:
    python tracking/eval_track.py [eval_jobs.yaml]

    If no argument is given, defaults to `tracking/eval_jobs.yaml`.
    To run all jobs in parallel, use `tracking/launch_all_evals.sh`,
    which generates per-window YAML configs and launches them in tmux.

Job configuration (trackers, models, fps, params_source) is defined in the
YAML file. See `eval_jobs.yaml` for the format and `launch_all_evals.sh`
for bulk parallel execution.

NOTE: Running tracker evaluation could change figures and tables
"""

import sys
import numpy as np
import os
import csv
import yaml

from utils.tracking_utils import run_tracks, get_tracker_args

# --- load configs ---
with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

job_config_path = sys.argv[1] if len(sys.argv) > 1 else "tracking/eval_jobs.yaml"
with open(job_config_path, 'r') as f:
    job_config = yaml.safe_load(f)

# paths from config
ultra_path = os.path.expanduser(config.get('ultralytics', ''))
trackeval_path = os.path.expanduser(config.get('trackeval', ''))
detection_data_path = os.path.expanduser(config.get('detection_data', ''))
optimal_params_path = os.path.expanduser(config.get('optimal_params_path', ''))

# job settings
TRACKERS = job_config["trackers"]
MODE = job_config["mode"]
DETECTION_SOURCE_TEMPLATE = job_config["detection_source_template"]

CLASSES_TO_EVAL = {"salmon": 0}

# import custom ultralytics with tracking functionality
sys.path.insert(0, ultra_path)
from ultralytics import YOLO

# import trackeval repo as package
sys.path.insert(0, trackeval_path)
import trackeval

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
GT_DATA_PATH = os.path.join(SCRIPT_DIR, "gt/mot_challenge/")
# params each tracker
TRACKER_ARGS = yaml.safe_load(open(optimal_params_path))


def main():
    """
    Iterates over jobs (model+fps), then trackers, running evaluation for each.
    """
    for job in job_config["jobs"]:
        model_name = job["model"]
        fps_list = job["fps"]
        params_source = job.get("params_source", "train_optimized")
        detection_source_base = DETECTION_SOURCE_TEMPLATE.format(model=model_name)

        for fps in fps_list:
            # compute paths for this job
            model_path = os.path.expanduser(config.get('model_weights', '')).replace("{DETECTIONS_SOURCE}", detection_source_base)
            video_path = os.path.expanduser(config.get('video_clips', '')).replace("{FPS}", str(fps))

            # handle frame rate prefix
            detection_source = detection_source_base
            if fps != 30.0:
                detection_source = f"fps-{fps}_{detection_source}"

            challenge_name = f"MOTFish_{fps}"

            for tracker_name in TRACKERS:
                print(f"\n{'='*60}")
                print(f"Job: {detection_source} | Tracker: {tracker_name} | Params: {params_source}")
                print(f"{'='*60}")

                # build experiment name based on params_source
                if params_source == "default":
                    experiment_name = f"default_settings-{tracker_name}-{detection_source}"
                elif params_source == "test_optimized":
                    experiment_name = f"test_optimized-{tracker_name}-{detection_source}"
                else:
                    experiment_name = f"{tracker_name}-{detection_source}"

                experiment_path_root = os.path.join(SCRIPT_DIR, "trackers/mot_challenge/")
                experiment_path = os.path.join(experiment_path_root, f"{challenge_name}-{MODE}", experiment_name)
                experiment_data_path = os.path.join(experiment_path, "data")

                # setup directories
                os.makedirs(experiment_data_path, exist_ok=True)

                # look up tracker params based on params_source
                if params_source == "default":
                    if "default" not in TRACKER_ARGS.get(tracker_name, {}):
                        print(f"WARNING: No 'default' params for {tracker_name}, skipping.")
                        continue
                    tracker_args = TRACKER_ARGS[tracker_name]["default"]
                elif params_source == "test_optimized":
                    tracker_args = TRACKER_ARGS[tracker_name][f"test_optimized-{detection_source}"]
                else:
                    tracker_args = TRACKER_ARGS[tracker_name][detection_source]

                detections_dir = os.path.join(detection_data_path, detection_source)

                # load YOLO model fresh per tracker (prevents result carryover)
                model = YOLO(model_path)

                eval_tracker(tracker_args, detections_dir, experiment_data_path,
                             experiment_path_root, experiment_name, challenge_name,
                             video_path, model)


def eval_tracker(smac_tracker_args, detections_dir, experiment_data_path,
                 experiment_path_root, experiment_name, challenge_name,
                 video_path, model):
    """
    Runs tracking and evaluation for a single tracker configuration.

    Args:
        smac_tracker_args: Dict of tracker args from optimal_params.yaml
        detections_dir: Directory containing all video detections
        experiment_data_path: Path to save experiment data
        experiment_path_root: Path to experiments root
        experiment_name: Name of experiment
        challenge_name: MOT challenge name (e.g. "MOTFish_30.0")
        video_path: Path to video clips
        model: Loaded YOLO model instance
    """
    tracker_args = get_tracker_args(dict(smac_tracker_args))

    # run tracking on all videos in repo of interest
    # get train/test vid names from seqmaps in gt folder
    seqmap_path = os.path.join(GT_DATA_PATH, "seqmaps", f"{challenge_name}-{MODE}.txt")
    with open(seqmap_path, "r") as seqmap:
        # skip first line
        seqmap.readline()
        vids = [line.strip() for line in seqmap.readlines()]

    # loop through videos in detection source
    all_detection_dirs = [entry.name for entry in os.scandir(detections_dir) if entry.is_dir()]
    vids_to_process = list(set(all_detection_dirs).intersection(vids))

    run_tracks(vids_to_process, video_path, experiment_data_path, detections_dir, tracker_args, model, list(CLASSES_TO_EVAL.values()))
    # run evaluator
    eval_config = trackeval.Evaluator.get_default_eval_config()
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config['GT_FOLDER'] = GT_DATA_PATH
    dataset_config['TRACKERS_FOLDER'] = experiment_path_root
    dataset_config['BENCHMARK'] = challenge_name
    dataset_config['DO_PREPROC'] = False
    dataset_config['CLASSES_TO_EVAL'] = list(CLASSES_TO_EVAL.keys())
    dataset_config['SPLIT_TO_EVAL'] = MODE
    dataset_config['TRACKERS_TO_EVAL'] = [experiment_name]
    metrics_config = {'METRICS': ['HOTA', 'IDF1', 'MOTA']}

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA(metrics_config)] + [trackeval.metrics.CLEAR(metrics_config)] + [trackeval.metrics.Identity(metrics_config)]
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
    save_res_table(output_res, os.path.abspath(os.path.join(experiment_data_path, os.pardir))) # save HOTA table

def save_res_table(output_res, experiment_path):
    """
    Saves results table
    """
    exp_name = os.path.basename(experiment_path)
    clips = list(output_res['MotChallenge2DBox'][exp_name].keys())
    metrics = list(output_res['MotChallenge2DBox'][exp_name][clips[0]]['salmon']['HOTA'].keys())
    data =[["sequence"] + metrics]
    for c in clips:
        row = [c]
        for m in metrics:
            score = np.mean(output_res['MotChallenge2DBox'][exp_name][c]['salmon']['HOTA'][m])
            row.append(score)
        data.append(row)

    # Write the table to the CSV file
    with open(os.path.join(experiment_path, "clip_data.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

if __name__ == "__main__":
    main()
