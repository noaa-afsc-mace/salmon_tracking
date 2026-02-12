"""
Uses the SMAC3 optimization package to find optimal tracker parameters.
Search ranges for parameters are hard-coded in the `SEARCH_DICT`, update them if necessary.

Results for each tracker are date-tagged and saved in a .txt file in tracking/optimization/.
Contents are ordered by HOTA score.

Usage:
    python tracking/smac_optimize_track.py [job_config.yaml]

    If no argument is given, defaults to `tracking/optimization_jobs.yaml`.
    To run all jobs in parallel, use `tracking/launch_all_optimizations.sh`,
    which generates per-window YAML configs and launches them in tmux.

Job configuration (trackers, models, fps, etc.) is defined in the YAML file.
See `optimization_jobs.yaml` for the format and `launch_all_optimizations.sh`
for bulk parallel execution.

NOTE:
Requires our Ultralytics fork (https://github.com/mlurbur/ultralytics, branch: all_features)
and our TrackEval fork (https://github.com/noaa-afsc-mace/TrackEval).
For exact replication, use Ultralytics commit ae1ec96 and TrackEval commit bd72ef6.
See README.md for details.
"""

import sys
import time
import numpy as np
import os
from tabulate import tabulate
from ConfigSpace import ConfigurationSpace, Integer, Float, Constant, Categorical
from pathlib import Path
from smac import BlackBoxFacade, Scenario
import shutil
import yaml
import traceback

from utils.tracking_utils import run_tracks, get_tracker_args

# --- load configs ---
with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

job_config_path = sys.argv[1] if len(sys.argv) > 1 else "tracking/optimization_jobs.yaml"
with open(job_config_path, 'r') as f:
    job_config = yaml.safe_load(f)

# paths from config
ultra_path = os.path.expanduser(config.get('ultralytics', ''))
trackeval_path = os.path.expanduser(config.get('trackeval', ''))
detection_data_path = os.path.expanduser(config.get('detection_data', ''))
workers = int(config.get('num_workers', 1))

# job settings
TRACKERS = job_config["trackers"]
MODE = job_config["mode"]
NUM_SEARCH_ITERATIONS = job_config["num_search_iterations"]
DETECTION_SOURCE_TEMPLATE = job_config["detection_source_template"]

CLASSES_TO_EVAL = {"salmon": 0}
CLASS_OF_INTEREST = "salmon"

# import custom ultralytics with tracking functionality
sys.path.insert(0, ultra_path)
from ultralytics import YOLO

# import trackeval repo as package
sys.path.insert(0, trackeval_path)
import trackeval

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
OPTIMIZING_DATA_PATH = os.path.join(SCRIPT_DIR, "optimization")
EXPERIMENT_PATH_ROOT = os.path.join(SCRIPT_DIR, "trackers/mot_challenge/")
GT_DATA_PATH = os.path.join(SCRIPT_DIR, "gt/mot_challenge/")

# search bounds for each tracker
SEARCH_DICT = {
    "bytetrack": {
        "track_buffer":
            Integer("track_buffer", bounds=(1, 60)),
        "match_thresh":
            Float("match_thresh", bounds=(0.0001, 10.0)),
        "new_track_scaling_factor":
            Float("new_track_scaling_factor", bounds=(0.0, 1.0)),
        "track_low_thresh":
            Float("track_low_thresh", bounds=(0.0001, 1.0)),
        "track_high_scaling_factor":
            Float("track_high_scaling_factor", bounds=(0.0, 1.0)),
        "fuse_score":
            Categorical("fuse_score", [True, False]),
        "tracker_type":
            Constant("tracker_type", "bytetrack")
            },
    "botsort": {
        "track_buffer":
            Integer("track_buffer", bounds=(1, 60)),
        "match_thresh":
            Float("match_thresh", bounds=(0.0001, 10.0)),
        "new_track_scaling_factor":
            Float("new_track_scaling_factor", bounds=(0.0, 1.0)),
        "track_low_thresh":
            Float("track_low_thresh", bounds=(0.0001, 1.0)),
        "track_high_scaling_factor":
            Float("track_high_scaling_factor", bounds=(0.0, 1.0)),
        "gmc_method":
            Constant("gmc_method", "sparseOptFlow"),
        "with_reid":
            Constant("with_reid", False),
        "model": # reid related args must be provided even though they aren't used
            Constant("model", "auto"),
        "proximity_thresh": 
            Constant("proximity_thresh", 0.5),
        "appearance_thresh":
            Constant("appearance_thresh", 0.25),
        "fuse_score":
            Categorical("fuse_score", [True, False]),
        "tracker_type":
            Constant("tracker_type", "botsort"),
            },
    "ioutrack": {
        "track_buffer":
            Integer("track_buffer", bounds=(1, 60)),
        "match_thresh":
            Float("match_thresh", bounds=(0.0001, 1.0)),
        "track_thresh":
            Float("track_thresh", bounds=(0.0001, 1.0)),
        "tracker_type":
            Constant("tracker_type", "ioutrack")
            },
    "centroidtrack": {
        "track_buffer":
            Integer("track_buffer", bounds=(1, 60)),
        "match_thresh":
            Float("match_thresh", bounds=(0.0001, 10.0)),
        "track_thresh":
            Float("track_thresh", bounds=(0.0001, 1.0)),
        "tracker_type":
            Constant("tracker_type", "centroidtrack")
            },
}


def make_eval_fn(tracker_name, detection_source, video_path, model,
                 experiment_data_path, challenge_name, experiment_name):
    """
    Creates a closure for SMAC to call during optimization.
    Captures job-specific variables so we don't need globals.
    """
    def eval_tracker_for_optimization(smac_tracker_args, seed: int = 0):
        try:
            tracker_args = get_tracker_args(dict(smac_tracker_args))
            print(f"Eval for {tracker_name} with args: {dict(tracker_args)}")
            # get train/test vid names from seqmaps in gt folder
            seqmap_path = os.path.join(GT_DATA_PATH, "seqmaps", f"{challenge_name}-{MODE}.txt")
            with open(seqmap_path, "r") as seqmap:
                seqmap.readline()
                vids = [line.strip() for line in seqmap.readlines()]

            # loop through videos in detection source
            detections_dir = os.path.join(detection_data_path, detection_source)
            all_detection_dirs = [entry.name for entry in os.scandir(detections_dir) if entry.is_dir()]
            vids_to_process = sorted(set(all_detection_dirs).intersection(vids))

            run_tracks(vids_to_process, video_path, experiment_data_path,
                       os.path.join(detection_data_path, detection_source),
                       dict(tracker_args), model, list(CLASSES_TO_EVAL.values()))

            # run evaluator
            eval_config = trackeval.Evaluator.get_default_eval_config()
            eval_config['DISPLAY_LESS_PROGRESS'] = False
            eval_config['PRINT_RESULTS'] = False
            eval_config['PRINT_CONFIG'] = False
            eval_config['TIME_PROGRESS'] = False
            eval_config['PLOT_CURVES'] = False
            dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
            dataset_config['GT_FOLDER'] = GT_DATA_PATH
            dataset_config['TRACKERS_FOLDER'] = EXPERIMENT_PATH_ROOT
            dataset_config['BENCHMARK'] = challenge_name
            dataset_config['DO_PREPROC'] = False
            dataset_config['PRINT_CONFIG'] = False
            dataset_config['CLASSES_TO_EVAL'] = list(CLASSES_TO_EVAL.keys())
            dataset_config['SPLIT_TO_EVAL'] = MODE
            dataset_config['TRACKERS_TO_EVAL'] = [experiment_name]
            metrics_config = {'METRICS': ['HOTA'], 'THRESHOLD': 0.5}

            evaluator = trackeval.Evaluator(eval_config)
            dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
            metrics_list = [trackeval.metrics.HOTA(metrics_config)]
            output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)

            hota = np.mean(output_res["MotChallenge2DBox"][experiment_name]["COMBINED_SEQ"][CLASS_OF_INTEREST]["HOTA"]["HOTA"])
            print(f"score: {1-hota}")
            return 1-hota
        except Exception as e:
            traceback.print_exc()
            return float("nan")

    return eval_tracker_for_optimization


def main():
    """
    Iterates over jobs (model+fps) and trackers, running SMAC optimization for each.
    """
    for job in job_config["jobs"]:
        model_name = job["model"]
        fps_list = job["fps"]
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
            experiment_challenge_path = os.path.join(EXPERIMENT_PATH_ROOT, f"{challenge_name}-{MODE}")

            # load model for this detection source
            model = YOLO(model_path)

            for tracker_name in TRACKERS:
                print(f"\n{'='*60}")
                print(f"Job: {detection_source} | Tracker: {tracker_name}")
                print(f"{'='*60}")

                # per-tracker+model experiment path to avoid race conditions in parallel runs
                experiment_name = f"optimizer_{tracker_name}_{model_name}"
                experiment_path = os.path.join(experiment_challenge_path, experiment_name)
                experiment_data_path = os.path.join(experiment_path, "data")

                # setup directories
                os.makedirs(experiment_data_path, exist_ok=True)

                cs = ConfigurationSpace(SEARCH_DICT[tracker_name])

                # unique smac path per tracker+detection to avoid conflicts
                smac_path = Path(os.path.join(OPTIMIZING_DATA_PATH, "smac", f"{tracker_name}_{MODE}_{detection_source}"))

                # clean old smac data if it's around
                if smac_path.exists():
                    shutil.rmtree(smac_path)
                scenario = Scenario(cs, deterministic=True, n_trials=NUM_SEARCH_ITERATIONS,
                                    n_workers=workers, output_directory=smac_path)

                print(f"Running SMAC for {tracker_name}")
                eval_fn = make_eval_fn(tracker_name, detection_source, video_path, model,
                                       experiment_data_path, challenge_name, experiment_name)
                smac = BlackBoxFacade(scenario, eval_fn)
                smac.optimize()

                # get all results
                all_results = []
                for k, v in smac.runhistory.items():
                    smac_config = smac.runhistory.get_config(k.config_id)
                    all_results.append([1-v.cost] + list(smac_config.values()))

                # sort by hota
                all_results = sorted(all_results, key=lambda x: x[0], reverse=True)
                all_results = [[index, *row] for index, row in enumerate(all_results)]
                headers = ['Rank', 'HOTA'] + list(smac_config.keys())

                print(f"Top 10 hyper parameter combinations for {tracker_name}:")
                print(tabulate(all_results[:10], headers=headers))
                time_str = time.strftime("%Y%m%d-%H%M%S")
                save_path = os.path.join(OPTIMIZING_DATA_PATH,
                    f"{time_str}_smac_{tracker_name}_{MODE}_{detection_source}_results.txt")
                write_results_smac(all_results, headers, save_path)

                # clear smac contents so it doesn't use them for future runs
                shutil.rmtree(smac_path)


def write_results_smac(results_list, tabulate_headers, save_path):
    """
    Saves txt file of results to save_path

    Args:
    results_list: list of form [[metrics for run i], ...]
    tabulate_headers: headers for table to save
    save_path: path to save file
    """
    with open(save_path, 'w') as f:
        f.write(tabulate(results_list, headers=tabulate_headers))

if __name__ == "__main__":
    main()
