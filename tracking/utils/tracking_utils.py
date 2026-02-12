"""
Some functions that I use in multiple files
"""
import numpy as np
from tqdm import tqdm
import os

def get_tracker_args(smac_tracker_args):
    """
    Apply scaling to determine confidence thresholds. 
    Only does scaling for trackers with multiple confidence thresholds
    """

    # scale if botsort or bytetrack and they don't already have the thresholds (for default params during eval)
    if (smac_tracker_args["tracker_type"] in ["botsort", "bytetrack"]) and (not {"track_high_thresh", "new_track_thresh"} & smac_tracker_args.keys()):
        low_conf_threshold = smac_tracker_args["track_low_thresh"]
        track_high_scaling_factor = smac_tracker_args["track_high_scaling_factor"]
        new_track_scaling_factor = smac_tracker_args["new_track_scaling_factor"]

        track_high_thresh = low_conf_threshold + ((1-low_conf_threshold)*track_high_scaling_factor)
        new_track_thresh = low_conf_threshold + ((1-low_conf_threshold)*new_track_scaling_factor)

        smac_tracker_args["track_high_thresh"] = track_high_thresh
        smac_tracker_args["new_track_thresh"] = new_track_thresh

    return smac_tracker_args

def run_tracks(vids_to_process, video_dir, experiment_data_path, saved_detections_path, tracker_args, model, classes_to_eval):
    """
    Args:
        vids_to_process: list of names of vids to track
        video_dir: path to dir containing videos
        experiment_data_path: path to save results
        saved_detections_path: path to saved detections to use for offline tracking
        tracker_args: dict of tracker args
        model: detection model
        class_to_eval: class to save tracking results for
    """
    for i in tqdm(range(len(vids_to_process))):
        clip = vids_to_process[i]
        # print(f"Processing {clip}")
        vid_path = os.path.join(video_dir, clip + ".avi")
        # verify that video exists
        if not os.path.exists(vid_path):
            raise ValueError(f"Matching Video {vid_path} does not exist for detections {clip}")
        # run tracking
        results = model.track(source=vid_path, tracker=tracker_args, \
            offline_tracking=os.path.join(saved_detections_path, clip,"labels"), classes=classes_to_eval, verbose=False, conf=0.0)
        # get data in MOT format
        track_data = []
        for i in range(len(results)): # some results are not tracks, just detections
            if results[i].boxes.is_track: # this happens once per frame
                track_ids = results[i].boxes.id
                track_cls = results[i].boxes.cls
                track_confs = results[i].boxes.conf
                boxes_xywh = np.asarray(results[i].boxes.xywh) # center x,y
                boxes =  np.asarray(results[i].boxes.xyxy)
                # make boxes x1y1wh
                boxes[:, [2,3]] = boxes_xywh[:, [2,3]]
                curr_frame = i+1 # MOT not zero indexed
                assert len(track_ids) == len(track_confs) == len(boxes)
 
                for track_id, box, conf, cls in zip(track_ids, boxes, track_confs, track_cls):
                        x,y,w,h = box
                        track_data.append(f"{curr_frame},{track_id},{x},{y},{w},{h},{conf},-1,-1,-1\n")
        # save results
        file_path = os.path.join(experiment_data_path, clip + ".txt")
        with open(file_path, "w+", newline="") as txtfile:
            txtfile.writelines(track_data)
        txtfile.close()
