import os
import math
import shutil
import configparser
import pytest

GT_DATA_PATH = "<your_path>/mot_annotations"
ORIGINAL_FRAME_RATE = 30
FRAME_REDUCTION_FACTOR = 4  # 1 is original. With 30 fps, 2->15, 3->10, 4->7.5 fps
NEW_FRAME_RATE = ORIGINAL_FRAME_RATE / FRAME_REDUCTION_FACTOR

def run_tests():
    pytest.main([
        "tracking/tracking_data_generation/frame_rate/test_convert_gt.py"])

def process_gt_data(gt_lines, frame_reduction_factor):
    """
    Reads GT data from a list of lines, applies frame reduction, and removes short tracks.
    Returns updated GT data and new sequence length.
    """
    gt_track_map = {}  # {track_id: ["data"]}

    for line in gt_lines:
        line_data = line.strip().split(',')
        frame_num = int(line_data[0])

        if (frame_num - 1) % frame_reduction_factor == 0:
            new_frame_num = int(((frame_num - 1) / frame_reduction_factor) + 1)
            line_data[0] = str(new_frame_num)
            track_id = int(line_data[1])

            if track_id in gt_track_map:
                gt_track_map[track_id].append(','.join(line_data))
            else:
                gt_track_map[track_id] = [','.join(line_data)]

    # Remove short tracks (<2 frames)
    new_gt_data = [track for track in gt_track_map.values() if len(track) > 1]
    new_gt_data = [line for track in new_gt_data for line in track]
    new_gt_data.sort(key=lambda x: int(x.split(",")[0]))  # Sort by frame number

    return new_gt_data

def write_gt_data(gt_file_path, new_gt_data, original_ini_path, new_ini_path, frame_reduction_factor):
    """
    Writes updated GT data and modifies the sequence info file.
    """
    if new_gt_data:
        with open(gt_file_path, 'w') as file:
            file.write('\n'.join(new_gt_data) + '\n')

        # Update seqinfo.ini
        config = configparser.ConfigParser()
        config.read(original_ini_path)
        config.set('Sequence', 'frameRate', str(NEW_FRAME_RATE))
        
        seq_length = int(config.get('Sequence', 'seqLength'))
        new_seq_length = math.ceil(seq_length / frame_reduction_factor)
        config.set('Sequence', 'seqLength', str(new_seq_length))

        with open(new_ini_path, 'w') as config_file:
            config.write(config_file)
    else:
        video_dir = os.path.dirname(os.path.dirname(gt_file_path))
        print(f"Removing video with empty GT {video_dir}")
        shutil.rmtree(video_dir)

def convert_gt(original_gt_path, frame_reduction_factor, new_frame_rate):
    """
    Copies GT to a new path and processes the annotation files.
    """
    new_gt_path = os.path.join(
        os.path.abspath(os.path.join(original_gt_path, "..")),
        f"fps-{new_frame_rate}_{os.path.basename(original_gt_path)}"
    )

    if os.path.exists(new_gt_path):
        print(f"GT source {new_gt_path} already exists. Overwriting")
    shutil.copytree(original_gt_path, new_gt_path, dirs_exist_ok=True)

    vid_dirs = [entry for entry in os.listdir(new_gt_path) if os.path.isdir(os.path.join(new_gt_path, entry))]
    
    for dir in vid_dirs:
        gt_file_path = os.path.join(new_gt_path, dir, "gt/gt.txt")
        original_ini_path = os.path.join(original_gt_path, dir, "seqinfo.ini")
        new_ini_path = os.path.join(new_gt_path, dir, "seqinfo.ini")

        # Open the gt.txt file and pass the content to process_gt_data
        with open(gt_file_path, 'r') as file:
            gt_lines = file.readlines()

        new_gt_data = process_gt_data(gt_lines, frame_reduction_factor)
        write_gt_data(gt_file_path, new_gt_data, original_ini_path, new_ini_path, frame_reduction_factor)

if __name__ == "__main__":
    run_tests()
    convert_gt(GT_DATA_PATH, FRAME_REDUCTION_FACTOR, NEW_FRAME_RATE)