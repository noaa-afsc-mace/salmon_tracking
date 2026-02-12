import os
import re
import shutil
from tqdm import tqdm
import pytest

# ---- Change these as needed ----

ORIGINAL_FRAME_RATE = 30
ORIGINAL_DETECTION_SOURCE = "model_type-yolo12x_data_source-clip_based_2_class_2_11_25_training_epochs-500_hyperparameter_source-default"
FRAME_REDUCTION_FACTOR = 4  # 1 is original. With 30 fps, 2->15, 3->10, 4->7.5 fps
DETECTION_DATA_PATH = "<your_path>/tracking_detections/"  # Adjust path as needed
SAVE_PATH = "<your_path>/tracking_detections/downsampled/"

NEW_FRAME_RATE = ORIGINAL_FRAME_RATE / FRAME_REDUCTION_FACTOR

# ----

def run_tests():
    pytest.main([
    "tracking/tracking_data_generation/frame_rate/test_convert_detection_frame_rate.py"])

def generate_file_map(file_list, frame_reduction_factor):
    """
    Generate a mapping of original file names to new file names and destination directory.
    Returns a dictionary with the frame number as the key, and a tuple containing
    the original file path and the new file name.
    """
    files_dict = {}  # {frame_num: (original file path, new file name)}
    valid_file_pattern = "^.+_\d+\.txt$"  # regex to match <anything>_<int>.txt
    split_on_pattern = "[_\.]"  # regex to split on "." and "_"
    frame_num_pattern = "_(\d+)\.txt$"
    
    for f in file_list:
        if re.match(valid_file_pattern, f):
            frame = int(re.split(split_on_pattern, f)[-2])
            files_dict[frame] = f
        else:
            raise FileExistsError(f"Found file {f} that does not end in '.txt.'")
    
    file_map = {}
    for frame_num, file_path in files_dict.items():
        if (frame_num - 1) % frame_reduction_factor == 0:  # zero-indexed
            new_frame_num = int(((frame_num - 1) / frame_reduction_factor) + 1)  # divide by frame reduction factor, un-zero index
            destination_file_name = re.sub(frame_num_pattern, f"_{new_frame_num}.txt", os.path.basename(file_path))
            file_map[frame_num] = (file_path, destination_file_name)

    return file_map

def copy_and_rename_files(file_map, destination_dir):
    """
    Copy and rename the files based on the file_map.
    file_map should be a dictionary where each key is a frame number,
    and the value is a tuple of (original file path, new file name).
    """
    for original_file, new_file_name in file_map.values():
        shutil.copy(original_file, os.path.join(destination_dir, new_file_name))

def main(frame_reduction_factor, destination_dir):
    # make new detection source
    new_detection_source_path = os.path.join(destination_dir, f"fps-{NEW_FRAME_RATE}_{ORIGINAL_DETECTION_SOURCE}")
    if not os.path.exists(new_detection_source_path):
        os.mkdir(new_detection_source_path)

    # loop through detections
    detections_dir = os.path.join(DETECTION_DATA_PATH, ORIGINAL_DETECTION_SOURCE)
    all_detection_dirs = [entry.name for entry in os.scandir(detections_dir) if entry.is_dir()]
    for clip in tqdm(all_detection_dirs):
        vid_detections = os.path.join(detections_dir, clip, "labels")
        clip_path = os.path.join(new_detection_source_path, clip)
        if not os.path.exists(clip_path):
            os.mkdir(clip_path)
        full_clip_path = os.path.join(clip_path, "labels")
        if not os.path.exists(full_clip_path):
            os.mkdir(full_clip_path)

        if not (isinstance(vid_detections, str) and os.path.isdir(vid_detections)):
            raise FileNotFoundError(f"{vid_detections} is not a valid directory")
        
        file_list = [os.path.join(vid_detections, f) for f in os.listdir(vid_detections)]
        file_map = generate_file_map(file_list, frame_reduction_factor)
        copy_and_rename_files(file_map, full_clip_path)

if __name__ == "__main__":
    run_tests()
    main(FRAME_REDUCTION_FACTOR, SAVE_PATH)
