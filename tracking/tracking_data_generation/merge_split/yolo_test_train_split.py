"""
Script to sort annotations based on clips

A bit complicated and convoluted

Creates test and train directories with the following structure:

<test/train>/
    <vid_name>_frame_<frame_number>.txt
    <vid_name>_frame_<frame_number>.png
    ...

usage: 
test_train_split.py [-h] [SAVE_DIR] [YOLO_ANNOTATIONS_DIR] [TEST_TRAIN_SPLIT_CSV]

arguments:
    -h, --help            
                    Show this help message and exit.
    SAVE_DIR
                    Path to directory where YOLO annotations will be saved
    YOLO_ANNOTATIONS_DIR
                    Directory containing YOLO annotations
    TEST_TRAIN_SPLIT_CSV
                    Path to csv file with test train split for each clip
"""

import os
import csv
import argparse
import shutil
import re
import pytest
from tqdm import tqdm

# Initiate argument parser
parser = argparse.ArgumentParser(description="Script to sort annotations based on clips")
parser.add_argument(
    "save_dir",
    help="Path to directory where yolo annotations will be saved",
    type=str,
)
parser.add_argument(
    "yolo_annotations_dir",
    help="Directory containing YOLO annotations",
    type=str,
)
parser.add_argument(
    "train_test_split_csv",
    help="Path to csv file with test train split for each clip",
    type=str,
)

args = parser.parse_args()

def run_tests():
    pytest.main(["tracking/tracking_data_generation/merge_split/tests/test_data_split.py"])

def load_train_test_csv_data(train_test_split_csv):
    """
    Loads train test split csv and returns list of train, test and excluded clip names
    """
    # load csv
    with open(train_test_split_csv, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = list(csv_reader)
        # remove first row
        rows.pop(0)
    csv_file.close()

    # make list of train and test annotations
    train_annotations = []
    test_annotations = []
    excluded_annotations = []
    for row in rows:
        annotation_name = row["Annotation File"]
        train_val = True if row["Train"] == "TRUE" else False
        test_val = True if row["Test"] == "TRUE" else False
        excluded_val = True if row["Excluded"] == "TRUE" else False
        assert train_val ^ test_val ^ excluded_val, "Clip must be either in train, test or excluded"

        if train_val:
            train_annotations.append(annotation_name)
        elif test_val:
            test_annotations.append(annotation_name)
        elif excluded_val:
            excluded_annotations.append(annotation_name)
    
    return train_annotations, test_annotations, excluded_annotations

def do_it(save_dir, yolo_annotations_dir, train_test_split_csv):
    """
    Does it all
    """

    # make train and test dir
    train_dir = os.path.join(save_dir, "train")
    test_dir = os.path.join(save_dir, "test")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # now I gotta do test/train stuff and sort it out
    train_annotations, test_annotations, excluded_annotations = load_train_test_csv_data(train_test_split_csv)

    # get names of all annotations
    all_img_files_in_annotation_dir = []
    # for dir in annotations:
    files = os.listdir(yolo_annotations_dir)
    for f in files:
        if f.endswith(".png"):
            all_img_files_in_annotation_dir.append(os.path.join(yolo_annotations_dir,f))
    
    num_train_annotations = 0
    num_test_annotations = 0
    num_excluded_annotations = 0
    num_unrecognized = 0

    # copy to correct train/test folder
    for ann in tqdm(all_img_files_in_annotation_dir, desc="Splitting annotations"):
        ann_no_file_extension = os.path.splitext(os.path.basename(ann))[0]
        ann_clip_name = re.sub(r"_frame_\d+$", "", ann_no_file_extension)

        if ann_clip_name in train_annotations:
            shutil.copy2(f"{ann[:-3]}png", os.path.join(train_dir, f"{ann_no_file_extension}.png")) # copy img
            shutil.copy2(f"{ann[:-3]}txt", os.path.join(train_dir, f"{ann_no_file_extension}.txt")) # copy annotation
            num_train_annotations+=1
        elif ann_clip_name in test_annotations:
            shutil.copy2(f"{ann[:-3]}png", os.path.join(test_dir, f"{ann_no_file_extension}.png")) # copy img
            shutil.copy2(f"{ann[:-3]}txt", os.path.join(test_dir, f"{ann_no_file_extension}.txt")) # copy annotation
            num_test_annotations+=1
        elif ann_clip_name in excluded_annotations:
            print(f"{ann} excluded")
            num_excluded_annotations+=1
        else:
            print(f"{ann} not recognized")
            num_unrecognized+=1

    print(f"Partition complete!\nTrain annotation files: {num_train_annotations}\
          \nTest annotation files: {num_test_annotations}\nExcluded annotation files: {num_excluded_annotations}\
          \nUnrecognized annotation files in source dir: {num_unrecognized}")

if __name__ == "__main__":
    run_tests()
    # parse arguments
    save_dir = args.save_dir
    yolo_annotations_dir = args.yolo_annotations_dir
    train_test_split_csv = args.train_test_split_csv

    # Check if directories/files exist
    all_paths = [save_dir, yolo_annotations_dir, train_test_split_csv]
    for path in all_paths:
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Path '{save_dir}' does not exist.")
        
    do_it(save_dir, yolo_annotations_dir, train_test_split_csv)
    

