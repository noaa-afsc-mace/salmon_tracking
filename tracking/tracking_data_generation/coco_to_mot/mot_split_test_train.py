"""

Splits MOT data into test and train directories based on csv file

Excludes clips with no salmon

Expects the csv file to contain the following columns: ["Train", "Test", "Annotation File"]
Expects first row to be column names

Expects the train and test columns contain "True"/"False" and the annotation file column contains the name of the annotation file folder

Expects the following directory structure:
    MOT_DIR/
        annotation_name/
            gt/
                gt.txt
            seqinfo.ini
        ...

Creates the following:
    SAVE_DIR/
        train/
            ...
        test/
            ...

Usage:
    python split_test_train.py [mot_dir] [save_dir] [csv_path] --exclude-empty
"""

import os
import csv
import shutil
import pytest
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Splits MOT annotations")
parser.add_argument(
    "mot_dir",
    help="Path to directory containing MOT annotations and seqinfo.ini files",
    type=str,
)
parser.add_argument(
    "save_dir",
    help="Path to save location",
    type=str,
)
parser.add_argument(
    "csv_path",
    help="Path to csv file",
    type=str,
)
parser.add_argument(
    "--exclude-empty",
    help="Exclude empty gt.txt files",
    action="store_true",
    default=False
)
args = parser.parse_args()

def run_tests():
    pytest.main([
    "tracking/tracking_data_generation/merge_split/tests/test_mot_data_split.py"])

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

def do_it(mot_dir, save_dir, train_test_split_csv, exclude_empty):
    """
    Does it all
    """
    
    # create train and test directories
    train_dir = os.path.join(save_dir, "train")
    test_dir = os.path.join(save_dir, "test")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    train_annotations, test_annotations, excluded_annotations = load_train_test_csv_data(train_test_split_csv)

    # loop through annotations and save tr record files to train or test
    for a in tqdm(os.listdir(mot_dir)): 
        sub_dir = os.path.join(mot_dir, a)
        if os.path.isdir(sub_dir):
            annotation_name = os.path.split(sub_dir)[1]
            mot_annotation = os.path.join(sub_dir, "gt/gt.txt")
            with open(mot_annotation, "r") as file:
                content = file.read()
            if not content and exclude_empty:
                print(f"Annotation {annotation_name} excluded, no MOT annotations for video")
            elif annotation_name in train_annotations:
                # copy annotation file to train directory
                shutil.copytree(sub_dir, os.path.join(train_dir, annotation_name)) 
            elif annotation_name in test_annotations:
                # copy annotation file to test directory
                shutil.copytree(sub_dir, os.path.join(test_dir, annotation_name))
            elif annotation_name in excluded_annotations:
                print(f"Annotation {annotation_name} excluded")
            else:
                print(f"Annotation {annotation_name} not found in train, test or excluded annotations")

    print("Done")

if __name__ == "__main__":
    run_tests()
    do_it(args.mot_dir, args.save_dir, args.csv_path, args.exclude_empty)


