import pytest
import os
import csv
import shutil
from yolo_test_train_split import do_it

def build_dest_dir(dest_dir):
    """
    Create the dest directory
    """

    # Create the dest directory at the start of each test
    if os.path.exists(dest_dir):
        # Clean the directory before the test
        for filename in os.listdir(dest_dir):
            file_path = os.path.join(dest_dir, filename)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
    else:
        os.makedirs(dest_dir)

def breakdown_dest_dir(dest_dir):
    """
    Deletes dest directory
    """

    # Cleanup the dest directory after each test
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

@pytest.fixture
def setup_existing_data():
    """
    Fixture to point to an existing COCO data directory for testing.
    """
    yolo_annotations_dir = "tracking/tracking_data_generation/merge_split/tests/test_data/yolo_train_test/source"
    save_dir = "tracking/tracking_data_generation/merge_split/tests/test_data/yolo_train_test/dest"
    train_test_split_csv = "tracking/tracking_data_generation/merge_split/tests/test_data/testing_split.csv"

    return {
        "train_test_split_csv": train_test_split_csv,
        "save_dir": save_dir,
        "yolo_annotations_dir": yolo_annotations_dir
    }

def test_directory_contents(setup_existing_data):
    """
    Tests that no duplicate files exist in either directory
    """

    train_test_split_csv = setup_existing_data["train_test_split_csv"]
    save_dir = setup_existing_data["save_dir"]
    yolo_annotations_dir = setup_existing_data["yolo_annotations_dir"]

    build_dest_dir(save_dir)

    # run split
    do_it(save_dir, yolo_annotations_dir, train_test_split_csv)

    # check split folders
    files_in_train = set([f for f in os.listdir(os.path.join(save_dir, "train")) if f.endswith(('.txt', '.png'))])
    files_in_test = set([f for f in os.listdir(os.path.join(save_dir, "test")) if f.endswith(('.txt', '.png'))])

    for sub_dir in ["test", "train"]:
        for f in os.listdir(os.path.join(save_dir, sub_dir)):
            if f.startswith("."):
                continue
            if sub_dir == "train":
                if f in files_in_test:
                    raise ValueError(f"Duplicate file found: {f} in both 'train' and 'test' directories.")
            elif sub_dir == "test":
                if f in files_in_train:
                    raise ValueError(f"Duplicate file found: {f} in both 'train' and 'test' directories.")

    # ensure that all files in the source directory are accounted for in either train or test
    source_files = set(os.listdir(yolo_annotations_dir))
    source_files.discard('label_map.txt') # ignore label_map
    all_split_files = files_in_train.union(files_in_test)

    missing_files = source_files - all_split_files
    if missing_files:
        raise ValueError(f"The following files are missing in 'train' or 'test' directories: {missing_files}")
    
    breakdown_dest_dir(save_dir)

def test_correct_vids(setup_existing_data):
    """
    Tests that no duplicate files exist in either directory
    """

    train_test_split_csv = setup_existing_data["train_test_split_csv"]
    save_dir = setup_existing_data["save_dir"]
    yolo_annotations_dir = setup_existing_data["yolo_annotations_dir"]

    build_dest_dir(save_dir)

    # run split
    do_it(save_dir, yolo_annotations_dir, train_test_split_csv)

    # check split folders
    files_in_train = set([f for f in os.listdir(os.path.join(save_dir, "train")) if f.endswith(('.txt', '.png'))])
    files_in_test = set([f for f in os.listdir(os.path.join(save_dir, "test")) if f.endswith(('.txt', '.png'))])

    # get train/test split
    # load csv
    with open(train_test_split_csv, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = list(csv_reader)
        # remove first row
        rows.pop(0)
    csv_file.close()

    train_annotations = []
    test_annotations = []
    for row in rows:
        annotation_name = row["Annotation File"]
        if row["Train"] == "TRUE":
            train_annotations.append(annotation_name)
        if row["Test"] == "TRUE":
            test_annotations.append(annotation_name)

    # go through all files and make sure none are in wrong dir

    for train_file in files_in_train:
        clip_name = train_file.split('_frame')[0]
        assert clip_name in train_annotations
    
    for test_file in files_in_test:
        clip_name = test_file.split('_frame')[0]
        assert clip_name in test_annotations
    
    breakdown_dest_dir(save_dir)






    
    