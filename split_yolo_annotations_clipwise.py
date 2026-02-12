import os
import csv
import shutil
import re
import yaml
from tqdm import tqdm

# Path to the folder containing all annotations
ANNOTATIONS_PATH = "<your_path>/clip_based_yolo_annotations"

# Path where the organized folders will be saved
ORGANIZED_PATH = "<your_path>/clip_based_yolo_annotations_clipwise"

# Path to the CSV file that outlines the clip mapping
CLIP_MAPPING_CSV = "final_results_2026/train_test_split.csv"

def load_train_test_csv_data(train_test_split_csv):
    """
    Loads train test split csv and returns list of train, test and excluded clip names
    """
    with open(train_test_split_csv, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = list(csv_reader)
        # remove first row
        rows.pop(0)
    csv_file.close()

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

def organize_annotations(annotations_path, organized_path, train_test_split_csv):
    train_annotations, test_annotations, excluded_annotations = load_train_test_csv_data(train_test_split_csv)

    # Create the organized path if it doesn't exist
    if not os.path.exists(organized_path):
        os.makedirs(organized_path)

    # Regular expression pattern to extract video clip information from filenames
    pattern = re.compile(r"(\d{4}_t\d+_vid\d+)")

    # Loop through all files in the annotations path
    all_img_files_in_annotation_dir = []
    files = os.listdir(annotations_path)
    for f in files:
        if f.endswith(".png"):
            all_img_files_in_annotation_dir.append(os.path.join(annotations_path,f))

    num_test_annotations = 0
    num_unrecognized = 0

    # Copy to correct test folder
    for ann in tqdm(all_img_files_in_annotation_dir, desc="Organizing annotations"):
        ann_no_file_extension = os.path.splitext(os.path.basename(ann))[0]
        ann_clip_name = re.sub(r"_frame_\d+$", "", ann_no_file_extension)

        if ann_clip_name in test_annotations:
            clip_path = os.path.join(organized_path, ann_clip_name)
            if not os.path.exists(clip_path):
                os.makedirs(clip_path)
            
            # Copy the file to the clip folder
            shutil.copy2(f"{ann}", os.path.join(clip_path, f"{ann_no_file_extension}.png")) # copy img
            shutil.copy2(f"{ann[:-3]}txt", os.path.join(clip_path, f"{ann_no_file_extension}.txt")) # copy annotation
            num_test_annotations+=1

            # Create data.yaml file
            # Define the YAML data
            data = {
                "train": [],
                "val": [clip_path],
                "nc": 2,  # Number of classes
                "names": {
                    0: "salmon",
                    1: "pollock"
                }
            }

            # Save YAML file
            yaml_path = os.path.join(clip_path, "data.yaml")
            with open(yaml_path, "w") as file:
                yaml.dump(data, file, default_flow_style=False)

    print(f"Partition complete!\nTest annotation files: {num_test_annotations}")

if __name__ == "__main__":
    organize_annotations(ANNOTATIONS_PATH, ORGANIZED_PATH, CLIP_MAPPING_CSV)
