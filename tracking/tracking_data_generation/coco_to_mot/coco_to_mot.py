"""
File for converting the COCO annotations from the merge_split workflow to the MOT format

Creates a directory with the following structure: (adheres to the MOTChallenge format structure)

SAVE_DIR/
    <clip_name>/
        gt/
            gt.txt
        seqinfo.ini
    ...

usage: 
coco_to_mot.py [-h] [COCO_ANNOTATION_DIR] [SAVE_DIR] [-i]

arguments:
    -h, --help            
                    Show this help message and exit.
    COCO_ANNOTATION_DIR
                    Path to directory containing COCO annotations
    SAVE_DIR
                    Path to directory where MOT annotations will be saved
    -i, --class_of_interest         
                    Name of class to include in MOT data (can only be one). Defaults to "Salmon". 
                    Ex: -i Pollock
    
"""

import os
import json
import argparse
import pytest

# Initiate argument parser
parser = argparse.ArgumentParser(description="Converts xml annotations of tracks to the MOT format to be used in tracker evaluation")
parser.add_argument(
    "coco_annotation_dir",
    help="Path to directory containing COCO annotations",
    type=str,
)
parser.add_argument(
    "save_dir",
    help="Path to directory where MOT annotations will be saved",
    type=str,
)
parser.add_argument(
    "-i",
    "--class_of_interest",
    help="Name of class to include in MOT data (can only be one). Defaults to 'Salmon'",
    default="Salmon",
    type=str,
)
args = parser.parse_args()

def run_tests():
        pytest.main([
        "tracking/tracking_data_generation/merge_split/tests/test_coco_to_mot.py"])

def convert_coco(coco_path, save_path, class_of_interest):
    """
    Convert annotations from coco and saves to the txt path in this format: 
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    Only includes the classes of interest
    Frame is NOT zero indexed

    Args:
        coco_path (str): path to coco annotations
        save_path (str): path to save txt file
        class_of_interest (str): name of class to include in MOT data (can only be one)
    Returns:
        width (int): width of images
        height (int): height of images
        num_frames (int): number of frames in sequence
    """
    rows_list = []

    # load coco json
    with open(coco_path) as json_file:
        coco_json = json.load(json_file)
    json_file.close()

    # find category id of class of interest
    cat_id = None
    for cat in coco_json["categories"]:
        if cat["name"].lower() == class_of_interest.lower():
            cat_id = cat["id"]
            break
    # create dict of image id to frame num (in image name)
    img_id_to_frame_num = {}
    for img in coco_json["images"]:
        basename = os.path.basename(img["file_name"])
        img_id_to_frame_num[img["id"]] = int(os.path.splitext(basename)[0][6:])
    
    # iterate through all annotations
    for ann in coco_json["annotations"]:
        # check that annotation is of class of interest
        if ann["category_id"] != cat_id:
            continue
        
        # get frame number
        frame_num = img_id_to_frame_num[ann["image_id"]]
        # get track id
        track_id = ann["attributes"]["track_id"]
        # get bbox
        bbox = ann["bbox"]

        rows_list.append(f"{frame_num+1},{track_id},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},1,-1,-1,-1\n") # NOTE: frame_num+1 because frame_num is not zero indexed for MOT eval

    with open(save_path, "w+", newline="") as txtfile:
        txtfile.writelines(rows_list)
    txtfile.close()

    width = coco_json["images"][0]["width"]
    height = coco_json["images"][0]["height"]
    last_basename = os.path.basename(coco_json["images"][-1]["file_name"])
    num_frames = int(os.path.splitext(last_basename)[0][6:]) + 1  # NOTE: frame_num+1 because frame_num is zero indexed

    return width, height, num_frames

def create_ini_file(ini_file_path, seq_name, seq_len, w, h):
    """
    Creates seqinfo.ini file for sequence

    Args:
        ini_file_path (str): path to save seqinfo.ini file
        seq_name (str): name of sequence
        seq_len (int): length of sequence
        w (int): width of images
        h (int): height of images
    Returns:
        None
    """
    # not sure what imgDir is for so leaving it empty for now
    ini_contents = \
    f"[Sequence]\nname={seq_name}\nimDir=\nframeRate=\nseqLength={seq_len}\nimWidth={w}\nimHeight={h}\nimExt="

    with open(ini_file_path, "w", newline="") as inifile:
        inifile.write(ini_contents)
    inifile.close()

def convert_all(annotation_dir, save_dir, class_of_interest):
    """
    Converts all coco annotations for all clips in annotation_dir to MOT format and saves to save_dir

    Args:
        annotation_dir (str): path to directory containing COCO annotations
        save_dir (str): path to directory where MOT annotations will be saved
        class_of_interest (str): name of class to include in MOT data (can only be one)
    Returns:
        None
    """

    print(f"\nConverting ground-truth COCO annotations to MOT format for class of interest: {class_of_interest}\n")
    print(f"COCO source directory: {annotation_dir}\n")

    for name in os.listdir(annotation_dir):
        clip_path = os.path.join(annotation_dir, name)
        if os.path.isdir(clip_path):
            # create directory for MOT clip
            clip_save_path = os.path.join(save_dir, name)
            gt_path = os.path.join(clip_save_path, "gt")
            if not os.path.isdir(clip_save_path):
                os.mkdir(clip_save_path)
            if not os.path.isdir(gt_path): 
                os.mkdir(gt_path)

            annotation_path = os.path.join(clip_path, f"{name}.json")
            txt_file_path = os.path.join(gt_path, "gt.txt")
            print(f"Converting sequence: {name}")
            width, height, frames = convert_coco(annotation_path, txt_file_path, class_of_interest)
            print("Creating seqinfo.ini file")
            create_ini_file(os.path.join(clip_save_path, "seqinfo.ini"), name, frames, width, height)
    print("\nConversion complete")

if __name__ == "__main__":
    # run tests
    run_tests()
    # parse arguments
    annotation_dir = args.coco_annotation_dir
    save_dir = args.save_dir
    class_of_interest = args.class_of_interest
    convert_all(annotation_dir, save_dir, class_of_interest)

