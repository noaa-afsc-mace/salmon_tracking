import pytest
import os
import shutil
import configparser
import numpy as np
import json
from coco_to_mot import convert_all

@pytest.fixture
def setup_existing_data(request):
    """
    Fixture to point to an existing COCO data directory for testing.
    """
    coco_annotation_dir = "tracking/tracking_data_generation/merge_split/tests/test_data/test_coco_folder"
    save_dir = "tracking/tracking_data_generation/merge_split/tests/test_data/mot_temp"
    
    return {
        "coco_annotation_dir": coco_annotation_dir,
        "save_dir": save_dir,
        "class_of_interest": request.param
    }

def build_mot_dir(save_dir):
    """
    Create the MOT directory
    """

    # Create the MOT directory at the start of each test
    if os.path.exists(save_dir):
        # Clean the directory before the test
        for filename in os.listdir(save_dir):
            file_path = os.path.join(save_dir, filename)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
    else:
        os.makedirs(save_dir)

def breakdown_mot_dir(save_dir):
    """
    Deletes MOT directory
    """

    # Cleanup the MOT directory after each test
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

@pytest.mark.parametrize("setup_existing_data", [
    "pollock",  
    "salmon"  # default is salmon
], indirect=True)
def test_conversion_structure(setup_existing_data):
    """
    Test the structure of the output files after conversion.
    """
    coco_annotation_dir = setup_existing_data["coco_annotation_dir"]
    save_dir = setup_existing_data["save_dir"]
    class_of_interest = setup_existing_data["class_of_interest"]
    
    # build dir
    build_mot_dir(save_dir)
    # Run the conversion
    convert_all(coco_annotation_dir, save_dir, class_of_interest)

    # Check if the correct files are generated
    for clip_name in os.listdir(coco_annotation_dir):
        clip_path = os.path.join(coco_annotation_dir, clip_name)
        
        if os.path.isdir(clip_path):
            # Check that seqinfo.ini file exists
            seqinfo_path = os.path.join(save_dir, clip_name, "seqinfo.ini")
            assert os.path.exists(seqinfo_path), f"Missing seqinfo.ini in {clip_name}"

            # Check that gt directory exists
            gt_dir = os.path.join(save_dir, clip_name, "gt")
            assert os.path.isdir(gt_dir), f"Missing gt directory in {clip_name}"

            # Check that gt directory contains only gt.txt
            gt_txt_path = os.path.join(gt_dir, "gt.txt")
            assert os.path.exists(gt_txt_path), f"Missing gt.txt in {clip_name}/gt"
            assert len(os.listdir(gt_dir)) == 1, f"Unexpected files in {clip_name}/gt"
    
    breakdown_mot_dir(save_dir)

@pytest.mark.parametrize("setup_existing_data", [
    "pollock",  
    "salmon"  # default is salmon
], indirect=True)
def test_class_of_interest_map(setup_existing_data):
    """
    Tests that MOT annotations contain only class of interest and ids match coco annotations
    Expected MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    """

    coco_annotation_dir = setup_existing_data["coco_annotation_dir"]
    save_dir = setup_existing_data["save_dir"]
    class_of_interest = setup_existing_data["class_of_interest"]
    
    build_mot_dir(save_dir)

    # Run the conversion
    convert_all(coco_annotation_dir, save_dir, class_of_interest)

    # Check if the correct files are generated
    for clip_name in os.listdir(coco_annotation_dir):
        clip_path = os.path.join(coco_annotation_dir, clip_name)
        
        if os.path.isdir(clip_path):
            with open(os.path.join(coco_annotation_dir,clip_name, f"{clip_name}.json")) as json_file:
                coco_json = json.load(json_file)
            json_file.close()

            gt_dir = os.path.join(save_dir, clip_name, "gt")
            gt_txt_path = os.path.join(gt_dir, "gt.txt")

            # Count the number of annotations for class of interest in the COCO JSON
            category_dict = {cat["id"]: cat["name"] for cat in coco_json["categories"]}
            coco_class_count = 0
            for ann in coco_json["annotations"]:
                if category_dict[ann["category_id"]].lower() == class_of_interest.lower():
                    coco_class_count+=1

            # Count the number of lines in the generated MOT file
            with open(gt_txt_path, "r") as mot_file:
                mot_lines = mot_file.readlines()

            mot_class_count = len(mot_lines)

            # Assert that the counts match
            assert (
                coco_class_count == mot_class_count
            ), f"Mismatch in annotation count for {clip_name}: COCO={coco_class_count}, MOT={mot_class_count}"
    
    breakdown_mot_dir(save_dir)

@pytest.mark.parametrize("setup_existing_data", [
    "pollock",
    "salmon"
], indirect=True)
def test_bounding_box_coordinates_and_frames(setup_existing_data):
    """
    Tests that bounding box coordinates and frame numbers in the MOT format match expected COCO values.
    Expected MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    """

    coco_annotation_dir = setup_existing_data["coco_annotation_dir"]
    save_dir = setup_existing_data["save_dir"]
    class_of_interest = setup_existing_data["class_of_interest"]

    build_mot_dir(save_dir)

    # Run the conversion
    convert_all(coco_annotation_dir, save_dir, class_of_interest)

    # Process each clip in the COCO annotation directory
    for clip_name in os.listdir(coco_annotation_dir):
        clip_path = os.path.join(coco_annotation_dir, clip_name)

        if os.path.isdir(clip_path):
            with open(os.path.join(clip_path, f"{clip_name}.json")) as json_file:
                coco_json = json.load(json_file)

            gt_dir = os.path.join(save_dir, clip_name, "gt")
            gt_txt_path = os.path.join(gt_dir, "gt.txt")

            # Load MOT annotations
            mot_annotations = {}
            with open(gt_txt_path, "r") as mot_file:
                for line in mot_file:
                    parts = line.strip().split(",")
                    frame_num = int(parts[0]) -1 # MOT is not zero indexed
                    bbox = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]  # x, y, width, height

                    if frame_num not in mot_annotations:
                        mot_annotations[frame_num] = []
                    mot_annotations[frame_num].append(bbox)

            # Load COCO category mappings
            category_dict = {cat["id"]: cat["name"] for cat in coco_json["categories"]}

            # Extract COCO bounding boxes and frame numbers for the class of interest
            id_to_frame_num = {img["id"]: int(img["file_name"].split("_")[1].split(".")[0]) for img in coco_json["images"]}
            coco_annotations = {}
            for ann in coco_json["annotations"]:
                if category_dict[ann["category_id"]].lower() == class_of_interest.lower():
                    frame_num = id_to_frame_num[ann["image_id"]]
                    bbox = ann["bbox"]

                    if frame_num not in coco_annotations:
                        coco_annotations[frame_num] = []
                    coco_annotations[frame_num].append(bbox)

            # Compare bounding boxes for each frame
            assert len(coco_annotations) == len(mot_annotations), f"Mismatch in number of annotations for frame {frame}"
            for frame in coco_annotations:
                coco_boxes = np.array(sorted(coco_annotations.get(frame, []))) # coco format: x,y,width,height
                mot_boxes = np.array(sorted(mot_annotations.get(frame, []))) 

                assert np.array_equal(coco_boxes, mot_boxes), f"Mismatch in frame {frame}"

    breakdown_mot_dir(save_dir)

@pytest.mark.parametrize("setup_existing_data", [
    "pollock",
    "salmon"
], indirect=True)
def test_seqinfo_file(setup_existing_data):
    """
    Tests that seqinfo.ini file has expected values
    """

    coco_annotation_dir = setup_existing_data["coco_annotation_dir"]
    save_dir = setup_existing_data["save_dir"]
    class_of_interest = setup_existing_data["class_of_interest"]

    build_mot_dir(save_dir)

    # Run the conversion
    convert_all(coco_annotation_dir, save_dir, class_of_interest)

    # Process each clip in the COCO annotation directory
    for clip_name in os.listdir(coco_annotation_dir):
        clip_path = os.path.join(coco_annotation_dir, clip_name)

        if os.path.isdir(clip_path):
            with open(os.path.join(clip_path, f"{clip_name}.json")) as json_file:
                coco_json = json.load(json_file)

            seqinfo_path = os.path.join(save_dir, clip_name, "seqinfo.ini")

            config = configparser.ConfigParser()
            config.read(seqinfo_path)
            sequence = config["Sequence"]
            name = sequence.get("name")
            seqLength = sequence.getint("seqLength", fallback=None)
            imWidth = sequence.getint("imWidth", fallback=None)
            imHeight = sequence.getint("imHeight", fallback=None)

            assert name == clip_name
            assert seqLength == len(coco_json["images"])
            assert imWidth == coco_json["images"][0]["width"]
            assert imHeight == coco_json["images"][0]["height"]

    breakdown_mot_dir(save_dir)


