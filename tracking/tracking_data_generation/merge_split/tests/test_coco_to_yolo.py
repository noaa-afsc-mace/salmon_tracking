import pytest
import os
import shutil
import cv2
import numpy as np
import json
from convert_coco_to_yolo import convert_all

@pytest.fixture
def setup_existing_data(request):
    """
    Fixture to point to an existing COCO data directory for testing.
    """
    coco_annotation_dir = "tracking/tracking_data_generation/merge_split/tests/test_data/test_coco_folder"
    save_dir = "tracking/tracking_data_generation/merge_split/tests/test_data/yolo_temp"
    
    return {
        "coco_annotation_dir": coco_annotation_dir,
        "save_dir": save_dir,
        "classes_of_interest": request.param
    }

def build_yolo_dir(save_dir):
    """
    Create the YOLO directory
    """

    # Create the YOLO directory at the start of each test
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

def breakdown_yolo_dir(save_dir):
    """
    Deletes YOLO directory
    """

    # Cleanup the YOLO directory after each test
    if os.path.exists(save_dir):
        for filename in os.listdir(save_dir):
            file_path = os.path.join(save_dir, filename)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)

@pytest.mark.parametrize("setup_existing_data", [
    ["pollock"],  # Single class of interest
    ["pollock", "salmon"]  # Multiple classes of interest
], indirect=True)
def test_conversion_structure(setup_existing_data):
    """
    Test the structure of the output files after conversion.
    """
    coco_annotation_dir = setup_existing_data["coco_annotation_dir"]
    save_dir = setup_existing_data["save_dir"]
    classes_of_interest = setup_existing_data["classes_of_interest"]
    
    # build dir
    build_yolo_dir(save_dir)
    # Run the conversion
    convert_all(coco_annotation_dir, save_dir, classes_of_interest)

    # Check if the correct files are generated
    assert os.path.exists(os.path.join(save_dir, "label_map.txt"))

    for clip_name in os.listdir(coco_annotation_dir):
        if os.path.isdir(os.path.join(coco_annotation_dir,clip_name)):
            for item in os.listdir(os.path.join(coco_annotation_dir,clip_name, "frames")):
                if not item.startswith('.'):
                    assert os.path.exists(os.path.join(save_dir, f"{clip_name}_{os.path.splitext(item)[0]}.txt")), "missing .txt file in YOLO annotations"
                    assert os.path.exists(os.path.join(save_dir, f"{clip_name}_{os.path.splitext(item)[0]}.png")), "missing .png file in YOLO annotations"
    
    breakdown_yolo_dir(save_dir)

@pytest.mark.parametrize("setup_existing_data", [
    ["pollock"],  # Single class of interest
    ["pollock", "salmon"]  # Multiple classes of interest
], indirect=True)
def test_images_identical(setup_existing_data):
    """
    Tests images are identical to coco annotations
    """
    coco_annotation_dir = setup_existing_data["coco_annotation_dir"]
    save_dir = setup_existing_data["save_dir"]
    classes_of_interest = setup_existing_data["classes_of_interest"]
    
    # build dir
    build_yolo_dir(save_dir)
    # Run the conversion
    convert_all(coco_annotation_dir, save_dir, classes_of_interest)

    for clip_name in os.listdir(coco_annotation_dir):
        if os.path.isdir(os.path.join(coco_annotation_dir,clip_name)):
            for item in os.listdir(os.path.join(coco_annotation_dir,clip_name, "frames")):
                if not item.startswith('.'):
                    coco_img = cv2.imread(os.path.join(coco_annotation_dir,clip_name, "frames", item))
                    yolo_img = cv2.imread(os.path.join(save_dir, f"{clip_name}_{os.path.splitext(item)[0]}.png"))

                    assert np.array_equal(coco_img, yolo_img), 'coco and yolo images not identical'

    breakdown_yolo_dir(save_dir)

@pytest.mark.parametrize("setup_existing_data", [
    ["pollock"],  # Single class of interest
    ["pollock", "salmon"]  # Multiple classes of interest
], indirect=True)
def test_label_map(setup_existing_data):
    """
    Tests that label map contains only classes of interest and ids match coco annotations
    """

    coco_annotation_dir = setup_existing_data["coco_annotation_dir"]
    save_dir = setup_existing_data["save_dir"]
    classes_of_interest = setup_existing_data["classes_of_interest"]
    
    build_yolo_dir(save_dir)

    # Run the conversion
    convert_all(coco_annotation_dir, save_dir, classes_of_interest)

    # load label_map for yolo
    with open(os.path.join(save_dir, "label_map.txt")) as f:
        label_map = dict(line.strip().split(":") for line in f)
        label_map = {int(k): v.strip() for k, v in label_map.items()}

    # check that only classes of interest in label map
    for v in label_map.values():
        assert v in classes_of_interest, "label map class not in classes of interest"

    # load and check each json
    for clip_name in os.listdir(coco_annotation_dir):
        if os.path.isdir(os.path.join(coco_annotation_dir,clip_name)):
            with open(os.path.join(coco_annotation_dir,clip_name, f"{clip_name}.json")) as json_file:
                coco_json = json.load(json_file)
            json_file.close()

            # check that id's in label map are correct
            for cat in coco_json["categories"]:
                if cat["name"].lower in classes_of_interest:
                    assert label_map[cat["id"]] == cat["name"].lower, "id and name in label map don't match COCO annotations"
    
    breakdown_yolo_dir(save_dir)

@pytest.mark.parametrize("setup_existing_data", [
    ["pollock"],  # Single class of interest
    ["pollock", "salmon"]  # Multiple classes of interest
], indirect=True)
def test_bounding_box_transformation(setup_existing_data):
    """
    Tests that the bounding box coordinates are correctly normalized and formatted and that class ids are in label map.
    """
    coco_annotation_dir = setup_existing_data["coco_annotation_dir"]
    save_dir = setup_existing_data["save_dir"]
    classes_of_interest = setup_existing_data["classes_of_interest"]
    
    build_yolo_dir(save_dir)

    # Run the conversion
    convert_all(coco_annotation_dir, save_dir, classes_of_interest)

    # load label_map for yolo
    with open(os.path.join(save_dir, "label_map.txt")) as f:
        label_map = dict(line.strip().split(":") for line in f)
        label_map = {int(k): v.strip() for k, v in label_map.items()}

    # Check bounding box transformation
    for file in os.listdir(save_dir):
        if file.endswith(".txt") and file not in ("label_map.txt", "labelmap.txt"):
            annotations = []
            with open(os.path.join(save_dir, file)) as f:
                for line in f:
                    annotations.append(list(map(float, line.split())))

            for l in annotations:
                assert len(l) == 5, "l is weird"
                yolo_class_id, x_center, y_center, width, height = l
                
                # Check that the bounding box has been normalized correctly
                assert 0 <= x_center <= 1
                assert 0 <= y_center <= 1
                assert 0 <= width <= 1
                assert 0 <= height <= 1
                assert int(yolo_class_id) in label_map

    breakdown_yolo_dir(save_dir)
