"""
Converts coco annotations to yolo format

Creates a directory with the following structure:

SAVE_DIR/
    <vid_name>_frame_<frame_number>.txt
    <vid_name>_frame_<frame_number>.png
    ...

usage: 
coco_to_yolo.py [-h] [COCO_ANNOTATION_DIR] [SAVE_DIR] [-i]

arguments:
    -h, --help            
                    Show this help message and exit.
    COCO_ANNOTATION_DIR
                    Path to directory containing COCO annotations
    SAVE_DIR
                    Path to directory where YOLO annotations will be saved
    -i, --classes_of_interest         
                    Name of classes
                    Ex: -i pollock salmon
"""

import os
import json
import argparse
import shutil
import pytest

# Initiate argument parser
parser = argparse.ArgumentParser(description="Converts coco annotations of tracks to the yolo format")
parser.add_argument(
    "coco_annotation_dir",
    help="Path to directory containing COCO annotations",
    type=str,
)
parser.add_argument(
    "save_dir",
    help="Path to directory where YOLO annotations will be saved",
    type=str,
)
parser.add_argument(
    "-i",
    "--class_of_interest",
    help="Name of class to include in yolo data",
    nargs='+',
    type=str,
    default=[]
)
args = parser.parse_args()

def run_tests():
    pytest.main(["tracking/tracking_data_generation/merge_split/tests/test_coco_to_yolo.py"])

def convert_coco(coco_path, images_path, save_path, classes_of_interest, name_to_yolo_id_dict):
    """
    Convert annotations from coco and saves to the txt path in this format: 
    class x_center y_center width height (normalized)
    Only includes the classes of interest

    NOTE: assumes all images are the same dimension

    Args:
        coco_path (str): path to coco annotations
        images_path (str): path to folder with images
        save_path (str): path to save txt file
        classes_of_interest (list): name of classes to include
        name_to_yolo_id_dict (dict):
    """

    # load coco json
    with open(coco_path) as json_file:
        coco_json = json.load(json_file)
    json_file.close()

    clip_name = os.path.splitext(os.path.basename(coco_path))[0]
    # re-name and copy all images into save_dir
    # also creates blank .txt file for each image, which will later (maybe) be edited if there are annotations for that image
    for img in os.listdir(images_path):
        if img.lower().endswith('.png'):
            source_file = os.path.join(images_path,img)
            frame_num = int(img[-10:-4])
            destination_img_file = os.path.join(save_path, f"{clip_name}_frame_{str(frame_num).zfill(6)}.png") # assumes that all frame files are of format "frame_<000000>.png"
            txt_file = os.path.join(save_path, f"{clip_name}_frame_{str(frame_num).zfill(6)}.txt")
            shutil.copy2(source_file, destination_img_file)
            with open(txt_file, 'w'):
                pass

    # create id maps for classes of interest
    yolo_id_to_coco_id_dict = {}

    for c_i in classes_of_interest:
        yolo_id = name_to_yolo_id_dict[c_i.lower()]
        for cat in coco_json["categories"]:
            if cat["name"].lower() == c_i.lower():
                yolo_id_to_coco_id_dict[yolo_id] = cat["id"]
                
    coco_id_to_yolo_id_dict = {value: key for key, value in yolo_id_to_coco_id_dict.items()}

    # save file with name to id dict
    with open(os.path.join(save_path, "label_map.txt"), "w") as txtfile:
        for name, yolo_id in name_to_yolo_id_dict.items():
            txtfile.write(f"{yolo_id}: {name}\n")
    txtfile.close()

    # create dict of image id to frame num (in image name)
    img_id_to_frame_num = {}
    for img in coco_json["images"]:
        basename = os.path.basename(img["file_name"])
        img_id_to_frame_num[img["id"]] = int(os.path.splitext(basename)[0][6:])
    img_width = coco_json["images"][0]["width"]
    img_height = coco_json["images"][0]["height"]

    # keep track of annotations for each frame
    # dict of form {frame_num: [annotation, annotation]}
    # annotation is string of form "yolo_class_id x_center y_center width height" all normalized
    frame_annotation_dict = {} 
    
    # iterate through all annotations
    for ann in coco_json["annotations"]:
        # check that annotation is in classes of interest
        coco_id = ann["category_id"]
        if coco_id not in yolo_id_to_coco_id_dict.values():
            continue
        yolo_id = coco_id_to_yolo_id_dict[coco_id]

        # get frame number
        frame_num = img_id_to_frame_num[ann["image_id"]]
        # get bbox of form: bb_left, bb_top, bb_width, bb_height
        xmin, ymin, bb_width, bb_height = ann["bbox"]
        scaled_width = bb_width/img_width
        scaled_height = bb_height/img_height
        scaled_x_center = (xmin/img_width) + (scaled_width/2)
        scaled_y_center = (ymin/img_height) + (scaled_height/2)
        # annotation is string of form "yolo_class_id x_center y_center width height" all normalized
        annotation_string = f"{int(yolo_id)} {scaled_x_center} {scaled_y_center} {scaled_width} {scaled_height}"

        if frame_num in frame_annotation_dict:
            frame_annotation_dict[frame_num].append(annotation_string)
        else:
            frame_annotation_dict[frame_num] = [annotation_string]

    # write all annotation files
    for frame_num, annotations in frame_annotation_dict.items():
        file_path = os.path.join(save_path, f"{clip_name}_frame_{str(frame_num).zfill(6)}.txt")
        with open(file_path, "w") as txtfile:
            txtfile.write('\n'.join(annotations))
        txtfile.close()

def convert_all(annotation_dir, save_dir, classes_of_interest):
    """
    Converts all coco annotations for all clips in annotation_dir to yolo format and saves to save_dir

    Args:
        annotation_dir (str): path to directory containing COCO annotations
        save_dir (str): path to directory where MOT annotations will be saved
        class_of_interest (list): list of classes to include in annotations
    Returns:
        None
    """

    print(f"\nConverting ground-truth COCO annotations to yolo format for classes of interest: {classes_of_interest}\n")
    print(f"COCO source directory: {annotation_dir}\n")

    # iterate through each clip
    # each clip dir contains:
    #   <clip_name>.json annotations
    #   frames/ folder containing each frame 
    #       frame_<000000>.PNG 

    name_to_yolo_id_dict = {item.lower(): index for index, item in enumerate(classes_of_interest)}
    
    for name in os.listdir(annotation_dir):
        clip_path = os.path.join(annotation_dir, name)
        if os.path.isdir(clip_path):
            # save png and txt file
            # save file name format <clip_name>_frame_<000000>.png/txt

            annotation_path = os.path.join(clip_path, f"{name}.json")
            images_dir = os.path.join(clip_path, "frames")
            print(f"Converting sequence: {name}")
            convert_coco(annotation_path, images_dir, save_dir, classes_of_interest, name_to_yolo_id_dict)
    print("\nConversion complete")
    print(f"ID dict: {name_to_yolo_id_dict}")
    with open(os.path.join(save_dir,"labelmap.txt"), "w") as f:
        for name, yolo_id in name_to_yolo_id_dict.items():
            f.write(f"{name}: {yolo_id}\n")

if __name__ == "__main__":
    run_tests()
    # parse arguments
    annotation_dir = args.coco_annotation_dir
    save_dir = args.save_dir
    classes_of_interest = args.class_of_interest
    convert_all(annotation_dir, save_dir, classes_of_interest)