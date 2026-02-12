"""
Creates csv clip info
"""

import os
import json
import csv

ANNOTATION_DIR = "<your_path>/clip_based_coco_annotations"
SAVE_DIR = "<your_path>"
COLUMNS = ["Annotation File", "Num Frames", "Num Salmon Annotations", "Num Pollock Annotations", "No Fish", "Fish Low", "Fish Med", "Fish High", "Occlusion", "Low Visibility", "Num Salmon Tracks", "Num Pollock Tracks"]


def process_annotation(annotation_json, annotation_file):
    """
    Gets info from annotation and returns data
    """
    num_frames = len(annotation_json["images"])
    num_salmon_annotations = 0
    num_pollock_annotations = 0
    no_fish = False
    fish_low = False
    fish_medium = False
    fish_high = False
    occlusion = False
    low_visibility = False

    salmon_track_ids = []
    pollock_track_ids = []

    for annotation in annotation_json["annotations"]:
        if annotation["category_id"] == 2:
            num_salmon_annotations += 1
            salmon_track_ids.append(annotation["attributes"]["track_id"])
        elif annotation["category_id"] == 1:
            num_pollock_annotations += 1
            pollock_track_ids.append(annotation["attributes"]["track_id"])
    
    for image in annotation_json["images"]:
        if "No Fish" in image["cvat_tags"]:
            no_fish = True
        if "Fish Low" in image["cvat_tags"]:
            fish_low = True
        if "Fish Med" in image["cvat_tags"]:
            fish_medium = True
        if "Fish High" in image["cvat_tags"]:
            fish_high = True
        if "Occlusion" in image["cvat_tags"]:
            occlusion = True
        if "Low Visibility" in image["cvat_tags"]:
            low_visibility = True
    
    # create row ["Annotation File", "Num Frames", "Num Salmon Annotations", "Num Pollock Annotations", "No Fish", "Fish Low", "Fish Med", "Fish High", "Occlusion", "Low Visibility", "Num Salmon Tracks", "Num Pollock Tracks"]

    return [annotation_file, num_frames, num_salmon_annotations, num_pollock_annotations, no_fish, fish_low, fish_medium, fish_high, occlusion, low_visibility, len(set(salmon_track_ids)), len(set(pollock_track_ids))]


def runner():
    """
    Runs through all annotations
    ANNOTATION_DIR has the following format:
        ANNOTATION_DIR/
            annotation_name/
                annotation_name.json
                frames/
                    frame_000000.jpeg
    """
    # Get all annotations
    annotations = os.listdir(ANNOTATION_DIR)
    data = []
    for annotation in annotations:
        annotation_dir = os.path.join(ANNOTATION_DIR, annotation)
        if os.path.isdir(annotation_dir):
            with open(os.path.join(annotation_dir, f"{annotation}.json")) as json_file:
                coco_json = json.load(json_file)
            json_file.close()
            row = process_annotation(coco_json, annotation)
            data.append(row)
    # Open the CSV file for writing
    with open(os.path.join(SAVE_DIR, "clip_info.csv"), "w", newline="") as csv_file:
        # Create a CSV writer object
        writer = csv.writer(csv_file)

        # Write the column names to the CSV file
        writer.writerow(COLUMNS)

        # Write the data to the CSV file
        for row in data:
            writer.writerow(row)
        csv_file.close()

if __name__ == "__main__":
    runner()


        

