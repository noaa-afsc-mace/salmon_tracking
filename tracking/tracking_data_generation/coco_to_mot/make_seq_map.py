"""
Quick script to make seq map for MOT eval

Gets names of all clips in MOT_DIR and saves .txt file of form:
name
<clip_name_1>
...

Expected structure of MOT_DIR:
    MOT_DIR/
        <clip_name_1>/
            ...
        ...
"""

import os

train_test = ["train", "test"]
fps = [30.0, 15.0, 10.0, 7.5]

for frame_rate in fps:
    for t in train_test:

        MOT_DIR = f"tracking/gt/mot_challenge/MOTFish_{frame_rate}-{t}"
        SAVE_PATH = "tracking/gt/mot_challenge/seqmaps"
        FILE_NAME = f"MOTFish_{frame_rate}-{t}.txt"

        with open(os.path.join(SAVE_PATH, FILE_NAME), "w") as f:
            f.write("name\n")
            for name in os.listdir(MOT_DIR):
                # check if name is a directory
                if os.path.isdir(os.path.join(MOT_DIR, name)):
                    f.write(f"{name}\n")
        f.close()