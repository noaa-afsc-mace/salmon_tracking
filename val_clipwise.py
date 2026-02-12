"""
Runs model eval

See: 
- https://docs.ultralytics.com/modes/val/#usage-examples 
"""
import os
import csv
from ultralytics import YOLO

# ---- Change these as needed ----
MODEL_NAME = "model_type-yolo12m_data_source-clip_based_2_class_2_11_25_training_epochs-500_hyperparameter_source-default"
MODEL_PATH = f"training/{MODEL_NAME}/weights/best.pt"
DATA_PATH = "<your_path>/clip_based_yolo_annotations_clipwise"
SAVE_PATH = f"tracking/clip_level_model_performance/{MODEL_NAME}.csv"
# ----
DEVICE = "mps"
# [0,1,2,3]
COLUMNS = ["sequence", "precision", "recall", "map50","map50-95"]


rows = [COLUMNS]
# loop through all that stuff
for clip in [folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))]:
    model = YOLO(MODEL_PATH)
    yaml_path = os.path.join(DATA_PATH, clip, "data.yaml")
    metrics = model.val(data=yaml_path, device=DEVICE, save_dir=None)
    # metrics = model.val(data=yaml_path, device="mps", save_dir=None)
    mean_precision = metrics.box.mp
    mean_recall = metrics.box.mr
    map = metrics.box.map # map50-95
    map50 = metrics.box.map50
    rows.append([clip, mean_precision, mean_recall, map50, map])

# save
with open(SAVE_PATH, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(rows)