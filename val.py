"""
Runs model eval

See: 
- https://docs.ultralytics.com/integrations/onnx/#installation
- https://docs.ultralytics.com/modes/val/#usage-examples 
- https://docs.ultralytics.com/reference/utils/metrics/#ultralytics.utils.metrics.Metric
"""
import os
import csv
from ultralytics import YOLO

# ---- Change these as needed ----
MODEL_NAME = "model_type-yolo12x_data_source-clip_based_2_class_2_11_25_training_epochs-500_hyperparameter_source-default"
# ---- (told you it was important) ----

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"data_config/clip_based_2_class_2_11_25.yaml")
MODEL_PATH =  os.path.join(os.path.dirname(os.path.realpath(__file__)), f"training/{MODEL_NAME}/weights/best.pt")

model = YOLO(MODEL_PATH)
metrics = model.val(data=DATA_PATH, device=[0,1,2,3], project="validation", name=MODEL_NAME)
mean_precision = metrics.box.mp
mean_recall = metrics.box.mr
map50_95 = metrics.box.map # map50-95, area under precision recall curve (not just precision)
map50 = metrics.box.map50
salmon_precision, salmon_recall, salmon_ap50, salmon_ap50_95 = metrics.box.class_result(0) # salmon is class 0
pollock_precision, pollock_recall, pollock_ap50, pollock_ap50_95 = metrics.box.class_result(1)

# save results
columns = ["Mean precision", "Mean recall", "mAP50-95", "mAP50", "Salmon precision", "Pollock precision", \
           "Salmon recall", "Pollock recall", "Salmon AP50-95", "Pollock AP50-95", "Salmon AP50", "Pollock AP50"]
results = [mean_precision, mean_recall, map50_95, map50, salmon_precision, pollock_precision, salmon_recall, pollock_recall, salmon_ap50_95, pollock_ap50_95, salmon_ap50, pollock_ap50]

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"validation", MODEL_NAME, f"{MODEL_NAME}.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows([columns, results])
