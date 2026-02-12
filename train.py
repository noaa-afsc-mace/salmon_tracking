from ultralytics import YOLO
import os

"""
Trains yolo

See https://docs.ultralytics.com/modes/train/ for mode info
"""

PROJECT_FOLDER = "training/"
MODEL_WEIGHTS = "<path_to_pretrained_weights>.pt"
DATA = "data_config/clip_based_2_class_2_11_25.yaml"
EPOCHS = 500
NAME = f"model_type-{os.path.splitext(os.path.basename(MODEL_WEIGHTS))[0]}_data_source-{os.path.splitext(os.path.basename(DATA))[0]}_training_epochs-{EPOCHS}_hyperparameter_source-default"

# Initialize the YOLO model
model = YOLO(MODEL_WEIGHTS)

# Train model
model.train(data=DATA, epochs=EPOCHS, device=[0,1,2,3], name=NAME, project=PROJECT_FOLDER)
