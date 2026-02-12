# Object detection
Documentation for all things object detection

## Train
Update parameters in [`train.py`](train.py) and run.

See the [Ultralytics documentation](https://docs.ultralytics.com/modes/train/) for more information.

### Usage
`python train.py`

## Predict

>NOTE:
>To use our saved model weights, download the [zipped model folders](https://console.cloud.google.com/storage/browser/nmfs_odp_afsc/RACE/MACE/salmon_pollock_object_detection/models/yolo12?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))) and unzip them in [`training`](training)

[`predict.py`](predict.py) generates detections for a directory of videos. It saves data in Ultralytics' format and includes detection confidence. The detection confidence threshold is set to 0.0.

### Usage

`MODEL_PATH`: Path to object detection model
`VIDEO_DIR`: Path to directory containing videos
`SAVE_PATH`: Path to save detections

`python predict.py [model_weight_path] [video_dir] [save_dir]`

## Evaluate

[`val.py`](val.py) evaluates model performance on the entire test set.
[`val_clipwise.py`](val_clipwise.py) evaluates model performance on on each clip in the test set. Use [`split_yolo_annotations_clipwise.py`](split_yolo_annotations_clipwise.py) to organize all test vids for clipwise evaluation (see [data_creation.md](data_creation.md)).

Some files paths are specific to the user and will need to be updated.

### Usage

`MODEL_NAME`: Name of model to use, corresponds to saved models in the [`training` folder](training)
`DATA_PATH`: Path to data for evaluation
