![tracking results video](final_results_2026/tracking_videos/quad_2019_t17_vid6_8609_8668_visualization.gif)

# Tracker optimization and evaluation for bycatch reduction


This repository contains the code used for model training, tracker optimization, evaluation, and analysis in the publication: *Towards automated bycatch monitoring: optimizing and evaluating multi-object tracking of salmon in pollock trawls*.

[[Paper link coming soon]()] | [NOAA InPort data entry](https://www.fisheries.noaa.gov/inport/item/79020)

We provide all code and data necessary to fully replicate our results or use our trained models and optimized trackers for your own applications.

The rest of this document will outline the steps to reproduce our work.

## Results

Our detailed final results can be found in [`final_results_2026`](final_results_2026).

To fully replicate our final results, you will need to:
1. Download and pre-process our dataset
2. Train and evaluate object detection models
3. Optimize and evaluate trackers

## Dependencies

### Core documents

Here are the data artifacts that are required to fully replicate our results or use our trained models and optimized trackers:

- Our dataset: [salmon and pollock trawl dataset](https://console.cloud.google.com/storage/browser/nmfs_odp_afsc/RACE/MACE/salmon_pollock_object_detection/annotated_data?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22)))
- Our test/train split for model training and tracker evaluation: [train_test_split.csv](tracking/tracking_data_generation/data/train_test_split.csv)
- Our model weights: [trained model weights](https://console.cloud.google.com/storage/browser/nmfs_odp_afsc/RACE/MACE/salmon_pollock_object_detection/models/yolo12?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22)))
- Our optimized tracker parameters: [optimal_params.yaml](final_results_2026/optimal_params.yaml)

### Repositories

We used our own forks of open-source tools (Ultralytics and TrackEval) for tracker optimization and evaluation. Our Ultralytics fork adds crucial tracking features not yet available in upstream Ultralytics: implementation of IoU and centroid trackers, and the ability to use saved detections during tracking.

- Ultralytics
    - Clone the [`all_features`](https://github.com/mlurbur/ultralytics/tree/all_features) branch of our [fork](https://github.com/mlurbur/ultralytics). For exact replication, use commit [`ae1ec96`](https://github.com/mlurbur/ultralytics/commit/ae1ec96103ae75852ee5caee4c2ad0463e4976c1).
    - Support our pull requests to upstream Ultralytics! ([`offline_tracking`](https://github.com/ultralytics/ultralytics/pull/23408), [`iou_and_centroid_trackers`](https://github.com/ultralytics/ultralytics/pull/23559), [`remove_zero_area_bbox`](https://github.com/ultralytics/ultralytics/pull/23407))
- TrackEval
    - We use [our own fork of TrackEval](https://github.com/noaa-afsc-mace/TrackEval) with minor changes to the evaluation classes. For exact replication, use commit [`bd72ef6`](https://github.com/noaa-afsc-mace/TrackEval/commit/bd72ef676ac8aa113f0269d0ae7c9c661868f313).

## Setup

1. Clone this repository
    - Install required packages using `pip install -r requirements.txt`
2. Clone the branches of Ultralytics and TrackEval linked above
3. Follow the instructions in [data_creation.md](data_creation.md) to prepare the dataset for object detection training and tracker optimization and evaluation
4. Update paths to the cloned Ultralytics and TrackEval repos and all data in [`config.yml`](config.yml)

> NOTE:
> Installing Ultralytics via `pip install ultralytics` will NOT work for this project. You must clone one of our versions that include our additional tracking features and update the above scripts with the path to the project.

## Data 

For documentation about data, see [data_creation.md](data_creation.md)

## Object detection

For documentation about model training and evaluation, see [object_detection.md](object_detection.md)

## Tracking

For documentation about tracker optimization and evaluation, see [tracker_optimization_eval.md](tracker_optimization_eval.md)

## Figures and tables

> NOTE: 
> All figure and table generating files currently use results saved in [`final_results_2026`](final_results_2026). These files are not updated during model or tracker evaluation.

Some figures rely on clip level mAP data. Generate this data using [`val_clipwise.py`](val_clipwise.py). See [object_detection.md](object_detection.md) for more details.

Figures, videos, and tables are created using four scripts:

- [`fishy_figs.py`](figure_makers/fishy_figs.py)
    Creates all figures for tracking paper. 

- [`latex_tables.py`](figure_makers/latex_tables.py)
    Creates all results-based latex tables for tracking paper.

- [`video_makers.py`](figure_makers/video_makers.py)
    Creates visualizations of tracking, plotting ground truths and predictions over videos. Creates dual and quad, and single tracker visualization videos for easy comparison of trackers

## License
This repository is released under the MIT license as found in the [LICENSE](LICENSE) file.

## Citation

If you find this repository useful, please consider giving a star ⭐ and citation 🦖:

```
@article{Lurbur_Salmon_Tracking_2026,
author = {Lurbur, Moses and Wilson, Katherine and Yochum, Noëlle},
doi = {},
journal = {Ecological Informatics},
title = {{Multi-object tracking in the trawl: a performance comparison of tracking accuracy across tracking algorithms, detection models, and frame rates}},
year = {2026}
}
```

## Disclaimer
This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project content is provided on an "as is" basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
