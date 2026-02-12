# Tracking optimization and evaluation
This guide describes how to optimize and evaluate trackers

## Tracker optimization

We used our own forks of open-source tools (Ultralytics and TrackEval) for tracker optimization and evaluation. Our Ultralytics fork adds crucial tracking features not yet available in upstream Ultralytics: implementation of IoU and centroid trackers, and the ability to use saved detections during tracking.

- Ultralytics
    - Clone the [`all_features`](https://github.com/mlurbur/ultralytics/tree/all_features) branch of our [fork](https://github.com/mlurbur/ultralytics). For exact replication, use commit [`ae1ec96`](https://github.com/mlurbur/ultralytics/commit/ae1ec96103ae75852ee5caee4c2ad0463e4976c1).
    - Support our pull requests to upstream Ultralytics! ([`offline_tracking`](https://github.com/ultralytics/ultralytics/pull/23408), [`iou_and_centroid_trackers`](https://github.com/ultralytics/ultralytics/pull/23559), [`remove_zero_area_bbox`](https://github.com/ultralytics/ultralytics/pull/23407))
- TrackEval
    - We use [our own fork of TrackEval](https://github.com/noaa-afsc-mace/TrackEval) with minor changes to the evaluation classes. For exact replication, use commit [`bd72ef6`](https://github.com/noaa-afsc-mace/TrackEval/commit/bd72ef676ac8aa113f0269d0ae7c9c661868f313).

### Optimization

1. Generate detections. Because we have to create detection data for a several frames rates, there are a few steps

    1. First, we use [`predict.py`](predict.py) to create detections for the original 30fps videos.

    See [object_detection.md](object_detection.md) for more documentation

    2. Next, we downsample videos and detections to simulate lower frame rates (15, 10, and 7.5 fps):
        - Downsample videos with [`convert_video_frame_rate.py`](tracking/tracking_data_generation/frame_rate/convert_video_frame_rate.py). Videos only need to be downsampled once per frame rate.
        - Downsample detections with [`convert_detection_frame_rate.py`](tracking/tracking_data_generation/frame_rate/convert_detection_frame_rate.py). Detections must be downsampled for every model.

    See [data_creation.md](data_creation.md#downsample-videos-and-detections-for-lower-frame-rates-optional) for complete documentation

2. Optimize trackers using [`smac_optimize_track.py`](tracking/smac_optimize_track.py)

    **To run all 28 jobs in parallel** (recommended), use the launch script:
    ```
    bash tracking/launch_all_optimizations.sh [gpu_ids]
    ```
    This generates per-window YAML configs and launches 14 tmux windows (tracker pairs run sequentially within each window). Jobs are distributed round-robin across GPUs. Pass comma-separated GPU IDs to select specific GPUs (e.g. `1,2`); defaults to all available GPUs. Monitor with `tmux attach -t salmon_opt`.

    **To run a single job manually**, create or edit a job YAML and pass it as an argument:
    ```
    python tracking/smac_optimize_track.py tracking/optimization_jobs.yaml
    ```

    Optimization can take a long time, fyi. Use `tmux`.

    See [documentation](#smac_optimize_trackpy) for more documentation

3. Save optimal parameters

    Once you are happy with the results of optimization, you will want to save the best tracker parameters
    1. Copy/paste the relative path of the optimization results file saved in [`tracking/optimization`](tracking/optimization) to the appropriate `source:` key in [`optimal_params.yaml`](tracking/optimal_params.yaml)

    2. Run [`get_optimal_params.py`](tracking/get_optimal_params.py) to automatically populate tracker parameters

    NOTE: be careful when running `get_optimal_params.py` as it can easily overwrite other parts of the file.

    See [documentation](#get_optimal_paramspy) for more info


### Evaluation

Evaluate trackers using [`eval_track.py`](tracking/eval_track.py)

**To run all jobs in parallel** (recommended), use the launch script:
```
bash tracking/launch_all_evals.sh [gpu_ids]
```
This generates per-window YAML configs and launches 20 tmux windows (tracker pairs run sequentially within each window). Jobs are distributed round-robin across GPUs. Pass comma-separated GPU IDs to select specific GPUs (e.g. `1,2`); defaults to all available GPUs. Monitor with `tmux attach -t salmon_eval`.

**To run a single job manually**, create or edit a job YAML and pass it as an argument:
```
python tracking/eval_track.py tracking/eval_jobs.yaml
```

See [documentation](#eval_trackpy) for more info

## File documentation

### [`eval_track.py`](tracking/eval_track.py)

`eval_track.py` runs full tracker evaluation using params saved in `optimal_params.yaml`. Results are saved in [tracking/trackers/mot_challenge](tracking/trackers/mot_challenge).

A summary of metrics is saved under the name `salmon_summary.txt`.

#### Known issues
None

#### Usage

The script accepts an optional job config YAML as a CLI argument (defaults to `eval_jobs.yaml`):

```
python tracking/eval_track.py [eval_jobs.yaml]
```

Job configuration YAML fields:
- `trackers`: list of trackers to evaluate
- `mode`: data split to evaluate on ("train" or "test")
- `detection_source_template`: template for detection source names
- `jobs`: list of model + fps + params_source combinations to evaluate

`params_source` controls which parameters are loaded from `optimal_params.yaml`:
- `"train_optimized"` (default) - params optimized on training data
- `"test_optimized"` - params optimized on test data (for validation of optimization method)
- `"default"` - default ultralytics tracker params (only bytetrack & botsort)

#### Parallel execution

To run all tracker/model/fps/params_source combinations in parallel, use [`launch_all_evals.sh`](tracking/launch_all_evals.sh):

```
bash tracking/launch_all_evals.sh [gpu_ids]
```

Pass comma-separated GPU IDs to select specific GPUs (e.g. `bash tracking/launch_all_evals.sh 1,2`); defaults to all available GPUs. Jobs are distributed round-robin across the selected GPUs.

This generates per-window YAML configs in `tracking/eval_jobs/` and launches 20 tmux windows in a `salmon_eval` session. Standard jobs run tracker pairs (botsort+ioutrack or bytetrack+centroidtrack) sequentially; default-params jobs run botsort+bytetrack only. Requires pyenv with a `salmon_tracking` environment.

```
tmux attach -t salmon_eval          # monitor
tmux list-windows -t salmon_eval    # check status
```

### [`get_optimal_params.py`](tracking/get_optimal_params.py)

`get_optimal_params.py` auto populates `optimal_params.yaml` based on source for each tracker/model configuration. It ignores configs labeled as "default". It checks that the model name in the source path matches that of the model in the config.

#### Known issues
None

#### Usage

```
python get_optimal_params.py
```

### [`smac_optimize_track.py`](tracking/smac_optimize_track.py)

`smac_optimize_track.py` uses the SMAC3 optimization package to determine optimal tracker parameters. Search ranges for parameters are hard-coded in the `SEARCH_DICT`, update them if necessary. Paths are configured in `config.yml`. Optimization can take a long time, use `tmux`.

Results for each tracker are date-tagged and saved in a .txt file in [`tracking/optimization`](tracking/optimization). Contents are ordered by HOTA score.

#### Known issues
None

#### Usage

The script accepts an optional job config YAML as a CLI argument (defaults to `optimization_jobs.yaml`):

```
python tracking/smac_optimize_track.py [job_config.yaml]
```

Job configuration YAML fields:
- `trackers`: list of trackers to optimize
- `mode`: data to use for optimization ("train" or "test")
- `num_search_iterations`: number of SMAC iterations
- `detection_source_template`: template for detection source names
- `jobs`: list of model + fps combinations to optimize

#### Parallel execution

To run all tracker/model/fps/mode combinations in parallel, use [`launch_all_optimizations.sh`](tracking/launch_all_optimizations.sh):

```
bash tracking/launch_all_optimizations.sh [gpu_ids]
```

Pass comma-separated GPU IDs to select specific GPUs (e.g. `bash tracking/launch_all_optimizations.sh 1,2`); defaults to all available GPUs. Jobs are distributed round-robin across the selected GPUs.

This generates per-window YAML configs in `tracking/optimization_jobs/` and launches 14 tmux windows in a `salmon_opt` session. Each window runs a pair of trackers (botsort+ioutrack or bytetrack+centroidtrack) sequentially. Requires pyenv with a `salmon_tracking` environment.

```
tmux attach -t salmon_opt          # monitor
tmux list-windows -t salmon_opt    # check status
```
