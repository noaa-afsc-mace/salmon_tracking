"""
Creates all figures for tracking paper. 

For figures where a single model and tracker are plotted, TRACKER_NAME and MODEL_NAME are used

Uncomment functions at the bottom of the file to create plots.

NOTE: All functions use final results, running tracker evaluation could add/change figures

Association recall vs association precision -> assre_vs_asspr()
HOTA vs frame rate -> hota_vs_fps()
HOTA vs object detection model -> hota_vs_detector()
HOTA vs clip MAP and average fish per frame -> hota_vs_plotter()
HOTA vs clip MAP or fish per frame for all trackers -> hota_vs_all_trackers("fish")
Boxplot of track len in train and test -> track_len_info() 

"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

FINAL_RESULTS_FOLDER = "final_results_2026"

TRACKER_NAME = "botsort"
MODEL_NAME = "yolo12x"
FPS = 30.0
TEST_GT_ANNOTATIONS = f"tracking/gt/mot_challenge/MOTFish_{FPS}-test"
TRAIN_GT_ANNOTATIONS = f"tracking/gt/mot_challenge/MOTFish_{FPS}-train"
 
ANNOTATION_DATA = f"{FINAL_RESULTS_FOLDER}/train_test_split.csv"
# csv of form ["sequence", "precision", "recall", "map50", "map50-95"]
MAP_DATA = f"{FINAL_RESULTS_FOLDER}/clip_level_model_performance/model_type-{MODEL_NAME}_data_source-clip_based_2_class_2_11_25_training_epochs-500_hyperparameter_source-default.csv"
SAVE_PATH = "figure_makers/plots/"

CUSTOM_SORT_DICT = {'YOLO12n': 0, 'YOLO12s': 1, 'YOLO12m': 2, 'YOLO12l': 3, 'YOLO12x': 4}

# global plotting settings

plt.style.use('fast')
colors = list(plt.rcParams['axes.prop_cycle'])
tracker_marker_dict = {"BoT-SORT":"o", "ByteTrack":"v", "IoU":"s", "Centroid":"^"}
marker_dict = {"YOLO12n":"o", "YOLO12s":"v", "YOLO12m":"s", "YOLO12l":"^", "YOLO12x":"D"}
color_dict = {"BoT-SORT":colors[0]["color"], "ByteTrack":colors[1]["color"], "IoU":colors[2]["color"], "Centroid":colors[3]["color"]}
tracker_name_map = {"botsort": "BoT-SORT", "bytetrack": "ByteTrack", "ioutrack": "IoU", "centroidtrack": "Centroid"}
dpi = 600
plt.rcParams['axes.grid'] = True

def hota_vs_detector():
    """
    Graphs HOTA performance of all trackers in data_path versus the detection model used to generate detections.

    Ignores results from max conf, defaults, and test optimized trackers.
    """
    data_path = f"{FINAL_RESULTS_FOLDER}/mot_challenge/MOTFish_30.0-test/"
    model_data_path = f"{FINAL_RESULTS_FOLDER}/validation/"

    trackers = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    tracker_data = {"BoT-SORT":[], "ByteTrack":[], "IoU":[], "Centroid":[]} # append tuples of (hota, model_name)
    for t in trackers:
        if 'default_settings' in t or "test_optimized" in t:
            continue
        df = pd.read_csv(os.path.join(data_path, t,"salmon_summary.txt"), delimiter=" ")
        df.columns = df.columns.str.rstrip()
        model_name_match = re.search(r"yolo\d+[a-zA-Z](?=_)", t)
        tracker = tracker_name_map[t.split("-")[0]]
        model_type = model_name_match.group().replace("yolo", "YOLO")
        hota = df["HOTA"].iloc[0]

        # get map
        model_match = re.search(r"model_type.*", t)
        model = model_match.group()
        df = pd.read_csv(os.path.join(model_data_path, model,f"{model}.csv"))
        df.columns = df.columns.str.rstrip()
        mAP = df["mAP50-95"].iloc[0]*100

        tracker_data[tracker].append((hota, f"{model_type} [{round(mAP,1)}]", mAP))

    for key in tracker_data.keys():
        # sorted_points = sorted(tracker_data[key], key=lambda x: CUSTOM_SORT_DICT[x[1]])
        sorted_points = sorted(tracker_data[key], key=lambda x: x[2])
        # colors = [color_dict[t[1]] for t in tracker_data[key]]
        plt.plot([t[1] for t in sorted_points], [t[0] for t in sorted_points],label=key, marker=tracker_marker_dict[key])
    
    plt.legend()
    plt.xlabel("Detection model [mAP]", fontsize=12)
    plt.ylabel("HOTA", fontsize=12)
    plt.savefig(os.path.join(SAVE_PATH, f"hota_vs_detector.png"), dpi=dpi)
    plt.clf()

def hota_vs_fps():
    """
    Graphs HOTA performance of all trackers using MODEL_NAME for detections

    Ignores results from max conf, defaults, test optimized trackers, and trackers not using MODEL_NAME.
    """
    tracker_data = {"BoT-SORT":[], "ByteTrack":[], "IoU":[], "Centroid":[]} # append tuples of (hota, fps)
    fpss = [30.0, 15.0, 10.0, 7.5]
    for fps in fpss:
        data_path = f"{FINAL_RESULTS_FOLDER}/mot_challenge/MOTFish_{fps}-test/"
        trackers = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        for t in trackers:
            if 'default_settings' in t or "test_optimized" in t or MODEL_NAME not in t:
                continue
            df = pd.read_csv(os.path.join(data_path, t,"salmon_summary.txt"), delimiter=" ")
            df.columns = df.columns.str.rstrip()
            tracker = tracker_name_map[t.split("-")[0]]
            hota = df["HOTA"].iloc[0]

            tracker_data[tracker].append((hota, fps))

    for key in tracker_data.keys():
        # sorted_points = sorted(tracker_data[key], reverse=True)
        # colors = [color_dict[t[1]] for t in tracker_data[key]]
        x = [t[1] for t in tracker_data[key]]
        y = [t[0] for t in tracker_data[key]]
        plt.plot(x, y, label=key, marker=tracker_marker_dict[key])
    custom_ticks = [7.5, 10, 15, 30]
    custom_labels = ['7.5', '10', '15', '30']
    plt.xticks(custom_ticks, custom_labels)
    plt.legend()
    plt.xlabel("Video FPS", fontsize=12)
    plt.ylabel("HOTA", fontsize=12)
    plt.savefig(os.path.join(SAVE_PATH, f"hota_vs_fps.png"), dpi=dpi)
    plt.clf()

def assre_vs_asspr():
    """
    Graphs association recall versus association precision for all trackers and models

    Ignores results from max conf, defaults, and test optimized trackers
    """
    data_path = f"{FINAL_RESULTS_FOLDER}/mot_challenge/MOTFish_30.0-test/"
    trackers = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    tracker_data = {"BoT-SORT":[], "ByteTrack":[], "IoU":[], "Centroid":[]} # append tuples of (hota, model_name)
    for t in trackers:
        if "default_settings" in t or "test_optimized" in t:
            continue
        df = pd.read_csv(os.path.join(data_path, t,"salmon_summary.txt"), delimiter=" ")
        df.columns = df.columns.str.rstrip()
        model_match = re.search(r"yolo\d+[a-zA-Z](?=_)", t)
        tracker = tracker_name_map[t.split("-")[0]]
        model_type = model_match.group().replace("yolo", "YOLO")
        assre = df["AssRe"].iloc[0]
        asspr = df["AssPr"].iloc[0]

        tracker_data[tracker].append((assre, asspr, model_type))

    models_in_data = set()
    for key in tracker_data.keys():
        for t in tracker_data[key]:
            models_in_data.add(t[2])
        for x,y,m in zip([t[1] for t in tracker_data[key]], [t[0] for t in tracker_data[key]], [marker_dict[t[2]] for t in tracker_data[key]]):
            plt.scatter(x, y, color=color_dict[key], marker=m, s=80, edgecolors='black', linewidths=1)
    
    patches = []
    for key in color_dict.keys():
        patches.append(mpatches.Patch(color=color_dict[key], label=key))
    for model in marker_dict.keys():
        if model in models_in_data:
            patches.append(Line2D([0], [0], label=model, marker=marker_dict[model], linestyle='', color="black", markerfacecolor="none"))
    plt.legend(handles=patches)
    plt.xlabel("Association precision", fontsize=12)
    plt.ylabel("Association recall", fontsize=12)
    plt.savefig(os.path.join(SAVE_PATH, f"asspre_vs_assre.png"),dpi=dpi)
    plt.clf()

def hota_vs_all_trackers(map_or_fish_per_frame):
    """
    Graphs HOTA vs clip mAP or average fish per frame for each tracker using MODEL_NAME for detections. 
    Plots line of best fit and OLS coefficients

    Uses tracker results with normal optimized settings (no test optimization or default settings)

    Args:
    map_or_fish_per_frame: str, "map" or "fish_per_frame"
    """

    trackers = ["botsort", "bytetrack", "ioutrack", "centroidtrack"]
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), layout='constrained', sharey=True)
    for i in range(len(trackers)):
        tracker_name = trackers[i]
        if map_or_fish_per_frame == "map":
            ind_var, hota = map50_95_vs_hota(tracker_name)
            model_name_pretty = MODEL_NAME.replace("yolo", "YOLO")
            tag = "map"
            x_label = f"{model_name_pretty} mAP"
        else:
            ind_var, hota = fish_per_frame_vs_hota(tracker_name)
            x_label = "Fish per frame"
            tag = "fish"
        # Create and fit model
        model = LinearRegression()
        model.fit(ind_var, hota)
        coefficients = model.coef_
        intercept = model.intercept_

        # get p value and R² via simple linear regression
        linreg = stats.linregress(np.ravel(ind_var), np.ravel(hota))
        r2 = linreg.rvalue ** 2
        p_val = linreg.pvalue
        print(f"P value of {x_label} with {tracker_name} = {round(p_val,4)}")
        print(f"Slope of {tracker_name} = {round(coefficients[0],4)}\n ------------")

        # Define parameters of the regression line
        num = len(hota)
        start = ind_var.min()
        end = ind_var.max()
        xseq = np.linspace(start, end, num=num)

        # Plot the line
        index = divmod(i, 2)
        axs[index].plot(xseq, intercept+coefficients[0]*xseq, color="black", lw=1.5)
        patches = [Line2D([0], [0], linestyle='', label=r"$R^{2}=$"+ str(round(r2,2))), \
                   Line2D([0], [0], linestyle='', label=f"$P={p_val:.2f}$")]
        axs[index].legend(handles=patches, markerscale=0, loc="upper right")
        axs[index].scatter(ind_var, hota)
        axs[index].set_title(tracker_name_map[tracker_name], fontsize=12)
    
    fig.supylabel("HOTA", fontsize=12)
    fig.supxlabel(x_label, fontsize=12)
    plt.savefig(os.path.join(SAVE_PATH, f"all_trackers_hota_vs_{tag}.png"), dpi=dpi)
    plt.clf()

def hota_vs_plotter():
    """
    Graphs HOTA vs clip mAP or average fish per frame for TRACKER_NAME using MODEL_NAME for detections. 
    Plots line of best fit and OLS coefficients

    Uses tracker results with normal optimized settings (no test optimization or default settings)
    """

    fish_per_frame, fish_per_frame_hota = fish_per_frame_vs_hota(TRACKER_NAME)
    # track_len, track_len_hota = track_len_vs_hota(TRACKER_NAME)
    map, hota = map50_95_vs_hota(TRACKER_NAME)
    hotas = [hota, fish_per_frame_hota]# , track_len_hota]
    independent_vars = [map, fish_per_frame]# , track_len]
    model_name_pretty = MODEL_NAME.replace("yolo", "YOLO")
    x_labels = [f"{model_name_pretty} mAP", "Fish per frame"]#, "Salmon track length (seconds)"]
    fig, axs = plt.subplots(1, len(x_labels), figsize=(10, 4), layout='constrained', sharey=True)

    for i in range(len(x_labels)):
        # Create and fit model
        model = LinearRegression()
        model.fit(independent_vars[i], hotas[i])
        # r2 = model.score(independent_vars[i], hotas[i])
        coefficients = model.coef_
        intercept = model.intercept_

        # get p value and R² via simple linear regression
        linreg = stats.linregress(np.ravel(independent_vars[i]), np.ravel(hotas[i]))
        r2 = linreg.rvalue ** 2
        p_val = linreg.pvalue
        print(f"P value of {x_labels[i]} = {round(p_val,4)}")

        # Define parameters of the regression line
        num = len(hota)
        start = independent_vars[i].min()
        end = independent_vars[i].max()
        xseq = np.linspace(start, end, num=num)

        # Plot the line
        axs[i].plot(xseq, intercept+coefficients[0]*xseq, color="black", lw=1.5)
        patches = [Line2D([0], [0], linestyle='', label=r"$R^{2}=$"+ str(round(r2,2))), Line2D([0], [0], linestyle='', label=f"$P={p_val:.2f}$")]
        axs[i].legend(handles=patches, markerscale=0)

    for i in range(len(x_labels)):
        axs[i].scatter(independent_vars[i], hotas[i])
        axs[i].set_xlabel(x_labels[i], fontsize=12)
    
    fig.supylabel("HOTA", fontsize=12)
    plt.savefig(os.path.join(SAVE_PATH, f"{TRACKER_NAME}_hota_vs.png"), dpi=dpi)
    plt.clf()

def fish_per_frame_vs_hota(tracker_name):
    """
    Gets average fish per frame for clips for tracker using MODEL_NAME for detections. 
    Uses tracker results with normal optimized settings (no test optimization or default settings)

    Args:
    tracker_name: tracker of interest
    """
    # load tracking data
    tracker_path = f"{FINAL_RESULTS_FOLDER}/mot_challenge/MOTFish_{FPS}-test/{tracker_name}-model_type-{MODEL_NAME}_data_source-clip_based_2_class_2_11_25_training_epochs-500_hyperparameter_source-default"
    df = pd.read_csv(os.path.join(tracker_path, "clip_data.csv"))
    clips = df['sequence'][:-1] # remove last row, not vid

    # load fish density data
    annotation_data = pd.read_csv(ANNOTATION_DATA)

    # get avg fish per frame
    avg_fish_per_frame = []
    for clip in clips:
        row_data = annotation_data[annotation_data['Annotation File'] == clip]
        total_salmon_pollock = row_data["Num Salmon Annotations"].iloc[0] + row_data["Num Pollock Annotations"].iloc[0]
        total_frames = row_data["Num Frames"]
        avg_fish_per_frame.append(total_salmon_pollock/total_frames)

    hota = np.asarray(df["HOTA"][:-1])
    avg_fish_per_frame = np.asarray(avg_fish_per_frame).reshape((-1, 1))

    return avg_fish_per_frame, hota

def track_len_vs_hota(tracker_name):
    """
    Gets average track length for clips for tracker using MODEL_NAME for detections. 
    Uses tracker results with normal optimized settings (no test optimization or default settings)

    Args:
    tracker_name: tracker of interest
    """
    # load tracking data
    tracker_path = f"{FINAL_RESULTS_FOLDER}/mot_challenge/MOTFish_{FPS}-test/{tracker_name}-model_type-{MODEL_NAME}_data_source-clip_based_2_class_2_11_25_training_epochs-500_hyperparameter_source-default"
    df = pd.read_csv(os.path.join(tracker_path, "clip_data.csv"))

    avg_track_lens = []
    vid_dirs = [entry for entry in os.listdir(TEST_GT_ANNOTATIONS) if os.path.isdir(os.path.join(TEST_GT_ANNOTATIONS, entry))]
    for dir in vid_dirs:
        gt_file_path = os.path.join(TEST_GT_ANNOTATIONS, dir, "gt/gt.txt")
        avg_track_len = calculate_average_track_length(gt_file_path)
        avg_track_lens.append(avg_track_len)

    hota = np.asarray(df["HOTA"][:-1])
    avg_track_lens = np.asarray(avg_track_lens).reshape((-1, 1))

    return avg_track_lens, hota

def track_len_info():
    """
    Plots track len for test and train GT annoations
    """

    track_lens = []
    for gt_dir in [TRAIN_GT_ANNOTATIONS, TEST_GT_ANNOTATIONS]:
        vid_dirs = [entry for entry in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, entry))]
        for dir in vid_dirs:
            gt_file_path = os.path.join(gt_dir, dir, "gt/gt.txt")
            track_lens.extend(get_track_lens_for_clip(gt_file_path))

    def timeformat(x, pos=None):
        total_seconds = x / FPS
        seconds = int(total_seconds)
        milliseconds = int((x % FPS) * (1000 / FPS))
        return f"{seconds:02d}.{milliseconds:03d}"
    
    # avg calc
    avg_track_len_frames = round(sum(track_lens) / len(track_lens),1)
    avg_track_len_seconds = avg_track_len_frames / FPS
    seconds = int(avg_track_len_seconds)
    milliseconds = int((avg_track_len_frames % FPS) * (1000 / FPS))
    avg_track_len_str = f"{seconds:02d}.{milliseconds:01d}"

    print(f"Avg track length (seconds): {avg_track_len_str}")
    print(f"Avg track length (frames): {avg_track_len_frames}")

    # max min
    max_len_frames = max(track_lens)
    min_len_frames = min(track_lens)
    min_len_seconds = min_len_frames / FPS
    max_len_seconds = max_len_frames / FPS
    min_seconds = int(min_len_seconds)
    min_milliseconds = int((min_len_frames % FPS) * (1000 / FPS))
    max_seconds = int(max_len_seconds)
    max_milliseconds = int((max_len_frames % FPS) * (1000 / FPS))
    
    min_track_len_str = f"{min_seconds:02d}.{min_milliseconds:01d}"
    max_track_len_str = f"{max_seconds:02d}.{max_milliseconds:01d}"

    print(f"Max track length (seconds): {max_track_len_str}")
    print(f"Min track length (seconds): {min_track_len_str}")

    # median calc
    median_len_frames = np.median(track_lens)
    median_len_seconds = median_len_frames / FPS
    med_seconds = int(median_len_seconds)
    med_milliseconds = int((median_len_frames % FPS) * (1000 / FPS))
    med_track_len_str = f"{med_seconds:02d}.{med_milliseconds:01d}"

    print(f"Median track length (seconds): {med_track_len_str}")
    print(f"Median track length (frames): {median_len_frames}")


    fig, ax = plt.subplots(figsize=(4, 4.8), layout='constrained')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(timeformat))
    # ax.yaxis.set_major_locator(mticker.MultipleLocator(base=30 * 60))
    
    ax.boxplot(track_lens, 0, '')
    ax.set_ylabel("Track length [ss.SSS]", fontsize=12)

    secax = ax.secondary_yaxis('right')
    secax.set_ylabel('Track length [frames]', fontsize=12)
    plt.xticks([])
    plt.savefig(os.path.join(SAVE_PATH, f"track_len_plot.png"),dpi=dpi)
    plt.clf()

def map50_95_vs_hota(tracker_name):
    """
    Gets hota and mAP for tracker using MODEL_NAME for detections. 
    Uses tracker results with normal optimized settings (no test optimization or default settings)

    Args:
    tracker_name: tracker of interest
    """

    metric = "map50-95"
    # load tracking data
    tracker_path = f"{FINAL_RESULTS_FOLDER}/mot_challenge/MOTFish_{FPS}-test/{tracker_name}-model_type-{MODEL_NAME}_data_source-clip_based_2_class_2_11_25_training_epochs-500_hyperparameter_source-default"
    df = pd.read_csv(os.path.join(tracker_path, "clip_data.csv"))
    clips = df['sequence'][:-1] # remove last row, not vid

    # load map data
    map_data = pd.read_csv(MAP_DATA)

    # get mets for clip
    mets = []
    for clip in clips:
        row_data = map_data[map_data['sequence'] == clip]
        m = row_data[metric]
        mets.append(m)

    hota = np.asarray(df["HOTA"][:-1])
    mets = np.asarray(mets).reshape((-1, 1))

    return mets, hota

def calculate_average_track_length(file_path):
    """
    Helper, calculates average track length given file_path

    Args:
    file_path: path to MOT data for clip
    """

    total_lens = get_track_lens_for_clip(file_path)

    return np.mean(total_lens)/FPS

def get_track_lens_for_clip(file_path):
    """
    Helper, returns list of track lens for a clip

    Args:
    file_path: path to MOT data for clip
    """
    tracks = {}

    with open(file_path, 'r') as file:
        for line in file:
            frame, track_id, *_ = line.strip().split(',')

            if track_id in tracks:
                tracks[track_id].append(frame)
            else:
                tracks[track_id] = [frame]

    total_lens = [len(t) for t in tracks.values()]

    return total_lens

if __name__ == "__main__":
    assre_vs_asspr()
    hota_vs_fps()
    hota_vs_detector()
    hota_vs_plotter()
    hota_vs_all_trackers("map")
    hota_vs_all_trackers("fish")
    track_len_info()