"""
Script for automatically making all results-based latex tables.

NOTE: Tables use saved evaluation results in `tracking/trackers/mot_challenge`, `tracking/gt`, and `tracking/clip_level_model_performance`. Changing any of these will change figures
"""

import numpy as np
import os
import sys
import re
import pandas as pd
import yaml
import configparser

sys.path.append(os.path.abspath('tracking/'))
from utils.tracking_utils import get_tracker_args

FINAL_RESULTS_FOLDER = "final_results_2026"

CUSTOM_SORT_DICT = {'YOLO12n': 0, 'YOLO12s': 1, 'YOLO12m': 2, 'YOLO12l': 3, 'YOLO12x': 4} 
TRACKER_NAME_MAP = {"botsort": "BoT-SORT", "bytetrack": "ByteTrack", "ioutrack": "IoU", "centroidtrack": "Centroid"}
PARAMS_TO_KEEP = {"track_thresh": "Track threshold", "track_buffer": "Track buffer", "match_thresh": "Match threshold", "new_track_thresh": "New track threshold", "track_high_thresh": "Track high threshold", "track_low_thresh": "Track low threshold", "fuse_score": "Fuse score"}

def custom_tracker_sort(item):
    f,s = item[0].split(" + ")
    return f, -CUSTOM_SORT_DICT.get(s, float('inf'))

def custom_param_tracker_sort(item):
    """Sort: test-optimized last; then Default then SMAC; then by model (12n, 12m, 12x)."""
    f, s = item[0].split(" + ")
    model_type = s.split(" (")[0].strip()
    is_optimized = "optimized on test" in item[0]
    param_source_rank = 0 if (len(item) > 1 and item[1] == "Default") else 1
    return (is_optimized, param_source_rank, CUSTOM_SORT_DICT.get(model_type, float("inf")), f)

def custom_fps_tracker_sort(item):
    fps = item[0]
    tracker = item[1]
    return fps, tracker

def add_hline_every_n(latex, n, num_hline=2):
    """
    Adds \hline\hline every n rows, excluding column titles and at end of table
    """
    splitter = "\\\\\n"
    # hline = "\\hline\\hline\n"
    hline = "\\hline"*num_hline + "\n"
    rows = latex.split("\\\\\n")
    new_rows = [rows[0]+splitter]
    for r in range(len(rows[1:-1])):
        if (r > 0) and ((r+1) % n == 0) and ((r+1) < len(rows)-2):
            new_rows.append(rows[1:-1][r]+splitter+hline)
        else:
            new_rows.append(rows[1:-1][r]+splitter)
    new_rows.append(rows[-1])
    return "".join(new_rows)

def highlight_all_max(column):
    """
    Highlight all maximum values in a DataFrame.
    """
    is_max = column == column.max()
    return np.where(is_max, "textbf:--rwrap;", '')

def highlight_all_min(column):
    """
    Highlight all maximum values in a DataFrame.
    """
    is_min = column == column.min()
    return np.where(is_min, "textbf:--rwrap;", '')

def format_latex_headers(latex_content, header_formats):
    """
    Applies custom multicolumn formatting to LaTeX table headers
    
    Args:
        latex_content (str): Raw LaTeX table output
        header_formats (list): Format strings for each column header
            ("" = no formatting, r"\multicolumn{1}{c|}" = apply formatting)
    
    Returns:
        str: Modified LaTeX table with formatted headers
    """
    lines = latex_content.split("\n")
    
    # Find header line (first line with & after initial hline)
    header_line_idx = None
    for i, line in enumerate(lines):
        if i+1 < len(lines) and "&" in lines[i+1]:
            header_line_idx = i+1
            break
    
    # Validation
    assert header_line_idx is not None, "No header found in LaTeX table"
    original_header = lines[header_line_idx].strip().rstrip("\\")
    columns = [col.strip() for col in original_header.split(" & ")]
    assert len(header_formats) == len(columns), \
        f"header_formats length ({len(header_formats)}) must match column count ({len(columns)})"
    
    # Build formatted headers
    formatted_columns = []
    for col, fmt in zip(columns, header_formats):
        if fmt:
            formatted_columns.append(f"{fmt}{{{col}}}")
        else:
            formatted_columns.append(col)
    
    # Replace header line
    lines[header_line_idx] = " & ".join(formatted_columns) + r" \\"
    
    return "\n".join(lines)


def make_detection_table():
    """
    Creates table of object detection results
    """
    caption = "Object detection model results with mAP, AP, \
        mAP50, precision (P), and recall (R) are shown. Best scores, based on the unrounded values, \
            are in bold. All metrics are calculated using a model confidence score threshold of 0.001."
    data_path = f"{FINAL_RESULTS_FOLDER}/validation/"
    save_path = "figure_makers/tables/model_results.tex"
    columns = ["Model", "mAP", "mAP50", "P", "R", "AP salmon", "AP pollock", "P salmon", "P pollock", "R salmon", "R pollock"]
    column_width = "|l|r|r|r|r|>{\\raggedleft\\arraybackslash}p{1.2cm}|>{\\raggedleft\\arraybackslash}p{1.2cm}|>{\\raggedleft\\arraybackslash}p{1.2cm}|>{\\raggedleft\\arraybackslash}p{1.2cm}|>{\\raggedleft\\arraybackslash}p{1.2cm}|>{\\raggedleft\\arraybackslash}p{1.2cm}|"
    header_format = [""] + ["\multicolumn{1}{c|}"]*4 + ["\multicolumn{1}{>{\centering\\arraybackslash}p{1.2cm}|}"]*6
    data = []
    models = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    for model in models:
        df = pd.read_csv(os.path.join(data_path, model,f"{model}.csv"))
        df.columns = df.columns.str.rstrip()
        match = re.search(r"yolo\d+[a-zA-Z](?=_)", model)
        model_type = match.group().replace("yolo", "YOLO")
        mAP = df["mAP50-95"].iloc[0]*100
        mAP50 = df["mAP50"].iloc[0]*100
        P = df["Mean precision"].iloc[0]*100
        R = df["Mean recall"].iloc[0]*100
        salmon_ap = df["Salmon AP50-95"].iloc[0]*100
        pollock_ap = df["Pollock AP50-95"].iloc[0]*100
        p_salmon = df["Salmon precision"].iloc[0]*100
        p_pollock = df["Pollock precision"].iloc[0]*100
        r_salmon = df["Salmon recall"].iloc[0]*100
        r_pollock = df["Pollock recall"].iloc[0]*100
        data.append([model_type, mAP, mAP50, P, R, salmon_ap, pollock_ap, p_salmon, p_pollock, r_salmon, r_pollock])
    table_df = pd.DataFrame(data, columns=columns)
    table_df = table_df.sort_values(by=['Model'], key=lambda x: x.map(CUSTOM_SORT_DICT))
    s = table_df.style.format({
        ("mAP"): '{:.1f}',
        ("mAP50"): '{:.1f}',
        ("P"): '{:.1f}',
        ("R"): '{:.1f}',
        ("AP salmon"): '{:.1f}',
        ("AP pollock"): '{:.1f}',
        ("P salmon"): '{:.1f}',
        ("P pollock"): '{:.1f}',
        ("R salmon"): '{:.1f}',
        ("R pollock"): '{:.1f}'})  
    s = s.apply(highlight_all_max, subset=["mAP", "mAP50", "P", "R", "AP salmon", "AP pollock", "P salmon", "P pollock", "R salmon", "R pollock"],axis=0).hide(axis="index")
    latex_content = s.to_latex(label="tab:model results", position="H", column_format=column_width, hrules=True, caption=caption)
    # other formatting I can't do with pandas
    latex_content = format_latex_headers(latex_content, header_format)
    latex_content = latex_content.replace("\\midrule", "\\hline")
    latex_content = latex_content.replace("\\toprule", "\\hline")
    latex_content = latex_content.replace("\\bottomrule", "")
    latex_content = latex_content.replace(r"\\", "\\\ \n \\hline")
    with open(save_path, 'w') as f:
        f.write(latex_content)

def make_tracking_results_table():
    """
    Creates table of tracking results

    Ignores max conf, test optimized and default settings trackers
    """
    caption = r"Salmon tracking results. HOTA, MOTA, IDF1, association accuracy (AssA), \
        association recall (AssRe), association precision (AssPr), detection accuracy (DetA), and localization accuracy \
        (LocA), and number of predicted track IDs are listed. The optimal value of metrics, high $\uparrow$ or low $\downarrow$, are indicated by arrows. Best scores, based on the unrounded values, are in bold. Ground truth annotations contain 51 IDs."
    data_path = f"{FINAL_RESULTS_FOLDER}/mot_challenge/MOTFish_30.0-test/"
    save_path = "figure_makers/tables/tracking_results_fps30.tex"
    columns = ["Tracker", "HOTA $\\uparrow$", "MOTA $\\uparrow$", "IDF1 $\\uparrow$", "AssA $\\uparrow$", "AssRe $\\uparrow$", "AssPr $\\uparrow$", "DetA $\\uparrow$", "LocA $\\uparrow$", "IDs"]
    header_format = [""] + ["\multicolumn{1}{c|}"]*9
    data = []
    trackers = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    for t in trackers:
        if "test_optimized" in t or "default_settings" in t:
            continue
        df = pd.read_csv(os.path.join(data_path, t,"salmon_summary.txt"), delimiter=" ")
        df.columns = df.columns.str.rstrip()
        model_match = re.search(r"yolo\d+[a-zA-Z](?=_)", t)
        tracker = TRACKER_NAME_MAP[t.split("-")[0]]
        model_type = model_match.group().replace("yolo", "YOLO")
        tracker_string = f"{tracker} + {model_type}"

        hota = df["HOTA"].iloc[0]
        mota = df["MOTA"].iloc[0]
        idf1 = df["IDF1"].iloc[0]
        ids = df["IDs"].iloc[0]
        assa = df["AssA"].iloc[0]
        assre = df["AssRe"].iloc[0]
        asspr = df["AssPr"].iloc[0]
        deta = df["DetA"].iloc[0]
        loca = df["LocA"].iloc[0]

        data.append([tracker_string, hota, mota, idf1, assa, assre, asspr, deta, loca, ids])

    # sort by track algorithm, then model
    data = sorted(data, key=custom_tracker_sort, reverse=True)
    table_df = pd.DataFrame(data, columns=columns)
    s = table_df.style.format({
        ("HOTA $\\uparrow$"): '{:.1f}',
        ("MOTA $\\uparrow$"): '{:.1f}',
        ("IDF1 $\\uparrow$"): '{:.1f}',
        ("AssA $\\uparrow$"): '{:.1f}',
        ("AssRe $\\uparrow$"): '{:.1f}',
        ("AssPr $\\uparrow$"): '{:.1f}',
        ("DetA $\\uparrow$"): '{:.1f}',
        ("LocA $\\uparrow$"): '{:.1f}'
        })  
    s = s.apply(highlight_all_max,subset=["HOTA $\\uparrow$", "MOTA $\\uparrow$", "IDF1 $\\uparrow$", "AssA $\\uparrow$", "AssRe $\\uparrow$", "AssPr $\\uparrow$", "DetA $\\uparrow$", "LocA $\\uparrow$"],axis=0).hide(axis="index") # index=False, float_format="{:.1f}".format,  
    # s = s.apply(highlight_all_min,subset=["IDs"],axis=0)
    latex_content = s.to_latex(label="tab:30fps tracking results", position="H", column_format="|l" + "|r" * (len(columns) - 1) + "|", hrules=True, caption=caption)
    # other formatting I can't do with pandas
    latex_content = format_latex_headers(latex_content, header_format)
    latex_content = latex_content.replace("\\midrule", "\\hline\\hline")
    latex_content = latex_content.replace("\\toprule", "\\hline")
    latex_content = latex_content.replace("\\bottomrule", "\\hline")
    latex_content = add_hline_every_n(latex_content, 3)
    # Get average change in HOTA from YOLO12n to YOLO12x
    pairs = [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]
    n_x_diff_list = []
    for p in pairs:
        n, m, x = p
        n_hota = round(table_df["HOTA $\\uparrow$"][n],1)
        m_hota = round(table_df["HOTA $\\uparrow$"][m],1)
        x_hota = round(table_df["HOTA $\\uparrow$"][x],1)
        n_x_diff_list.append(x_hota-n_hota)

    print(f"Average percentage-point change between YOLO12n and YOLO12x for all trackers: {round(np.mean(n_x_diff_list),1)}")
    with open(save_path, 'w') as f:
        f.write(latex_content)

def simplified_default_and_optimized_hota():
    """
    Creates simplified table showing tracking results using default versus optimized tracking parameters

    Ignores max conf and test optimized trackers
    """
    caption = r"Salmon tracking results for BoT-SORT and ByteTrack when using default parameters given by Ultralytics versus SMAC optimized parameters. Best scores, based on the unrounded values, are in bold. Ground truth annotations contain 51 IDs. For complete metrics see \autoref{tab:default and optimized results}."
    data_path = f"{FINAL_RESULTS_FOLDER}/mot_challenge/MOTFish_30.0-test/"
    save_path = "figure_makers/tables/simple_default_opt_fps30.tex"
    columns = ["Tracker", "HOTA $\\uparrow$", "MOTA $\\uparrow$", "IDF1 $\\uparrow$", "IDs"]
    data = []
    trackers = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    for t in trackers:
        if "test_optimized" in t:
            continue
        df = pd.read_csv(os.path.join(data_path, t,"salmon_summary.txt"), delimiter=" ")
        df.columns = df.columns.str.rstrip()
        model_match = re.search(r"yolo\d+[a-zA-Z](?=_)", t)
        model_type = model_match.group().replace("yolo", "YOLO")
        if "default_settings" in t:
            tracker = TRACKER_NAME_MAP[t.split("-")[1]]
            tracker_string = f"{tracker} (default) + {model_type}"
        else:
            tracker = TRACKER_NAME_MAP[t.split("-")[0]]
            tracker_string = f"{tracker} (SMAC) + {model_type}"
        if tracker not in ["BoT-SORT", "ByteTrack"]: # others don't have defaults
            continue

        hota = df["HOTA"].iloc[0]
        mota = df["MOTA"].iloc[0]
        idf1 = df["IDF1"].iloc[0]
        ids = df["IDs"].iloc[0]

        data.append([tracker_string, hota, mota, idf1, ids])

    # sort by track algorithm, then model
    data = sorted(data, key=custom_tracker_sort, reverse=True)
    table_df = pd.DataFrame(data, columns=columns)
    # Extract YOLO versions
    calc_table = table_df.copy()
    calc_table["YOLO"] = calc_table["Tracker"].str.extract(r'(YOLO\d+[nmx])')

    # Extract tracker type (ByteTrack or BoT-SORT)
    calc_table["Tracker Type"] = calc_table["Tracker"].str.extract(r'^(ByteTrack|BoT-SORT)')
    default_df = calc_table[calc_table["Tracker"].str.contains("default")].set_index(["Tracker Type", "YOLO"])
    smac_df = calc_table[calc_table["Tracker"].str.contains("SMAC")].set_index(["Tracker Type", "YOLO"])
    hota_change = smac_df["HOTA $\\uparrow$"] - default_df["HOTA $\\uparrow$"]
    hota_avg_change = hota_change.groupby("Tracker Type").mean().round(2)
    print("HOTA change (SMAC-default) for trackers:")
    print(hota_avg_change)

    s = table_df.style.format({
        ("HOTA $\\uparrow$"): '{:.1f}',
        ("MOTA $\\uparrow$"): '{:.1f}',
        ("IDF1 $\\uparrow$"): '{:.1f}',
        })  
    s = s.apply(highlight_all_max,subset=["HOTA $\\uparrow$", "MOTA $\\uparrow$", "IDF1 $\\uparrow$"],axis=0).hide(axis="index")  
    # s = s.apply(highlight_all_min,subset=["IDs"],axis=0)
    latex_content = s.to_latex(label="tab:simple default and optimized results", position="H", column_format="|l" + "|r" * (len(columns) - 1) + "|", hrules=True, caption=caption)
    # other formatting I can't do with pandas
    latex_content = latex_content.replace("\\midrule", "\\hline\\hline")
    latex_content = latex_content.replace("\\toprule", "\\hline")
    latex_content = latex_content.replace("\\bottomrule", "\\hline")
    latex_content = add_hline_every_n(latex_content, 3)
    with open(save_path, 'w') as f:
        f.write(latex_content)

def default_and_optimized_hota():
    """
    Creates full table showing tracking results using default and optimized tracking parameters

    Ignores max conf and test optimized trackers
    """
    caption = r"Complete tracking results for BoT-SORT and ByteTrack using default parameters and SMAC optimized parameters. HOTA, MOTA, IDF1, association accuracy (AssA), \
        association recall (AssRe), association precision (AssPr), detection accuracy (DetA), and localization accuracy \
        (LocA), and number of predicted track IDs are listed. The optimal value of metrics, high $\uparrow$ or low $\downarrow$, are indicated by arrows. Best scores, based on the unrounded values, are in bold. Ground truth annotations contain 51 IDs."
    data_path = f"{FINAL_RESULTS_FOLDER}/mot_challenge/MOTFish_30.0-test/"
    save_path = "figure_makers/tables/default_opt_fps30.tex"
    columns = ["Tracker", "HOTA $\\uparrow$", "MOTA $\\uparrow$", "IDF1 $\\uparrow$", "AssA $\\uparrow$", "AssRe $\\uparrow$", "AssPr $\\uparrow$", "DetA $\\uparrow$", "LocA $\\uparrow$", "IDs"]
    header_format = [""] + ["\multicolumn{1}{c|}"]*9
    data = []
    trackers = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    for t in trackers:
        if "test_optimized" in t:
            continue
        df = pd.read_csv(os.path.join(data_path, t,"salmon_summary.txt"), delimiter=" ")
        df.columns = df.columns.str.rstrip()
        model_match = re.search(r"yolo\d+[a-zA-Z](?=_)", t)
        model_type = model_match.group().replace("yolo", "YOLO")
        if "default_settings" in t:
            tracker = TRACKER_NAME_MAP[t.split("-")[1]]
            tracker_string = f"{tracker} (default) + {model_type}"
        else:
            tracker = TRACKER_NAME_MAP[t.split("-")[0]]
            tracker_string = f"{tracker} (SMAC) + {model_type}"
        if tracker not in ["BoT-SORT", "ByteTrack"]: # others don't have defaults
            continue

        hota = df["HOTA"].iloc[0]
        mota = df["MOTA"].iloc[0]
        idf1 = df["IDF1"].iloc[0]
        ids = df["IDs"].iloc[0]
        assa = df["AssA"].iloc[0]
        assre = df["AssRe"].iloc[0]
        asspr = df["AssPr"].iloc[0]
        deta = df["DetA"].iloc[0]
        loca = df["LocA"].iloc[0]

        data.append([tracker_string, hota, mota, idf1, assa, assre, asspr, deta, loca, ids])

    # sort by track algorithm, then model
    data = sorted(data, key=custom_tracker_sort, reverse=True)
    table_df = pd.DataFrame(data, columns=columns)
    s = table_df.style.format({
        ("HOTA $\\uparrow$"): '{:.1f}',
        ("MOTA $\\uparrow$"): '{:.1f}',
        ("IDF1 $\\uparrow$"): '{:.1f}',
        ("AssA $\\uparrow$"): '{:.1f}',
        ("AssRe $\\uparrow$"): '{:.1f}',
        ("AssPr $\\uparrow$"): '{:.1f}',
        ("DetA $\\uparrow$"): '{:.1f}',
        ("LocA $\\uparrow$"): '{:.1f}'
        })  
    s = s.apply(highlight_all_max,subset=["HOTA $\\uparrow$", "MOTA $\\uparrow$", "IDF1 $\\uparrow$", "AssA $\\uparrow$", "AssRe $\\uparrow$", "AssPr $\\uparrow$", "DetA $\\uparrow$", "LocA $\\uparrow$"],axis=0).hide(axis="index") # index=False, float_format="{:.1f}".format,  
    # s = s.apply(highlight_all_min,subset=["IDs"],axis=0)
    latex_content = s.to_latex(label="tab:default and optimized results", position="H", column_format="|l" + "|r" * (len(columns) - 1) + "|", hrules=True, caption=caption)
    # other formatting I can't do with pandas
    latex_content = format_latex_headers(latex_content, header_format)
    latex_content = latex_content.replace("\\midrule", "\\hline\\hline")
    latex_content = latex_content.replace("\\toprule", "\\hline")
    latex_content = latex_content.replace("\\bottomrule", "\\hline")
    latex_content = add_hline_every_n(latex_content, 3)
    with open(save_path, 'w') as f:
        f.write(latex_content)

def make_params_table(tracker):
    """
    Creates table with the optimal parameters from optimization for given tracker

    Ignores max conf and deafult settings trackers

    Args:
    tracker: tracker of interest
    """
    byte_bot_caption = f"HOTA scores and default and optimized tracker parameters for the {TRACKER_NAME_MAP[tracker]} tracker. Confidence and matching thresholds are rounded for readability. The last row shows the \"best case\" parameters and HOTA score found by optimizing \\textit{{and}} evaluating the tracker directly on the evaluation dataset.  The best HOTA score, based on the unrounded HOTA value, is in bold."
    caption = f"HOTA scores and tracker parameters from our optimization method for the {TRACKER_NAME_MAP[tracker]} tracker. Confidence and matching thresholds are rounded for readability. The last row shows the \"best case\" parameters and HOTA score found by optimizing \\textit{{and}} evaluating the tracker directly on the evaluation dataset.  The best HOTA score, based on the unrounded HOTA value, is in bold."
    bot_byte_column_width = "|l|>{\\raggedright\\arraybackslash}p{1.4cm}|>{\\raggedleft\\arraybackslash}p{1.3cm}|>{\\raggedright\\arraybackslash}p{1cm}|>{\\raggedleft\\arraybackslash}p{1.8cm}|>{\\raggedleft\\arraybackslash}p{1.8cm}|>{\\raggedleft\\arraybackslash}p{1.1cm}|>{\\raggedleft\\arraybackslash}p{1.8cm}|>{\\raggedleft\\arraybackslash}p{1.8cm}|"
    bot_byte_header_format = [""] + ["\multicolumn{1}{>{\\raggedright\\arraybackslash}p{1.4cm}|}", "\multicolumn{1}{>{\centering\\arraybackslash}p{1.3cm}|}", "\multicolumn{1}{>{\\raggedright\\arraybackslash}p{1cm}|}", "\multicolumn{1}{>{\centering\\arraybackslash}p{1.8cm}|}", "\multicolumn{1}{>{\centering\\arraybackslash}p{1.8cm}|}", "\multicolumn{1}{>{\centering\\arraybackslash}p{1.1cm}|}", "\multicolumn{1}{>{\centering\\arraybackslash}p{1.8cm}|}", "\multicolumn{1}{>{\centering\\arraybackslash}p{1.8cm}|}"]
    header_format = [""] + ["\multicolumn{1}{c|}"]*4
    data_path = f"{FINAL_RESULTS_FOLDER}/mot_challenge/MOTFish_30.0-test/"
    params_path = f"{FINAL_RESULTS_FOLDER}/optimal_params.yaml"
    save_path = f"figure_makers/tables/{tracker}_optimal_params.tex"

    if tracker in ["bytetrack", "botsort"]:
        columns = ["Tracker", "Parameter source", "HOTA $\\uparrow$"]
        header_format = bot_byte_header_format
    else:
        columns = ["Tracker", "HOTA $\\uparrow$"]

    data = []
    trackers = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    for t in trackers:
        if (tracker == "iou" or tracker == "centroid") and "default_settings" in t: # ignore default for iou and centroid
            continue
        df = pd.read_csv(os.path.join(data_path, t,"salmon_summary.txt"), delimiter=" ")
        df.columns = df.columns.str.rstrip()
        model_match = re.search(r"yolo\d+[a-zA-Z](?=_)", t)
        detection_source_match = re.search(r"model_type.*", t)
        model_type = model_match.group().replace("yolo", "YOLO")
        detection_source = detection_source_match.group()

        hota = df["HOTA"].iloc[0]
        tracker_name = TRACKER_NAME_MAP[tracker]
        if "test_optimized" in t:
            tracker_string = f"{tracker_name} + {model_type} (optimized on test)"
            detection_source = f"test_optimized-{detection_source}"
            param_source = "SMAC"
        elif "default_settings" in t:
            tracker_string = f"{tracker_name} + {model_type}"
            param_source = "Default"
        else:
            tracker_string = f"{tracker_name} + {model_type}"
            param_source = "SMAC"
        if tracker not in t:
            continue
        
        with open(params_path, 'r') as stream:
            optimal_params = yaml.safe_load(stream)

        # Support both YAML key conventions: "model_type-..." (2026) and "maxconf-False-model_type-..." (2025)
        lookup_key = detection_source
        if lookup_key not in optimal_params[tracker]:
            lookup_key = "maxconf-False-" + detection_source
        if lookup_key not in optimal_params[tracker]:
            raise KeyError(f"Params key not found for {tracker}: tried {detection_source!r} and maxconf-False-...")

        row = [tracker_string, hota]
        if tracker in ["botsort", "bytetrack"]:
            row = [tracker_string, param_source, hota]
        tracker_params = get_tracker_args(optimal_params[tracker][lookup_key])
        if f"default_settings-{tracker}" in t:
            tracker_params = optimal_params[tracker]["default"]
        tracker_params = dict(sorted(tracker_params.items()))
        for k, v in tracker_params.items():
            if k in PARAMS_TO_KEEP.keys():
                if PARAMS_TO_KEEP[k] not in columns:
                    columns.append(PARAMS_TO_KEEP[k])
                row.append(v)
        data.append(row)

    # sort by track algorithm, then model
    data = sorted(data, key=custom_param_tracker_sort, reverse=False)
    table_df = pd.DataFrame(data, columns=columns)

    s = table_df.style.format({
        ("HOTA $\\uparrow$"): '{:.1f}',
        ("Match threshold"): '{:.2f}',
        ("New track threshold"): '{:.2f}',
        ("Track high threshold"): '{:.2f}',
        ("Track low threshold"): '{:.2f}',
        ("Track threshold"): '{:.2f}',
        })  
    s = s.apply(highlight_all_max,subset=["HOTA $\\uparrow$"],axis=0).hide(axis="index")
    # get custom column wdith so things fit nice
    if tracker in ["botsort", "bytetrack"]:
        column_width = bot_byte_column_width
        caption=byte_bot_caption
    else:
        column_width = "|l" + "|r" * (len(columns) - 1) + "|"
    latex_content = s.to_latex(label=f"tab:{tracker} params", position="H", column_format=column_width, hrules=True, caption=caption)
    # other formatting I can't do with pandas
    latex_content = format_latex_headers(latex_content, header_format)
    latex_content = latex_content.replace("\\midrule", "\\hline\\hline")
    latex_content = latex_content.replace("\\toprule", "\\hline")
    latex_content = latex_content.replace("\\bottomrule", "\\hline")
    latex_content = add_hline_every_n(latex_content, 3, 1)
    with open(save_path, 'w') as f:
        f.write(latex_content)

def make_fps_table():
    """
    Creates table with tracking results for all frame rates and trackers using yolov8x. 
    
    Ignores max conf, default settings, test optimized trackers
    """
    
    caption = "Salmon tracking results for different frame rates. HOTA, MOTA, IDF1, association accuracy (AssA), \
        association recall (AssRe), association precision (AssPr), detection accuracy (DetA), and localization accuracy \
        (LocA), number of predicted track IDs, and number of IDs in ground truth annotations (GT IDs) are listed. The optimal value of metrics, high $\\uparrow$ or low $\\downarrow$, are indicated by arrows. \
        Best scores, based on the unrounded values, are in bold. Lower frame rates were simulated by downsampling videos and annotations. All trackers used YOLO12x for detections."
    data_path = f"{FINAL_RESULTS_FOLDER}/mot_challenge/"
    save_path = "figure_makers/tables/frame_rate_tracking_results.tex"
    columns = ["Frame rate","Tracker", "HOTA $\\uparrow$", "MOTA $\\uparrow$", "IDF1 $\\uparrow$", "AssA $\\uparrow$", "AssRe $\\uparrow$", "AssPr $\\uparrow$", "DetA $\\uparrow$", "LocA $\\uparrow$", "IDs", "GT IDs"]
    header_format = ["\multicolumn{1}{|c|}", ""] + ["\multicolumn{1}{c|}"]*10
    frame_rates = [30.0, 15.0, 10.0, 7.5]
    data = []

    for fps in frame_rates:
        frame_rate_dir = os.path.join(data_path, f"MOTFish_{fps}-test")
        trackers = [d for d in os.listdir(frame_rate_dir) if os.path.isdir(os.path.join(frame_rate_dir, d))]
        for t in trackers:
            if ("yolo12x" not in t) or ("default_settings" in t) or ("test_optimized" in t):
                continue
            df = pd.read_csv(os.path.join(frame_rate_dir, t,"salmon_summary.txt"), delimiter=" ")
            df.columns = df.columns.str.rstrip()
            tracker = TRACKER_NAME_MAP[t.split("-")[0]]

            hota = df["HOTA"].iloc[0]
            mota = df["MOTA"].iloc[0]
            idf1 = df["IDF1"].iloc[0]
            ids = df["IDs"].iloc[0]
            assa = df["AssA"].iloc[0]
            assre = df["AssRe"].iloc[0]
            asspr = df["AssPr"].iloc[0]
            deta = df["DetA"].iloc[0]
            loca = df["LocA"].iloc[0]
            gt_ids = df["GT_IDs"].iloc[0]

            data.append([fps, tracker, hota, mota, idf1, assa, assre, asspr, deta, loca, ids, gt_ids])

    # sort by track algorithm, then model
    data = sorted(data, key=custom_fps_tracker_sort, reverse=True)
    table_df = pd.DataFrame(data, columns=columns)
    s = table_df.style.format({
        ("Frame rate"): '{:.1f}',
        ("HOTA $\\uparrow$"): '{:.1f}',
        ("MOTA $\\uparrow$"): '{:.1f}',
        ("IDF1 $\\uparrow$"): '{:.1f}',
        ("AssA $\\uparrow$"): '{:.1f}',
        ("AssRe $\\uparrow$"): '{:.1f}',
        ("AssPr $\\uparrow$"): '{:.1f}',
        ("DetA $\\uparrow$"): '{:.1f}',
        ("LocA $\\uparrow$"): '{:.1f}'
        })  
    s = s.apply(highlight_all_max,subset=["HOTA $\\uparrow$", "MOTA $\\uparrow$", "IDF1 $\\uparrow$", "AssA $\\uparrow$", "AssRe $\\uparrow$", "AssPr $\\uparrow$", "DetA $\\uparrow$", "LocA $\\uparrow$"],axis=0).hide(axis="index")
    latex_content = s.to_latex(label="tab:frame rate tracking results", position="H", column_format="|r|l" + "|r" * (len(columns) - 2) + "|", hrules=True, caption=caption)
    # other formatting I can't do with pandas
    latex_content = format_latex_headers(latex_content, header_format)
    latex_content = latex_content.replace("\\midrule", "\\hline\\hline")
    latex_content = latex_content.replace("\\toprule", "\\hline")
    latex_content = latex_content.replace("\\bottomrule", "\\hline")
    latex_content = add_hline_every_n(latex_content, 4)
    with open(save_path, 'w') as f:
        f.write(latex_content)

def tracking_dataset_table(fps):
    """
    Makes tracking dataset table from ground truth MOT
    """

    test_path = f"tracking/gt/mot_challenge/MOTFish_{fps}-test"
    train_path = f"tracking/gt/mot_challenge/MOTFish_{fps}-train"
    save_path = f"figure_makers/tables/tracking_dataset_{fps}.tex"

    caption_head = "Tracking dataset properties." 

    if float(fps) < 30:
        caption_head = f"{fps} FPS tracking dataset properties."

    caption = caption_head + " Training data were used to optimize tracker parameters. Testing data were used for tracker evaluation. The test and train sets are identical to the object detection dataset, except clips without salmon tracks were removed."

    def parse_gt_file(gt_path):
        df = pd.read_csv(gt_path, header=None)
        df.columns = ['frame', 'track_id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis1', 'vis2']
        num_tracks = df['track_id'].nunique()
        return num_tracks

    def get_seq_length(video_dir):
        ini_path = os.path.join(video_dir, "seqinfo.ini")
        config = configparser.ConfigParser()
        config.read(ini_path)
        return int(config['Sequence']['seqLength'])

    def gather_stats(dir_path, split_name):
        video_dirs = [os.path.join(dir_path, v) for v in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, v))]

        total_tracks = 0
        total_frames = 0
        total_videos = 0

        for video_dir in video_dirs:
            gt_path = os.path.join(video_dir, "gt", "gt.txt")
            if not os.path.exists(gt_path):
                continue

            try:
                num_tracks = parse_gt_file(gt_path)
                seq_len = get_seq_length(video_dir)
            except Exception:
                continue

            if num_tracks > 0:
                total_tracks += num_tracks
                total_frames += seq_len
                total_videos += 1

        return total_tracks, total_frames, total_videos

    train_stats = gather_stats(train_path, "train")
    test_stats = gather_stats(test_path, "test")

    total_stats = (
        train_stats[0] + test_stats[0],
        train_stats[1] + test_stats[1],
        train_stats[2] + test_stats[2]
    )

    def format_value(val, total):
        pct = (val / total * 100) if total > 0 else 0
        return f"{val:,} ({pct:.0f}\\%)"

    data = [
        ["Train", format_value(train_stats[0], total_stats[0]),
                 format_value(train_stats[1], total_stats[1]),
                 format_value(train_stats[2], total_stats[2])],
        ["Test", format_value(test_stats[0], total_stats[0]),
                 format_value(test_stats[1], total_stats[1]),
                 format_value(test_stats[2], total_stats[2])],
        ["Total", f"{total_stats[0]:,}", f"{total_stats[1]:,}", f"{total_stats[2]:,}"]
    ]

    df = pd.DataFrame(data, columns=["Dataset split", "Salmon tracks", "Frames", "Video clips"])
    header_format = ["", "\multicolumn{1}{c|}", "\multicolumn{1}{c|}", "\multicolumn{1}{c|}"]
    styled = df.style.hide(axis="index")

    latex_content = styled.to_latex(
        label=f"tab:tracking_dataset_{fps}",
        position="H",
        column_format="|l|r|r|r|",
        hrules=True,
        caption=caption
    )
    latex_content = format_latex_headers(latex_content, header_format)
    latex_content = latex_content.replace("\\midrule", "\\hline\\hline")
    latex_content = latex_content.replace("\\toprule", "\\hline")
    latex_content = latex_content.replace("\\bottomrule", "\\hline")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(latex_content)


def detection_dataset_table():
    """
    Makes detection dataset table
    """
    
    clip_info_path = f"{FINAL_RESULTS_FOLDER}/clip_info.csv"
    train_test_split_path = f"{FINAL_RESULTS_FOLDER}/train_test_split.csv"
    save_path = "figure_makers/tables/full_dataset.tex"
   
    # Load CSV
    clip_info = pd.read_csv(clip_info_path)
    train_test_split = pd.read_csv(train_test_split_path)
    
    # Merge data
    merged_clip_info = clip_info.merge(
        train_test_split[['Annotation File', 'Excluded', 'Test', 'Train']], 
        on='Annotation File', 
        how='left'
    )

    # Compute totals
    stats = {
        "Train Salmon Tracks": merged_clip_info.loc[merged_clip_info['Train'] == True, 'Num Salmon Tracks'].sum(),
        "Test Salmon Tracks": merged_clip_info.loc[merged_clip_info['Test'] == True, 'Num Salmon Tracks'].sum(),
        "Train Salmon Annotations": merged_clip_info.loc[merged_clip_info['Train'] == True, 'Num Salmon Annotations'].sum(),
        "Test Salmon Annotations": merged_clip_info.loc[merged_clip_info['Test'] == True, 'Num Salmon Annotations'].sum(),
        "Train Pollock Annotations": merged_clip_info.loc[merged_clip_info['Train'] == True, 'Num Pollock Annotations'].sum(),
        "Test Pollock Annotations": merged_clip_info.loc[merged_clip_info['Test'] == True, 'Num Pollock Annotations'].sum(),
        "Train Frames": merged_clip_info.loc[merged_clip_info['Train'] == True, 'Num Frames'].sum(),
        "Test Frames": merged_clip_info.loc[merged_clip_info['Test'] == True, 'Num Frames'].sum(),
        "Train Video Clips": merged_clip_info['Train'].sum(),
        "Test Video Clips": merged_clip_info['Test'].sum(),
    }

    # Compute totals
    total_stats = {key.replace("Train ", "").replace("Test ", ""): stats["Train " + key.replace("Train ", "")] + stats["Test " + key.replace("Train ", "")] for key in stats.keys() if "Train" in key}

    # Function to format numbers with percentages
    def format_value(value, total):
        percent = (value / total * 100) if total > 0 else 0
        return f"{value:,} ({percent:.0f}\%)"

    # Create DataFrame for LaTeX table
    columns = ["Dataset split", "Salmon tracks", "Salmon annotations", "Pollock annotations", "Frames", "Video clips"]
    header_format = ["", "\multicolumn{1}{c|}", "\multicolumn{1}{c|}", "\multicolumn{1}{c|}", "\multicolumn{1}{c|}", "\multicolumn{1}{c|}"]
    data = [
        ["Train",format_value(stats["Train Salmon Tracks"], total_stats["Salmon Tracks"]),
                 format_value(stats["Train Salmon Annotations"], total_stats["Salmon Annotations"]), 
                 format_value(stats["Train Pollock Annotations"], total_stats["Pollock Annotations"]), 
                 format_value(stats["Train Frames"], total_stats["Frames"]),
                 format_value(stats["Train Video Clips"], total_stats["Video Clips"])],
        
        ["Test", format_value(stats["Test Salmon Tracks"], total_stats["Salmon Tracks"]),
                    format_value(stats["Test Salmon Annotations"], total_stats["Salmon Annotations"]), 
                 format_value(stats["Test Pollock Annotations"], total_stats["Pollock Annotations"]), 
                 format_value(stats["Test Frames"], total_stats["Frames"]),
                 format_value(stats["Test Video Clips"], total_stats["Video Clips"])],
        
        ["Total", f"{total_stats['Salmon Tracks']:,}", 
                  f"{total_stats['Salmon Annotations']:,}", 
                  f"{total_stats['Pollock Annotations']:,}",  
                  f"{total_stats['Frames']:,}",
                  f"{total_stats['Video Clips']:,}",]
    ]
    
    table_df = pd.DataFrame(data, columns=columns)

    # Format table
    s = table_df.style.hide(axis="index")
    
    # Convert to LaTeX
    latex_content = s.to_latex(
        label="tab:detection_dataset", 
        position="H", 
        column_format="|l" + "|r" * (len(columns) - 1) + "|", 
        hrules=True, 
        caption="Object detection dataset properties. Training data were used to fine-tune object detection models. Test data were used for model evaluation."
    )
    
    latex_content = format_latex_headers(latex_content, header_format)
    # Additional formatting
    latex_content = latex_content.replace("\\midrule", "\\hline\\hline")
    latex_content = latex_content.replace("\\toprule", "\\hline")
    latex_content = latex_content.replace("\\bottomrule", "\\hline")
    
    # Write to file
    with open(save_path, 'w') as f:
        f.write(latex_content)


def make_default_opt_fps_table():
    """
    Creates a table comparing the default and SMAC (optimized) performance of BoT-SORT and ByteTrack
    for all frame rates using YOLO12x detections.
    """
    caption = ("Comparison of default and SMAC parameters for BoT-SORT and ByteTrack across different frame rates. The optimal value of metrics, high $\\uparrow$ or low $\\downarrow$, are indicated by arrows. \
        Best scores, based on the unrounded values, are in bold. Lower frame rates were simulated by downsampling videos and annotations. All trackers used YOLO12x for detections. The number of IDs in ground truth annotations (GT IDs) are given for each frame rate.")
    data_path = f"{FINAL_RESULTS_FOLDER}/mot_challenge/"
    save_path = "figure_makers/tables/default_opt_fps_table.tex"
    columns = ["Frame rate", "Tracker", "Parameter", "HOTA $\\uparrow$", "MOTA $\\uparrow$",
               "IDF1 $\\uparrow$", "AssA $\\uparrow$", "AssRe $\\uparrow$",
               "AssPr $\\uparrow$", "DetA $\\uparrow$", "LocA $\\uparrow$", "IDs", "GT IDs"]
    header_format = ["\multicolumn{1}{|c|}", "", ""] + ["\multicolumn{1}{c|}"]*10
    frame_rates = [30.0, 15.0, 10.0, 7.5]
    data = []
    
    for fps in frame_rates:
        fps_dir = os.path.join(data_path, f"MOTFish_{fps}-test")
        trackers = [d for d in os.listdir(fps_dir) if os.path.isdir(os.path.join(fps_dir, d))]
        for t in trackers:
            if ("yolo12x" not in t) or ("test_optimized" in t):
                continue
            # Use the first part of the folder name to map tracker
            if "default_settings" in t:
                tracker = TRACKER_NAME_MAP[t.split("-")[1]]
            else:
                tracker = TRACKER_NAME_MAP[t.split("-")[0]]

            if tracker not in ["BoT-SORT", "ByteTrack"]:
                continue
            
            # Determine parameter setting from folder name
            parameter = "Default" if "default_settings" in t else "SMAC"
            
            summary_file = os.path.join(fps_dir, t, "salmon_summary.txt")
            df = pd.read_csv(summary_file, delimiter=" ")
            df.columns = df.columns.str.rstrip()
            hota = df["HOTA"].iloc[0]
            mota = df["MOTA"].iloc[0]
            idf1 = df["IDF1"].iloc[0]
            ids = df["IDs"].iloc[0]
            assa = df["AssA"].iloc[0]
            assre = df["AssRe"].iloc[0]
            asspr = df["AssPr"].iloc[0]
            deta = df["DetA"].iloc[0]
            loca = df["LocA"].iloc[0]
            gt_ids = df["GT_IDs"].iloc[0] if "GT_IDs" in df.columns else None
            
            data.append([fps, tracker, parameter, hota, mota, idf1,
                         assa, assre, asspr, deta, loca, ids, gt_ids])
    
    # Optionally sort the table (custom sort function can be defined as needed)
    table_df = pd.DataFrame(data, columns=columns)
    table_df.sort_values(by=["Frame rate", "Tracker", "Parameter"], ascending=[False, True, True], inplace=True)
    
    # calc diff
    copy_calc_table = table_df.copy()
    pivot = copy_calc_table.pivot_table(index=['Frame rate', 'Tracker'],
                        columns='Parameter',
                        values="HOTA $\\uparrow$")
    # compute difference: default - SMAC
    pivot['HOTA SMAC - default'] = pivot['SMAC'] - pivot['Default']
    result = pivot['HOTA SMAC - default'].reset_index()
    print(result)
    
    s = table_df.style.format({
        "Frame rate": '{:.1f}',
        "HOTA $\\uparrow$": '{:.1f}',
        "MOTA $\\uparrow$": '{:.1f}',
        "IDF1 $\\uparrow$": '{:.1f}',
        "AssA $\\uparrow$": '{:.1f}',
        "AssRe $\\uparrow$": '{:.1f}',
        "AssPr $\\uparrow$": '{:.1f}',
        "DetA $\\uparrow$": '{:.1f}',
        "LocA $\\uparrow$": '{:.1f}'
    })
    s = s.apply(highlight_all_max, subset=["HOTA $\\uparrow$", "MOTA $\\uparrow$", "IDF1 $\\uparrow$",
                                             "AssA $\\uparrow$", "AssRe $\\uparrow$", "AssPr $\\uparrow$",
                                             "DetA $\\uparrow$", "LocA $\\uparrow$"], axis=0).hide(axis="index")
    # s = s.apply(highlight_all_min, subset=["IDs"], axis=0)
    
    latex_content = s.to_latex(label="tab:default_opt_fps", position="H",
                               column_format="|r|l|l" + "|r" * (len(columns) - 3) + "|",
                               hrules=True, caption=caption)
    latex_content = format_latex_headers(latex_content, header_format)
    latex_content = latex_content.replace("\\midrule", "\\hline\\hline")
    latex_content = latex_content.replace("\\toprule", "\\hline")
    latex_content = latex_content.replace("\\bottomrule", "\\hline")
    latex_content = add_hline_every_n(latex_content, 4)
    
    with open(save_path, 'w') as f:
        f.write(latex_content)

def make_default_opt_fps_table_simplified():
    """
    Creates a simplified table comparing the default and SMAC (optimized) performance of BoT-SORT and ByteTrack
    for all frame rates using YOLO12x detections.
    """
    caption = ("Comparison of default and SMAC parameters for BoT-SORT and ByteTrack across different frame rates. The optimal value of metrics, high $\\uparrow$ or low $\\downarrow$, are indicated by arrows. \
        Best scores, based on the unrounded values, are in bold. Lower frame rates were simulated by downsampling videos and annotations. All trackers used YOLO12x for detections. The number of IDs in ground truth annotations (GT IDs) are given for each frame rate. For detailed results see \\autoref{tab:default_opt_fps}.")
    data_path = f"{FINAL_RESULTS_FOLDER}/mot_challenge/"
    save_path = "figure_makers/tables/default_opt_fps_table_simplified.tex"
    columns = ["Frame rate", "Tracker", "Parameter", "HOTA $\\uparrow$", "MOTA $\\uparrow$",
               "IDF1 $\\uparrow$", "IDs", "GT IDs"]
    header_format = [""] + [""]*2 + ["\multicolumn{1}{c|}"]*5
    frame_rates = [30.0, 15.0, 10.0, 7.5]
    data = []
    
    for fps in frame_rates:
        fps_dir = os.path.join(data_path, f"MOTFish_{fps}-test")
        trackers = [d for d in os.listdir(fps_dir) if os.path.isdir(os.path.join(fps_dir, d))]
        for t in trackers:
            if ("yolo12x" not in t) or ("test_optimized" in t):
                continue
            # Use the first part of the folder name to map tracker
            if "default_settings" in t:
                tracker = TRACKER_NAME_MAP[t.split("-")[1]]
            else:
                tracker = TRACKER_NAME_MAP[t.split("-")[0]]

            if tracker not in ["BoT-SORT", "ByteTrack"]:
                continue
            
            # Determine parameter setting from folder name
            parameter = "Default" if "default_settings" in t else "SMAC"
            
            summary_file = os.path.join(fps_dir, t, "salmon_summary.txt")
            df = pd.read_csv(summary_file, delimiter=" ")
            df.columns = df.columns.str.rstrip()
            hota = df["HOTA"].iloc[0]
            mota = df["MOTA"].iloc[0]
            idf1 = df["IDF1"].iloc[0]
            ids = df["IDs"].iloc[0]
            assa = df["AssA"].iloc[0]
            assre = df["AssRe"].iloc[0]
            asspr = df["AssPr"].iloc[0]
            deta = df["DetA"].iloc[0]
            loca = df["LocA"].iloc[0]
            gt_ids = df["GT_IDs"].iloc[0] if "GT_IDs" in df.columns else None
            
            data.append([fps, tracker, parameter, hota, mota, idf1, ids, gt_ids])
    
    # Optionally sort the table (custom sort function can be defined as needed)
    table_df = pd.DataFrame(data, columns=columns)
    table_df.sort_values(by=["Frame rate", "Tracker", "Parameter"], ascending=[False, True, True], inplace=True)
    
    s = table_df.style.format({
        "Frame rate": '{:.1f}',
        "HOTA $\\uparrow$": '{:.1f}',
        "MOTA $\\uparrow$": '{:.1f}',
        "IDF1 $\\uparrow$": '{:.1f}',
    })
    s = s.apply(highlight_all_max, subset=["HOTA $\\uparrow$", "MOTA $\\uparrow$", "IDF1 $\\uparrow$"], axis=0).hide(axis="index")
    # s = s.apply(highlight_all_min, subset=["IDs"], axis=0)
    
    latex_content = s.to_latex(label="tab:default_opt_fps_simplified", position="H",
                               column_format="|r|l|l" + "|r" * (len(columns) - 3) + "|",
                               hrules=True, caption=caption)
    latex_content = format_latex_headers(latex_content, header_format)
    latex_content = latex_content.replace("\\midrule", "\\hline\\hline")
    latex_content = latex_content.replace("\\toprule", "\\hline")
    latex_content = latex_content.replace("\\bottomrule", "\\hline")
    latex_content = add_hline_every_n(latex_content, 4)
    
    with open(save_path, 'w') as f:
        f.write(latex_content)



if __name__ == "__main__":
    make_params_table("botsort")
    make_params_table("bytetrack")
    make_params_table("ioutrack")
    make_params_table("centroidtrack")
    make_fps_table()
    make_tracking_results_table()
    default_and_optimized_hota()
    # simplified_default_and_optimized_hota() # not used in current manuscript
    make_detection_table()
    detection_dataset_table()

    tracking_dataset_table("30.0")
    tracking_dataset_table("15.0")
    tracking_dataset_table("10.0")
    tracking_dataset_table("7.5")

    make_default_opt_fps_table()
    make_default_opt_fps_table_simplified()