""""
Visualization functions for debugging and interpretation of MOT
"""
import cv2
import numpy as np

def load_data(file_path):
    """
    reads text file containing data in the form:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z> 
    and returns dict
    """
    data = {}
    with open(file_path) as f:
        lines = f.readlines()
    for l in lines:
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, *rest = l.split(",")
        if int(frame) not in data:
            data[int(frame)] = [[int(float(id)), int(float(bb_left)), int(float(bb_top)), int(float(bb_width)), int(float(bb_height)), float(conf)]]
        else:
            data[int(frame)].append([int(float(id)), int(float(bb_left)), int(float(bb_top)), int(float(bb_width)), int(float(bb_height)), float(conf)])
    return data

def visualize_MOT(
    video_file,
    gt_data,
    track_data,
    save_path,
    frame_offset=0,
    annotations_only=False
):
    """
    stuff and things
    visualize single video, gets paths for everything it needs
    """
    
    # load tracks and stuff
    
    frame_num = 1
    vid = cv2.VideoCapture(video_file)
    success, frame = vid.read()
    vid_fps = vid.get(cv2.CAP_PROP_FPS)
    vid_height, vid_width, _ = frame.shape
    output_video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"jpeg"), vid_fps, (vid_width, vid_height))
    

    while success:
        if annotations_only:
            draw_frame = visualize_MOT_annotation(frame, frame_num, gt_data)
        else:
            draw_frame = visualize_MOT_boxes(frame, frame_num, gt_data, track_data)
        output_video.write(draw_frame)
        for _ in range(frame_offset + 1):
            success, frame = vid.read()
            frame_num += 1
    vid.release()
    output_video.release()

def visualize_MOT_quad(video_file, gt_data, track_data_1, tracker_name_1, track_data_2, tracker_name_2, track_data_3, tracker_name_3, track_data_4, tracker_name_4, save_path):
    """
    Creates quad frame video of the same video with 4 different tracker results being plotted

    Args:
    video_file: path to video
    gt_data: dict of gt data
    track_data_<1-4>: dict of track data for tracker
    track_name_<1-4>: string, name of tracker to be plotted on video 
    """
    tracker_name_map = {"botsort": "BoT-SORT", "bytetrack": "ByteTrack", "ioutrack": "IoU", "centroidtrack": "Centroid"}
    slow_by = 2 # factor to slow video (add duplicate frames)
    frame_num = 1
    vid = cv2.VideoCapture(video_file)
    success, frame = vid.read()
    vid_fps = vid.get(cv2.CAP_PROP_FPS)
    vid_height, vid_width, _ = frame.shape
    title_gap = 60  # Gap between title quadrants
    gap = 10 # standard gap
    new_vid_height = (2 * vid_height) + (2*title_gap) + gap
    new_vid_width = (2 * vid_width) + (3*gap)
    output_video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"jpeg"), vid_fps, (new_vid_width, new_vid_height))

    while success:
        # draw results for each tracker
        drawn_track_frames = []
        for track_data in [track_data_1, track_data_2, track_data_3, track_data_4]:
            draw_frame = visualize_MOT_boxes(frame.copy(), frame_num, gt_data, track_data)
            drawn_track_frames.append(draw_frame)

        # Create a larger canvas with gaps and write titles
        output_frame = np.ones((new_vid_height, new_vid_width, 3), dtype=np.uint8) * 255

        output_frame[title_gap:vid_height+title_gap, gap:vid_width+gap] = drawn_track_frames[0]  # Top-left quadrant
        output_frame[title_gap:vid_height+title_gap, vid_width + (2*gap):(2 * vid_width) + (2*gap)] = drawn_track_frames[1]  # Top-right quadrant
        output_frame[vid_height + (2*title_gap):(2 * vid_height) + (2*title_gap), gap:vid_width+gap] = drawn_track_frames[2]  # Bottom-left quadrant
        output_frame[vid_height + (2*title_gap):(2 * vid_height) + (2*title_gap), vid_width + (2*gap):(2 * vid_width) + (2*gap)] = drawn_track_frames[3]  # Bottom-right quadrant

        font_scale = 1.5
        thickness = 3
        # Write titles
        for i, title in enumerate([tracker_name_1, tracker_name_2, tracker_name_3, tracker_name_4]):
            title = tracker_name_map[title]
            # Get the text size
            text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

            # annoying cases for each quadrant:
            if i == 0: # first quadrant
                text_x = ((vid_width - text_size[0]) // 2) + gap
                text_y = title_gap - text_size[1] // 2
            elif i == 1:
                text_x = ((vid_width - text_size[0]) // 2) + ((2*gap)+vid_width)
                text_y = title_gap - text_size[1] // 2
            elif i == 2:
                text_x = ((vid_width - text_size[0]) // 2) + gap
                text_y = (title_gap - text_size[1] // 2) + (title_gap+vid_height)
            else:
                text_x = ((vid_width - text_size[0]) // 2) + ((2*gap)+vid_width)
                text_y = (title_gap - text_size[1] // 2) + (title_gap+vid_height)

            # black text
            output_frame = cv2.putText(output_frame, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        for i in range(slow_by):
            output_video.write(output_frame)
        success, frame = vid.read()
        frame_num += 1
    vid.release()
    output_video.release()

import cv2
import numpy as np

def visualize_MOT_dual(video_file, gt_data, track_data_1, tracker_name_1, track_data_2, tracker_name_2, save_path):
    """
    Creates side-by-side video of the same video with 2 different tracker results being plotted

    Args:
    video_file: path to video
    gt_data: dict of gt data
    track_data_1: dict of track data for tracker 1
    tracker_name_1: string, name of tracker 1 to be plotted on video
    track_data_2: dict of track data for tracker 2
    tracker_name_2: string, name of tracker 2 to be plotted on video
    save_path: path to save the output video
    """
    tracker_name_map = {"botsort": "BoT-SORT", "bytetrack": "ByteTrack", "ioutrack": "IoU", "centroidtrack": "Centroid"}
    slow_by = 2  # factor to slow video (add duplicate frames)
    frame_num = 1
    vid = cv2.VideoCapture(video_file)
    success, frame = vid.read()
    vid_fps = vid.get(cv2.CAP_PROP_FPS)
    vid_height, vid_width, _ = frame.shape
    title_gap = 60  # Gap between title and video
    gap = 10  # standard gap
    new_vid_height = vid_height + title_gap + gap
    new_vid_width = (2 * vid_width) + (3 * gap)
    output_video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"jpeg"), vid_fps, (new_vid_width, new_vid_height))

    while success:
        # Draw results for each tracker
        drawn_track_frames = []
        for track_data in [track_data_1, track_data_2]:
            draw_frame = visualize_MOT_boxes(frame.copy(), frame_num, gt_data, track_data)
            drawn_track_frames.append(draw_frame)

        # Create a larger canvas with gaps and write titles
        output_frame = np.ones((new_vid_height, new_vid_width, 3), dtype=np.uint8) * 255

        output_frame[title_gap:vid_height + title_gap, gap:vid_width + gap] = drawn_track_frames[0]  # Left video
        output_frame[title_gap:vid_height + title_gap, vid_width + (2 * gap):(2 * vid_width) + (2 * gap)] = drawn_track_frames[1]  # Right video

        font_scale = 1.5
        thickness = 3

        # Write titles
        for i, title in enumerate([tracker_name_1, tracker_name_2]):
            # Get the text size
            text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

            if i == 0:  # Left video
                text_x = ((vid_width - text_size[0]) // 2) + gap
                text_y = title_gap - text_size[1] // 2
            else:  # Right video
                text_x = ((vid_width - text_size[0]) // 2) + ((2 * gap) + vid_width)
                text_y = title_gap - text_size[1] // 2

            # Black text
            output_frame = cv2.putText(output_frame, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        for i in range(slow_by):
            output_video.write(output_frame)
        success, frame = vid.read()
        frame_num += 1

    vid.release()
    output_video.release()

def visualize_MOT_boxes(frame, frame_num, gt_full_data, track_full_data):
    """
    Draws stuff and things for the given frame
    data has all data but some frames don't exist
    """
    # gotta draw each box on it's own because there may be different numbers of each

    # start with the truth
    if frame_num not in gt_full_data:
        gt_frame_data = []
    else:
        gt_frame_data = gt_full_data[frame_num]

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.25
    text_thickness = 3
    gt_color_blue = (177, 103, 47)  # in BGR
    track_color_red = (35, 44, 191) # in BGR
    white = (255,255,255)
    gt_thickness = 4
    

    for gt_data in gt_frame_data:
        id, bb_left, bb_top, bb_width, bb_height, conf = gt_data
        # draw white outline for visibility
        frame = cv2.rectangle(
            frame,
            (bb_left, bb_top),
            (bb_width+bb_left, bb_height+bb_top),
            white,
            gt_thickness+3,
        )
        frame = cv2.rectangle(
            frame,
            (bb_left, bb_top),
            (bb_width+bb_left, bb_height+bb_top),
            gt_color_blue,
            gt_thickness,
        )

    # tracks
    if frame_num not in track_full_data:
        track_frame_data = []
    else:
        track_frame_data = track_full_data[frame_num]

    track_thickness = 3
    x_t_adj = 5
    y_t_adj = 35

    for track_data in track_frame_data:
        id, bb_left, bb_top, bb_width, bb_height, conf = track_data
        frame = cv2.rectangle(frame, (bb_left, bb_top), (bb_width+bb_left, bb_height+bb_top), white, track_thickness+3)
        frame = cv2.rectangle(frame, (bb_left, bb_top), (bb_width+bb_left, bb_height+bb_top), track_color_red, track_thickness)

        frame = cv2.putText(frame,f"{id}",(bb_left + x_t_adj, bb_top + y_t_adj),font,fontScale,white,text_thickness+3,cv2.LINE_AA,)
        frame = cv2.putText(frame,f"{id}",(bb_left + x_t_adj, bb_top + y_t_adj),font,fontScale,track_color_red,text_thickness,cv2.LINE_AA,)


    # putting opaque box and text
    text_size = cv2.getTextSize("Human annotation", font, fontScale, text_thickness)
    offset = 10
    vid_height, vid_width, _ = frame.shape
    num_lines = 3
    # First we crop the sub-rect from the image
    x, y, w, h = (
        0,
        0,
        min(text_size[0][0] + (2 * offset), vid_width),
        min((text_size[0][1] + offset) * num_lines + offset, vid_height),
    )
    sub_img = frame[y : y + h, x : x + w]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
    res = cv2.addWeighted(sub_img, 0.25, white_rect, 0.75, 1.0)
    # Putting the image back to its position
    frame[y : y + h, x : x + w] = res

    frame = cv2.putText(frame, f"Frame: {frame_num}",(offset, text_size[0][1]+offset),font,fontScale, (0,0,0), text_thickness,cv2.LINE_AA)
    frame = cv2.putText(frame, "Human annotation",(offset, (text_size[0][1]+offset)*2),font,fontScale, gt_color_blue, text_thickness,cv2.LINE_AA)
    frame = cv2.putText(frame, "Tracker",(offset, (text_size[0][1]+offset)*3),font,fontScale, track_color_red, text_thickness,cv2.LINE_AA)

    return frame


def visualize_MOT_annotation(frame, frame_num, gt_full_data):
    """
    """
    # gotta draw each box on it's own because there may be different numbers of each

    # start with the truth
    if frame_num not in gt_full_data:
        gt_frame_data = []
    else:
        gt_frame_data = gt_full_data[frame_num]

    gt_color_blue = (177, 103, 47)  # in BGR
    track_color_red = (35, 44, 191) # in BGR
    white = (255,255,255)
    gt_thickness = 3
    

    for gt_data in gt_frame_data:
        id, bb_left, bb_top, bb_width, bb_height, conf = gt_data
        # draw white outline for visibility
        frame = cv2.rectangle(
            frame,
            (bb_left, bb_top),
            (bb_width+bb_left, bb_height+bb_top),
            white,
            gt_thickness+3,
        )
        frame = cv2.rectangle(
            frame,
            (bb_left, bb_top),
            (bb_width+bb_left, bb_height+bb_top),
            gt_color_blue,
            gt_thickness,
        )

    return frame