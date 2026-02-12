import cv2
import os
from tqdm import tqdm

# ---- Change these as needed ----

ORIGINAL_FRAME_RATE = 30
FRAME_REDUCTION_FACTOR = 4  # 1 is original. With 30 fps, 2->15, 3->10, 4->7.5 fps
VIDEO_SAVE_DIR = "/home/shared/annotations/video_clips/"  # Adjust path as needed
ORIGINAL_VIDEO_DIR = "/home/shared/annotations/video_clips/fps-30.0/"  # Adjust path as needed

# ----

NEW_FRAME_RATE = ORIGINAL_FRAME_RATE / FRAME_REDUCTION_FACTOR
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv", ".webm")

def convert_vids(original_vid_path, new_vid_source_path):
    """
    Converts vid to new frame rate
    """
    frame_num = 0
    vid = cv2.VideoCapture(original_vid_path)
    success, frame = vid.read()
    vid_fps = vid.get(cv2.CAP_PROP_FPS)
    vid_height, vid_width, _ = frame.shape
    output_video = cv2.VideoWriter(os.path.join(new_vid_source_path, os.path.basename(original_vid_path)), 
                                  cv2.VideoWriter_fourcc(*'MJPG'), vid_fps, (vid_width, vid_height))

    while success:
        if frame_num % FRAME_REDUCTION_FACTOR == 0:
            output_video.write(frame)
        
        success, frame = vid.read()
        frame_num += 1
    vid.release()
    output_video.release()

def main(original_video_dir, save_dir, new_frame_rate, video_extensions):
    # make new video source
    new_video_source_path = os.path.join(save_dir, f"fps-{new_frame_rate}")
    if not os.path.exists(new_video_source_path):
        os.mkdir(new_video_source_path)

    all_videos = [
        entry.name for entry in os.scandir(original_video_dir)
        if entry.is_file() and entry.name.lower().endswith(video_extensions)
    ]

    # loop through videos in the original source path
    for vid_name in tqdm(all_videos):
        vid_path = os.path.join(original_video_dir, vid_name)
        convert_vids(vid_path, new_video_source_path)

if __name__ == "__main__":
    main(ORIGINAL_VIDEO_DIR, VIDEO_SAVE_DIR, NEW_FRAME_RATE, VIDEO_EXTENSIONS)
