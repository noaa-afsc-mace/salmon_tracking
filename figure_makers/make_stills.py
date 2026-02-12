import os
import cv2

VIDEOS_TO_THUMBNAIL = ["final_results_2026/tracking_videos/quad_2019_t17_vid15_9089_9148_visualization.mov",
                   "final_results_2026/tracking_videos/quad_2019_t17_vid15_15119_15178_visualization.mov",
                   "final_results_2026/tracking_videos/quad_2019_t22_vid2_12954_12975_visualization.mov"]

VIDEOS_TO_TRI_FRAME = ["final_results_2026/tracking_videos/2019_t17_vid15_9089_9148_botsort_visualization.mov"]

SAVE_FOLDER = "figure_makers/stills"


def make_thumbnails():
    """
    Saves first frame of each video as a thumbnail
    """
    for video_path in VIDEOS_TO_THUMBNAIL:
        vid = cv2.VideoCapture(video_path)
        success, frame = vid.read()
        vid.release()
        if not success:
            print(f"Failed to read {video_path}")
            continue
        basename = os.path.splitext(os.path.basename(video_path))[0]
        save_path = os.path.join(SAVE_FOLDER, f"thumbnail_{basename}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"Saved {save_path}")


def make_tri_frames():
    """
    Saves first, middle, and last frames of each video
    """
    for video_path in VIDEOS_TO_TRI_FRAME:
        vid = cv2.VideoCapture(video_path)
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.splitext(os.path.basename(video_path))[0]

        frame_indices = [0, total_frames // 2, total_frames - 1]
        for i, frame_idx in enumerate(frame_indices, start=1):
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = vid.read()
            if not success:
                print(f"Failed to read frame {frame_idx} from {video_path}")
                continue
            save_path = os.path.join(SAVE_FOLDER, f"img{i}_{basename}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"Saved {save_path}")
        vid.release()


if __name__ == "__main__":
    make_thumbnails()
    make_tri_frames()
