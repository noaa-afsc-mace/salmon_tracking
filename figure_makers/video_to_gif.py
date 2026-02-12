"""
Converts a video file to an animated GIF.
"""

import cv2
from PIL import Image
from tqdm import tqdm

VIDEO_PATH = "final_results_2026/tracking_videos/quad_2019_t17_vid6_8609_8668_visualization.mov"
SAVE_PATH = "figure_makers/tracking_videos/quad_2019_t17_vid6_8609_8668_visualization.gif"
GIF_FPS = 20
SCALE = 1.0


def video_to_gif(video_path, save_path, fps=10, scale=1.0):
    """
    Encode video as an animated GIF.

    Args
    ----
    video_path : str
        Path to input video file.
    save_path : str
        Path for output GIF file.
    fps : float, optional
        Frames per second for the GIF (default 10). Lower values yield smaller files.
    scale : float, optional
        Scale factor for width/height (default 1.0). Use < 1 to shrink for smaller files.

    Returns
    -------
    None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_duration_ms = max(1, int(1000 / fps))

    frames = []
    for _ in tqdm(range(total_frames), desc="Reading frames"):
        ret, bgr = cap.read()
        if not ret:
            break
        if scale != 1.0:
            w = int(bgr.shape[1] * scale)
            h = int(bgr.shape[0] * scale)
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))

    cap.release()

    if not frames:
        raise ValueError(f"No frames read from {video_path}")

    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )


if __name__ == "__main__":
    video_to_gif(VIDEO_PATH, SAVE_PATH, fps=GIF_FPS, scale=SCALE)
    print(f"Saved: {SAVE_PATH}")
