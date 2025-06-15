import subprocess

import cv2
from imageio_ffmpeg import get_ffmpeg_exe


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

def convert_to_mp4(input_path: str, output_path: str):
    ffmpeg_path = get_ffmpeg_exe()
    try:
        subprocess.run([
            ffmpeg_path,
            "-y", "-i", input_path,
            "-vcodec", "libx264", "-acodec", "aac", output_path
        ], check=True)
        print(f"[INFO] Converted to MP4 with imageio-ffmpeg: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg failed: {e}")