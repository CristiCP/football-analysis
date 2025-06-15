import sys

import numpy as np
import torch
from pathlib import Path

from speed_and_distance_estimator import SpeedAndDistanceEstimator
from utils import read_video, save_video, convert_to_mp4
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer


def main(video_path="input_videos/demo.mp4"):
    # Determine base path based on current script location
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "models" / "best _model.pt"
    output_path = base_dir / "output_videos" / "output_video.avi"

    # Validate paths
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Input video not found at: {video_path}")

    # Read the video
    video_frames = read_video(video_path)

    # Initialize tracker
    tracker = Tracker(str(model_path))
    tracks = tracker.get_object_tracks(video_frames)

    # Get players positions
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.camera_movement(video_frames)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # Assign speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_track(tracks)

    # Assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            track['team'] = team
            track['team_color'] = team_assigner.team_colors.get(team, (128, 128, 128))

    # Assign ball acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    team_ball_control = np.array(team_ball_control)

    # Draw output
    output_videos_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_videos_frames = camera_movement_estimator.draw_camera_movement(output_videos_frames,
                                                                          camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_videos_frames, tracks)

    # Save the video
    save_video(output_videos_frames, str(output_path))
    avi_path = str(output_path)
    mp4_path = avi_path.replace(".avi", ".mp4")
    convert_to_mp4(avi_path, mp4_path)

    print(f"[INFO] Video saved at: {output_path}")

    print("\n[INFO] Final Speed and Distance per Player:")
    final_stats = {}

    for frame in tracks['players']:
        for player_id, data in frame.items():
            speed = data.get('speed')
            distance = data.get('distance')
            if speed is not None and distance is not None:
                final_stats[player_id] = (speed, distance)

    all_ids = set()
    for frame in tracks['players']:
        all_ids.update(frame.keys())

    for player_id in sorted(all_ids):
        if player_id in final_stats:
            speed, distance = final_stats[player_id]
            print(f"Player ID {player_id}: {speed:.2f} km/h, {distance:.2f} meters")
        else:
            print(f"Player ID {player_id}: Speed or distance not available")


if __name__ == "__main__":
    print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = Path(__file__).resolve().parent / "input_videos" / "demo.mp4"
    main(str(video_path))
