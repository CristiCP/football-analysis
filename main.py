import cv2

from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner


def main():
    # Read the video
    video_frames = read_video("input_videos/demo.mp4")

    # Initialize tracker
    tracker = Tracker("models/best _model.pt")
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True,
                                       stub_path="stubs/track_stubs.pkl")

    # Assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Draw output
    # Draw object tracks
    output_videos_frames = tracker.draw_annotations(video_frames, tracks)

    # Save the video
    save_video(output_videos_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
