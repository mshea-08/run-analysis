import pandas as pd
from skillcorner.client import SkillcornerClient
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from io import BytesIO



def off_ball_run_info(runs, tracking_data):
    """
    runs: a single off-ball run from off_ball_runs_df
    tracking_data: full tracking data for the match
    returns: dictionary with run_info and tracking_frames during the run
    """
    run_info = {
    "match_id": runs["match_id"],
    "frame_start": runs["frame_start"],
    "frame_end": runs["frame_end"],
    "duration": runs["duration"],
    "event_subtype": runs.get("event_subtype"),
    "player_id": runs["player_id"],
    "team_id": runs.get("team_id"),
    "x_start": runs["x_start"],
    "y_start": runs["y_start"],
    "x_end": runs["x_end"],
    "y_end": runs["y_end"],
    "distance_covered": runs["distance_covered"],
    "trajectory_angle": runs.get("trajectory_angle"),
    "speed_avg": runs["speed_avg"],
    "n_simultaneous_runs": runs.get("n_simultaneous_runs"),
    "targeted": runs.get("targeted", False),
    "received": runs.get("received", False),
    "dangerous": runs.get("dangerous", False),
    "xthreat": runs.get("xthreat"),
    "xpass_completion": run.get("xpass_completion")
}
    # tracking frames during the run
    frame_range = range(runs["frame_start"], runs["frame_end"] + 1)
    tracking_frames = {}

    for i, frame in enumerate(frames, start=runs["frame_start"]):
        tracking_frames[i] = {
            "players": frame["player_data"],  # list of dicts
            "ball": frame["ball_data"]        # dict
        }

    return {"run_info": run_info, "tracking_frames": tracking_frames}



