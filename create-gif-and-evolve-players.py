# Import Packages
import pandas as pd
from skillcorner.client import SkillcornerClient
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image as IPimg 
from IPython.display import display 

#function to create off the ball run gifs
def create_run_gif(run_index, off_ball_runs_df, match_tracking_data, players_df,
                   home_team, away_team, pitch, duration=125):
    """
    run_index : int
        index of the off the ball run of interest
    off_ball_runs_df : df
        dataframe of off the ball runs
    match_tracking_data : df
        match tracking data df
    players_df : df
        dataframe of the players info
    home_team : df
        dataframe of players on home team
    """
    
    # Load run
    run_for_gif = off_ball_runs_df.loc[[run_index]]
    run_player = run_for_gif['player_id'].iloc[0]

    first_frame = int(run_for_gif['frame_start'].iloc[0])
    last_frame  = int(run_for_gif['frame_end'].iloc[0])

    # frames during which run occurred
    match_data = pd.DataFrame(match_tracking_data)
    frames = match_data.loc[
        (match_data['frame'] >= first_frame) &
        (match_data['frame'] <= last_frame),
        'frame'
    ]
    frames_to_render = frames.tolist()

    pil_frames = []

    for f in frames_to_render:
        current_time_stamp_data = match_tracking_data[f]
        current_player_positions = pd.DataFrame(current_time_stamp_data['player_data'])

        # --- draw pitch ---
        fig, ax = pitch.draw()

        # plot players
        for _, player in current_player_positions.iterrows():
            pid = player.player_id
            x, y = player.x, player.y

            # team color logic
            team = players_df.loc[players_df.id == pid, 'team_id'].iloc[0]
            if pid == run_player and team == home_team.id.iloc[0]:
                ax.scatter(x, y, color="red", marker= "*") #can take this out if you want to not make the color of the player making the run different
            elif pid == run_player and team == away_team.id.iloc[0]:
                ax.scatter(x, y, color = "yellow", marker = "*")
            elif team == away_team.id.iloc[0]:
                ax.scatter(x, y, color='#41B6E6')
            elif team == home_team.id.iloc[0]:
                ax.scatter(x, y, color='#5F249F')

        # plot ball
        ball_x = current_time_stamp_data['ball_data']['x']
        ball_y = current_time_stamp_data['ball_data']['y']
        ax.scatter(ball_x, ball_y, s=10, color='black')

        # fig -> PIL image
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgb_bytes = fig.canvas.tostring_rgb()
        pil_img = Image.frombytes("RGB", (w, h), rgb_bytes)
        pil_img = pil_img.convert("P", palette=Image.ADAPTIVE, colors=256)

        pil_frames.append(pil_img)
        plt.close(fig)

    # Save GIF to memory and display
    run_gif = BytesIO()
    if pil_frames:
        pil_frames[0].save(
            run_gif,
            format="GIF",
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0,
            disposal=2
        )
        run_gif.seek(0)
    display(IPimg(data=run_gif.read()))

create_run_gif(6,off_ball_runs_df,match_tracking_data,players_df,home_team,away_team,pitch, duration=125)

def calculate_v(how_many_frames_back, frame_of_interest, match_tracking_data):
    """
    Calculate player velocities between a previous frame and the current frame pulled from match tracking data.

    Parameters:
    how_many_frames_back : int
        Number of frames to look back for velocity calculation
    frame_of_interest : int
        Current frame number (same as current frame's index)
    match_tracking_data : df
        tracking data containing player positions for a specific match

    Returns:
    velocities : df
        df containing player_id, x_start, y_start, x_end, y_end, vx, and vy.
    """
    # --- find the frames ---
    prior_frame_index = int(frame_of_interest) - int(how_many_frames_back)
    prior_frame = pd.DataFrame(match_tracking_data[prior_frame_index]["player_data"])
    current_frame = pd.DataFrame(match_tracking_data[frame_of_interest]["player_data"])

    cols = ["player_id", "x", "y"] #cols were using to calculate v
    prior_positions = prior_frame[cols] #gets prior x and y
    current_positions = current_frame[cols] #current x and y

    # merge start and end positions by player_id
    merged = current_positions.merge(
        prior_positions,
        on="player_id",
        suffixes=("_end", "_start"),
        how="inner"  # only keep players present in both frames
    )

    # compute displacements and velocity
    merged["vx"] = merged["x_end"] - merged["x_start"]
    merged["vy"] = merged["y_end"] - merged["y_start"]

    # convert to list of dicts
    velocities = pd.DataFrame(merged.to_dict(orient="records"))
    return velocities

def calculate_v(how_many_frames_back, frame_of_interest, match_tracking_data):
    """
    Calculate player velocities between a previous frame and the current frame pulled from match tracking data.

    Parameters:
    how_many_frames_back : int
        Number of frames to look back for velocity calculation
    frame_of_interest : int
        Current frame number (same as current frame's index)
    match_tracking_data : df
        tracking data containing player positions for a specific match

    Returns:
    velocities : df
        df containing player_id, x_start, y_start, x_end, y_end, vx, and vy.
    """
    # --- find the frames ---
    prior_frame_index = int(frame_of_interest) - int(how_many_frames_back)
    prior_frame = pd.DataFrame(match_tracking_data[prior_frame_index]["player_data"])
    current_frame = pd.DataFrame(match_tracking_data[frame_of_interest]["player_data"])

    cols = ["player_id", "x", "y"] #cols were using to calculate v
    prior_positions = prior_frame[cols] #gets prior x and y
    current_positions = current_frame[cols] #current x and y

    # merge start and end positions by player_id
    merged = current_positions.merge(
        prior_positions,
        on="player_id",
        suffixes=("_end", "_start"),
        how="inner"  # only keep players present in both frames
    )

    # compute displacements and velocity
    merged["vx"] = merged["x_end"] - merged["x_start"]
    merged["vy"] = merged["y_end"] - merged["y_start"]

    # convert to list of dicts
    velocities = pd.DataFrame(merged.to_dict(orient="records"))
    return velocities

#evolve players (uses the velocity function) and return residuals (error from projection) and calculated velocity
def evolve_player_residuals(t, frame_of_interest, frames_before_for_v, match_tracking_data):
    """
    Evolves players forward t seconds from frame_of_interest

    Parameters:
    t : int
        the number of seconds you want to evolve players forward assuming constant velocity and direction
    frame_of_interest : int
        the frame of the match you want to pull data from and evolve players forward
    frames_before_for_v : int
        the number of frames you want to go back to calcualte the velocity
    match_tracking_data : df
        tracking data containing player positions for a specific match
    """
    #get player velocities
    pid_x_y_v_df = calculate_v(frames_before_for_v, frame_of_interest, match_tracking_data)
    
    #define columns of interest for evolution
    cols = ['player_id', 'x_end', 'y_end', 'vx', 'vy']
    
    #filter to only have relevant data for evolution
    current_pid_x_y_v = pid_x_y_v_df[cols] 
    
    #evolve forward and store it as x/y_projected
    current_pid_x_y_v["x_projected"] = current_pid_x_y_v["x_end"] + current_pid_x_y_v["vx"] * t
    current_pid_x_y_v["y_projected"] = current_pid_x_y_v["y_end"] + current_pid_x_y_v["vy"] * t
    
    #define relevant columns for evolution
    pid_and_projections = ['player_id', 'x_projected', 'y_projected', 'vx', 'vy'] #added vx and vy
    
    #filter to just projections for residual calculation
    projected_data = current_pid_x_y_v[pid_and_projections]
    
    #define relevant comparison columns to pull from match data
    comparison_cols = ['player_id', 'x', 'y']
    
    #get match tracking data at time of projection
    future_frame_index = t * 10 + frame_of_interest #gets future frame's index based on t and frame of interest
    future_match_tracking_data = pd.DataFrame(match_tracking_data[future_frame_index]["player_data"])

    #grab actual match data for residual calculation
    actual_data = future_match_tracking_data[comparison_cols]

    #merge on player id
    merged_projected_actual = projected_data.merge(actual_data, on="player_id", how="inner")
    merged_projected_actual["residual_x"] = merged_projected_actual["x"] - merged_projected_actual["x_projected"] #error for x
    merged_projected_actual["residual_y"] = merged_projected_actual["y"] - merged_projected_actual["y_projected"] #error for y
    merged_projected_actual["residual_dist"] = (merged_projected_actual["residual_x"]**2 + merged_projected_actual["residual_y"]**2) ** 0.5 #error as a distance from actual location

    return merged_projected_actual[[
        "player_id",
        "x", "y",
        "x_projected", "y_projected",
        "residual_x", "residual_y", "residual_dist", "vx", "vy"
    ]]

#example of a timestamp (frame 180) evoution (1 second) based on velocity calculated from player movement in the last second of match tracking data (10 frames)
evolution1 = evolve_player_residuals(2, 180, 10, match_tracking_data)
print(evolution1)
