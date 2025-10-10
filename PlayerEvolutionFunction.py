
import pandas as pd
from skillcorner.client import SkillcornerClient
from io import BytesIO
import matplotlib.pyplot as plt
from mplsoccer import Pitch



def velocity_calc(frames_back, frame_now, match_tracking_data, frame_rate = 10):
    """
    frames_back: number of frames to look back for velocity calculation
    frame_now: current frame number
    match_tracking_data: full tracking data for the match
    frame_rate: frames per second (default 10)
    """
    frame_prev = frame_now - frames_back

    #converting players to data frame
    df_prev = pd.DataFrame(match_tracking_data[frame_prev]['player_data'])
    df_now = pd.DataFrame(match_tracking_data[frame_now]['player_data'])

    #merge two dataframes on player_id
    merged = df_now.merge(df_prev, on='player_id', suffixes=("_now", "_prev"))

    #time difference
    dt = frames_back / frame_rate

    #compute velocity and speed
    merged['v_x'] = (merged['x_now'] - merged['x_prev']) / dt
    merged['v_y'] = (merged['y_now'] - merged['y_prev']) / dt
    merged['speed'] = np.sqrt(merged['v_x']**2 + merged['v_y']**2)

    return merged[['player_id', 'x_now', 'y_now', 'v_x', 'v_y', 'speed']]



def evolve_players(frame_now, t, frames_back, match_tracking_data, frame_rate = 10):

    df = velocity_calc(frames_back, frame_now, match_tracking_data, frame_rate)

    #predicting future positions
    df['x_pred'] = df['x_now'] + df['v_x'] * t
    df['y_pred'] = df['y_now'] + df['v_y'] * t

    #future positions
    frame_future = frame_now + (t * frame_rate)
    df_future = pd.DataFrame(match_tracking_data[frame_future]['player_data'])

    #merge predicted and actual positions
    merged = df.merge(df_future[['player_id', 'x', 'y']], on='player_id', suffixes=('', '_actual'))
    merged = merged.rename(columns={'x': 'x_actual', 'y': 'y_actual'})

    #computing residuals / errors
    merged['error_x'] = merged['x_actual'] - merged['x_pred']
    merged['error_y'] = merged['y_actual'] - merged['y_pred']
    merged['error_dist'] = np.sqrt(merged['error_x']**2 + merged['error_y']**2)

    return merged[[
        "player_id", "x_now", "y_now", "x_pred", "y_pred", "x_actual", "y_actual", "error_dist"
    ]]


def plot_predictions(pred_df, match_data, home_team, away_team, match_tracking_data, frame_now, t, frame_rate=10):
    """
    pred_df: data frame output from evolve_players();
    match_data: match data containing pitch length/width;
    home_team, away_team: team data frames for color coding;
    match_tracking_data: full tracking data for the match;
    frame_now: current frame number;
    t: time in seconds to predict forward;
    frame_rate: frames per second (default 10);
    """

    #set up pitch
    pitch = Pitch(
        pitch_type='skillcorner',
        pitch_length=match_data['pitch_length'],
        pitch_width=match_data['pitch_width'],
        pitch_color='#aabb97',
        line_color='white',
        stripe_color='#c2d59d',
        stripe=True
    )
    fig, ax = pitch.draw(figsize=(8, 5))

    #merge team info to pred_df
    players_df = pd.DataFrame(match_data["players"])[["id", "team_id"]]
    pred_df = pred_df.merge(players_df, left_on="player_id", right_on="id", how="left")


    #plot players
    for _, p in pred_df.iterrows():
        team_id = p.get("team_id", None)
        color = "#5F249F" if team_id == home_team.id.iloc[0] else "#41B6E6"

        #current position (circle)
        ax.scatter(p["x_now"], p["y_now"], color=color, s=25, alpha=0.6, label="_nolegend_")

        #predicted position (x)
        ax.scatter(p["x_pred"], p["y_pred"], color=color, marker='x', s=40, label="_nolegend_")

        #actual future position (triangle)
        ax.scatter(p["x_actual"], p["y_actual"], color=color, marker='^', s=40, edgecolor='black', label="_nolegend_")

        #arrow from predicted to actual
        ax.arrow(
            p["x_pred"], p["y_pred"],
            p["x_actual"] - p["x_pred"],
            p["y_actual"] - p["y_pred"],
            color="red", alpha=0.5, width=0.2, length_includes_head=True
        )

    #plot ball position
    ball_prev = match_tracking_data[frame_now]['ball_data']
    ax.scatter(ball_prev['x'], ball_prev['y'], color='black', s=30, marker='o', label="_nolegend_")

    frame_future = frame_now + (t * frame_rate)
    ball_actual = match_tracking_data[frame_future]['ball_data']
    ax.scatter(ball_actual['x'], ball_actual['y'], color='black', s=30, marker='^', edgecolor='white', label="_nolegend_")

    #arrow from current to actual ball position
    ax.arrow(
        ball_prev['x'], ball_prev['y'],
        ball_actual['x'] - ball_prev['x'],
        ball_actual['y'] - ball_prev['y'],
        color="blue", alpha=0.5, width=0.2, length_includes_head=True
    )   

    #legend
    ax.set_title("Predicted vs Actual Player Positions", fontsize=14, fontweight='bold')
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Current Position', markerfacecolor='gray', markersize=8),
        plt.Line2D([0], [0], marker='x', color='w', label='Predicted Position', markerfacecolor='black', markersize=8),
        plt.Line2D([0], [0], marker='^', color='w', label='Actual Position', markerfacecolor='black', markersize=8),
        plt.Line2D([0], [0], color='red', lw=2, label='Prediction Error')
    ]
    ax.legend(handles=handles, loc='upper right')

    plt.show()
