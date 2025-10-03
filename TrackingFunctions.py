#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 14:31:42 2025

@author: meredithshea
"""

import pandas as pd
import numpy as np
from scipy.stats import logistic
import math
from collections import defaultdict

def timestamp_to_seconds(t_str):
    """
    Converts strings of the form 'hh:mm:ss.ff' into total 
    seconds as a float.
    
    """
    if t_str is None or t_str == "":
        return None  # handles missing values
    h, m, s = t_str.split(':')
    return int(h)*3600 + int(m)*60 + float(s)

def add_velocities(match_tracking_data):
    """
    Adds velocities and speed to match_tracking_data (in m/s)

    """
    # re-organize by player information
    by_player = defaultdict(list)  # player_id -> list of (t, x, y, frame_idx,  person_idx)
    for frame_idx, frame in enumerate(match_tracking_data):
        t = timestamp_to_seconds(frame["timestamp"])
        for person_idx, p in enumerate(frame.get("player_data", [])):
            pid = p["player_id"]
            x, y = p["x"], p["y"]
            by_player[pid].append((t, x, y, frame_idx, person_idx))

    # add velocities and speed toby_player
    for pid, obs in by_player.items():
        # sort by time to be safe
        obs.sort(key=lambda r: r[0])

        # setting up
        times = [r[0] for r in obs]
        xs    = [r[1] for r in obs]
        ys    = [r[2] for r in obs]
        vxs = [None]*len(obs)
        vys = [None]*len(obs)

        for i in range(len(obs)):
            if len(obs) == 1:
                # Only one observation for this player: can't compute velocity
                vxs[i] = None
                vys[i] = None
                continue

            if 0 < i < len(obs)-1:
                # Central difference to compute velocity
                dt = times[i+1] - times[i-1]
                if dt == 0:
                    vx = vy = None
                else:
                    vx = (xs[i+1] - xs[i-1]) / dt
                    vy = (ys[i+1] - ys[i-1]) / dt
            elif i == 0:
                # Forward difference
                dt = times[i+1] - times[i]
                if dt == 0:
                    vx = vy = None
                else:
                    vx = (xs[i+1] - xs[i]) / dt
                    vy = (ys[i+1] - ys[i]) / dt
            else:  # i == len(obs)-1
                # Backward difference
                dt = times[i] - times[i-1]
                if dt == 0:
                    vx = vy = None
                else:
                    vx = (xs[i] - xs[i-1]) / dt
                    vy = (ys[i] - ys[i-1]) / dt

            vxs[i] = vx
            vys[i] = vy

        # put the results back into match_tracking_data
        for (vx, vy), (_, _, _, frame_idx, person_idx) in zip(zip(vxs, vys), obs):
            speed = (None if (vx is None or vy is None) else math.hypot(vx, vy))
            player_dict = match_tracking_data[frame_idx]["player_data"][person_idx]
            player_dict["v_x"] = vx
            player_dict["v_y"] = vy
            player_dict["speed"] = speed
    
    return match_tracking_data

def create_player_positions(match_data,match_tracking_data,frame):
    # check that the frame has tracking
    if not match_tracking_data[frame]["player_data"]:
        return None
    # add velocities to the data
    tracking = add_velocities(match_tracking_data)
    # get player tracking data from tracking
    player_tracking = pd.DataFrame(
        tracking[frame]["player_data"]
        )
    # get player information from match_data
    players_df = pd.DataFrame(match_data["players"])
    position_df = pd.DataFrame(players_df.player_role.to_list())
    position_df = position_df.rename(columns={'id': 'position_id'})
    players_df = pd.concat([players_df,position_df],axis=1)
    players_df = players_df[["number", "team_id", "id","short_name","acronym"]]
    players_df.rename(
        columns={"id": "player_id", "acronym": "position"}, 
        inplace=True
        )
    # merge on player_id
    player_positions = pd.merge(
        players_df, player_tracking, 
        on="player_id"
        )
    return player_positions

def ball_position(match_tracking_data,frame):
    if not match_tracking_data[frame]["ball_data"]:
        return None
    ball_x = match_tracking_data[frame]["ball_data"]["x"]
    ball_y = match_tracking_data[frame]["ball_data"]["y"]
    return ball_x, ball_y

def player_pitch_control_at_coord(x,y,ball_x,ball_y,player_positions):
    # speed assumptions
    ball_speed = 15   # ball speed
    player_speed = 5  # max player speed
    
    # time for the ball to get to the location
    dist = np.sqrt((x-ball_x)**2+(y-ball_y)**2)
    ball_time = dist/ball_speed
    
    # add intermediate player positions (x1,y1)
    # idea: players will continue along their trajectory for 0.7s
    player_positions['x1'] = (
        player_positions.x + 0.7*player_positions.v_x
        )
    player_positions['y1'] = (
        player_positions.y + 0.7*player_positions.v_y
        )
    
    # compute the time for the player to get to the ball
    # time = 0.7 s + dist from intermediate position to (x,y)
    player_positions['time_to_ball'] = (
        np.sqrt((x - player_positions.x1)**2 + (y - player_positions.y1)**2)/player_speed
        + 0.7
        )
    
    # probability assumptions
    s = 0.37     # parameter for logistic arrival time
    l = 4.3      # parameter for exponential control time
    dt = 0.04    # time step for numerical integral 
    # initialization 
    player_positions['p_tot'] = 0  # individual player probabilities
    p_sum = 0                      # sum of all player probabilities (should approach 1)
    i = 0                          # counts iterations
    # loop to compute probabilities
    while (1-p_sum) > 0.01 and i < 2500: # compute until probabilities add to nearly 1 or for a long time
        i += 1
        p_sum = player_positions['p_tot'].sum()
        # compute prob of having arrived at the location
        player_positions['arrival_prob'] = (
            logistic.cdf(ball_time + (i+1)*dt,loc=player_positions.time_to_ball,scale=s)
            )
        # compute probability of controlling the ball
        player_positions[f'p_{i}'] = (
            (1-p_sum)*player_positions['arrival_prob']*l*dt
            )
        # clean up columns
        player_positions['p_tot'] += player_positions[f'p_{i}']
        player_positions.drop(columns=['arrival_prob', f'p_{i}'], 
                              inplace=True)
    return player_positions
    
    
def team_pitch_control_at_coord(x,y,ball_x,ball_y,player_positions,team_id):
    # compute player pitch control
    df = player_pitch_control_at_coord(x, y, ball_x, ball_y, player_positions)
    
    # sum over team_id
    team_control = df[df.team_id == team_id].sum()
    
    return team_control

def team_pitch_control_matrix(
        ball_x,ball_y,player_positions,
        team_id,pitch_length,pitch_width,dx=1,dy=1):
    # initialize matrix of probabilities
    rows = int(pitch_width/dy)
    cols = int(pitch_length/dx)
    probs = np.zeros((rows,cols))
    # loop to compute probabilities (any way to speed up?)
    for i in range(rows):
        for j in range(cols):
            x_c = -pitch_length/2 + dx*(j + 0.5)
            y_c = -pitch_width/2 + dy*(i + 0.5)
            
            team_prob = team_pitch_control_at_coord(x_c,y_c,ball_x,ball_y,player_positions,team_id)
            
            probs[i,j] = team_prob
    return probs
    
    
    
    
    
    
    
    
    
    
    