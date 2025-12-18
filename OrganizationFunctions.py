#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 09:32:26 2025

@author: meredithshea
"""

import pandas as pd
import math
from collections import defaultdict

# ===============================================
# Organization and other miscellaneous functions.
# ===============================================

def timestamp_to_seconds(t_str):
    """
    Converts strings of the form 'hh:mm:ss.ff' into total seconds as a float.
    
    """
    if t_str is None or t_str == "":
        return None  # handles missing values
    h, m, s = t_str.split(':')
    return int(h)*3600 + int(m)*60 + float(s)

def add_velocities(match_tracking_data):
    """
    Adds velocities and speed to match_tracking_data (in m/s).
    Computes the central difference if possible, else uses one sided difference.
    
    """
    # Re-organize by player information
    by_player = defaultdict(list)  # player_id -> list of (t, x, y, frame_idx,  person_idx)
    for frame_idx, frame in enumerate(match_tracking_data):
        t = timestamp_to_seconds(frame["timestamp"])
        for person_idx, p in enumerate(frame.get("player_data", [])):
            pid = p["player_id"]
            x, y = p["x"], p["y"]
            by_player[pid].append((t, x, y, frame_idx, person_idx))

    # Add velocities and speed to by_player
    for pid, obs in by_player.items():
        # Sort by time
        obs.sort(key=lambda r: r[0])

        # Set up
        times = [r[0] for r in obs]
        xs    = [r[1] for r in obs]
        ys    = [r[2] for r in obs]
        vxs = [None]*len(obs)
        vys = [None]*len(obs)

        for i in range(len(obs)):
            if len(obs) == 1:
                # Only one observation for this player, can't compute velocity
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

        # Put the results back into match_tracking_data
        for (vx, vy), (_, _, _, frame_idx, person_idx) in zip(zip(vxs, vys), obs):
            speed = (None if (vx is None or vy is None) else math.hypot(vx, vy))
            player_dict = match_tracking_data[frame_idx]["player_data"][person_idx]
            player_dict["v_x"] = vx
            player_dict["v_y"] = vy
            player_dict["speed"] = speed
    
    return match_tracking_data

def create_player_positions(match_data,match_tracking_data,frame):
    """
    Creates the Data Frame player_positions for a given time stamp. 
    
    Data Frame includes:
        - player id (as id), 
        - team_id, 
        - position (x,y), 
        - velocity (v_x,v_y), 
        - field position (as position), 
        - and player name (as short_name).
    """
    # Check that the frame has tracking
    if not match_tracking_data[frame]["player_data"]:
        return None
    # Get player tracking data from tracking
    player_tracking = pd.DataFrame(
        match_tracking_data[frame]["player_data"]
        )
    # Get player information from match_data
    players_df = pd.DataFrame(match_data["players"])
    
    position_df = pd.DataFrame(players_df.player_role.to_list())
    position_df = position_df.rename(columns={'id': 'position_id'})
    
    players_df = pd.concat([players_df,position_df],axis=1)
    players_df = players_df[["number", "team_id", "id","short_name","acronym"]]
    players_df.rename(
        columns={"id": "player_id", "acronym": "position"}, 
        inplace=True
        )
    # Merge on player_id
    player_positions = pd.merge(
        players_df, 
        player_tracking, 
        on="player_id"
        )
    return player_positions

def create_player_positions_2(match_data,frame_data):
    """
    Alternative set up for computing player positions, instead of taking all
    the match_data it just takes the frame data. This is because we often 
    stored the single frame_data in the run_data for faster computing. 
    
    """

    # Get player tracking data from tracking
    player_tracking = pd.DataFrame(
        frame_data["player_data"]
        )
    
    if list(player_tracking.columns) == [0]:
        first_val = player_tracking.iloc[0, 0]
        if isinstance(first_val, (dict, list, pd.Series)):
            player_tracking = player_tracking[0].apply(pd.Series)
            
    if player_tracking.size == 0:  
        return None
    
    # Get player information from match_data
    players_df = pd.DataFrame(match_data["players"])
    if list(players_df.columns) == [0]:
        first_val = players_df.iloc[0, 0]
        if isinstance(first_val, (dict, list, pd.Series)):
            players_df = players_df[0].apply(pd.Series)
    position_df = pd.DataFrame(players_df.player_role.to_list())
    position_df = position_df.rename(columns={'id': 'position_id'})
    players_df = pd.concat([players_df,position_df],axis=1)
    players_df = players_df[["number", "team_id", "id","short_name","acronym"]]
    players_df.rename(
        columns={"id": "player_id", "acronym": "position"}, 
        inplace=True
        )
    # Merge on player_id
    player_positions = pd.merge(
        players_df, player_tracking, 
        on="player_id"
        )
    return player_positions

def ball_position(match_tracking_data,frame):
    """
    Takes match_tracking_data and frame and returns the location of the 
    soccer ball. 
    
    """
    if not match_tracking_data[frame]["ball_data"]:
        return None
    ball_x = match_tracking_data[frame]["ball_data"]["x"]
    ball_y = match_tracking_data[frame]["ball_data"]["y"]
    return ball_x, ball_y

def ball_position_2(frame_data):
    """
    Same as ball_position but takes only frame_data.
    
    """
    if not frame_data["ball_data"]:
        return None
    ball_x = frame_data["ball_data"]["x"]
    ball_y = frame_data["ball_data"]["y"]
    return ball_x, ball_y