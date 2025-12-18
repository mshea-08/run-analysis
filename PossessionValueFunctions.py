#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 09:43:14 2025

@author: meredithshea
"""

import numpy as np
from scipy.stats import logistic

# ===============================================
# Pitch Control Functions
# ===============================================

def team_pitch_control_matrix(
    ball_x,             # ball position x
    ball_y,             # ball position y
    player_positions,   # player_positions DataFrame computed by player_positions function
    team_id,            # id of team in possession
    pitch_length,       # length of pitch (stored in match_data["pitch_length"])
    pitch_width,        # width of pitch (stored in match_data["pitch_width"])
    rows=12,            # vertical discretization of pitch
    cols=16             # horizontal discretization of pitch
    ):
    """
    Returns a matrix of the total team pitch control at each grid cell center. 
    Rows indexed bottom -> top; cols indexed left -> right.
    
    Note: this function is not used in our metrix evaluation, but it is nice 
    to have. 
    
    """
    # Parameters
    ball_speed = 15.0
    player_speed = 5.0
    s = 0.37            # logistic parameter
    l = 4.3             # control rate
    dt = 0.04           # time step
    t_lead = 0.7        # player reaction time
    max_steps = 2500
    eps = 1e-2

    # Set up pitch grid points for computation
    dy = pitch_width/rows
    dx = pitch_length/cols
    x_centers = -pitch_length / 2 + dx * (np.arange(cols) + 0.5)  
    y_centers = -pitch_width / 2 + dy * (np.arange(rows) + 0.5)   
    Xg, Yg = np.meshgrid(x_centers, y_centers)   # (rows, cols)                  
    G = rows * cols                              # number of matrix entries
    X = Xg.ravel()  # (G, )
    Y = Yg.ravel()  # (G, )

    # Turn player info into (P,) arrays (P = number of players)
    df = player_positions  
    px  = df["x"].to_numpy(dtype=float) 
    py  = df["y"].to_numpy(dtype=float)  
    vx  = df["v_x"].to_numpy(dtype=float)   
    vy  = df["v_y"].to_numpy(dtype=float)  
    pid = df["team_id"].to_numpy()
    team_mask = (pid == team_id)            
    P = px.shape[0]

    # Intermediate positions
    x1 = px + t_lead * vx                 
    y1 = py + t_lead * vy            

    # Time for the ball to reach each grid location
    ball_time = np.hypot(X - ball_x, Y - ball_y) / ball_speed

    # How long it takes each player to reach each grid position
    dx_gp = X[:, None] - x1[None, :]        # (G,P)
    dy_gp = Y[:, None] - y1[None, :]        # (G,P)
    time_to_ball_gp = t_lead + np.hypot(dx_gp, dy_gp) / player_speed  # (G,P)

    # Set up for computing probabilities
    p_tot_gp = np.zeros((G, P), dtype=float)   # probs for each grid point
    S = np.ones(G, dtype=float)                # checking how close we are to prob sum = 1

    # Loop to compute all probabilities at once
    for step in range(1, max_steps + 1):
        t_now = ball_time + step * dt          # (G,)
        F_gp = logistic.cdf(t_now[:, None], loc=time_to_ball_gp, scale=s)  # (G,P)
        total_F = F_gp.sum(axis=1)             # (G,)
        incr_gp = (S[:, None] * l * F_gp * dt) # (G,P)
        p_tot_gp += incr_gp
        decay = np.exp(-l * total_F * dt)
        S *= decay
        S *= (1.0 - l * total_F * dt)         # update S
        # Global early stop when all grid points done
        if np.all(1.0 - p_tot_gp.sum(axis=1) <= eps):
            break

    # Sum players on requested team
    team_prob_flat = p_tot_gp[:, team_mask].sum(axis=1)  # (G,)
    probs = team_prob_flat.reshape(rows, cols)
    return probs

def team_pitch_control_no_runner_matrix(
    ball_x, 
    ball_y, 
    player_positions,
    team_id, 
    runner_id,
    pitch_length, 
    pitch_width, 
    rows=12, 
    cols=16
    ):
    """
    Same as above but remove the runner from the computation. This is the 
    pitch control model we use in the space manipulation metrix computation. 
    
    """
    # Parameters
    ball_speed = 15.0
    player_speed = 5.0
    s = 0.37            # logistic parameter
    l = 4.3             # control rate
    dt = 0.04           # time step
    t_lead = 0.7        # player reaction time
    max_steps = 2500
    eps = 1e-2

    # Set up pitch grid points for computation
    dy = pitch_width/rows
    dx = pitch_length/cols
    x_centers = -pitch_length / 2 + dx * (np.arange(cols) + 0.5)  
    y_centers = -pitch_width / 2 + dy * (np.arange(rows) + 0.5)   
    Xg, Yg = np.meshgrid(x_centers, y_centers)   # (rows, cols)                  
    G = rows * cols                              # number of matrix entries
    X = Xg.ravel()  # (G, )
    Y = Yg.ravel()  # (G, )

    ################################################
    # Remove runner from player positions   
    df = player_positions.loc[
            player_positions.player_id != runner_id]
    ################################################
    
    # Turn player info into (P,) arrays (P = number of players) 
    px  = df["x"].to_numpy(dtype=float) 
    py  = df["y"].to_numpy(dtype=float)  
    vx  = df["v_x"].to_numpy(dtype=float)   
    vy  = df["v_y"].to_numpy(dtype=float)  
    pid = df["team_id"].to_numpy()
    team_mask = (pid == team_id)            
    P = px.shape[0]

    # Intermediate positions
    x1 = px + t_lead * vx                 
    y1 = py + t_lead * vy            

    # Time for the ball to reach each grid location
    ball_time = np.hypot(X - ball_x, Y - ball_y) / ball_speed

    # How long it takes each player to reach each grid position
    dx_gp = X[:, None] - x1[None, :]        # (G,P)
    dy_gp = Y[:, None] - y1[None, :]        # (G,P)
    time_to_ball_gp = t_lead + np.hypot(dx_gp, dy_gp) / player_speed  # (G,P)

    # Set up for computing probabilities
    p_tot_gp = np.zeros((G, P), dtype=float)   # probs for each grid point
    S = np.ones(G, dtype=float)                # checking how close we are to prob sum = 1

    # Loop for computing all probabilities    
    for step in range(1, max_steps + 1):
        t_now = ball_time + step * dt          # (G,)
        F_gp = logistic.cdf(t_now[:, None], loc=time_to_ball_gp, scale=s)  # (G,P)
        total_F = F_gp.sum(axis=1)             # (G,)
        incr_gp = (S[:, None] * l * F_gp * dt) # (G,P)
        p_tot_gp += incr_gp
        decay = np.exp(-l * total_F * dt)
        S *= decay
        S *= (1.0 - l * total_F * dt)         # update S
        # global early stop when all grid points done
        if np.all(1.0 - p_tot_gp.sum(axis=1) <= eps):
            break

    # Sum players on requested team
    team_prob_flat = p_tot_gp[:, team_mask].sum(axis=1)  # (G,)
    probs = team_prob_flat.reshape(rows, cols)
    return probs

# ================================================
# Runner Removed Possession Value Function
# ================================================


def p_val_no_runner(
        ball_x,             # ball position x
        ball_y,             # ball position y
        player_positions,   # player_positions DataFrame computed by player_positions function
        runner_id,          # player id of runner
        team_id,            # team id of team in possession
        attacking_side,     # direction of attack (string: "right_to_left" or "left_to_right")
        pitch_length,       # length of pitch (stored in match_data["pitch_length"])
        pitch_width,        # width of pitch (stored in match_data["pitch_width"])
        xT_grid,            # xT grid is stored as matrix with the same 12x16 dimensions as pitch control
        ):
    """
    Computes runner removed possession value. For each grid center of our 
    discretized pitch we compute,
    
    xT(x,y)*pitch_control_no_runner(x,y)
    
    and then we sum over all values. 

    """
    
    # Compute runner removed pitch_control matrix
    pitch_control = team_pitch_control_no_runner_matrix(
        ball_x, 
        ball_y, 
        player_positions, 
        team_id,
        runner_id, 
        pitch_length, 
        pitch_width
        )
    # xT assumes attack is left_to_right, if this is not the case we flip pitch_control
    if attacking_side == "right_to_left":
        pitch_control = np.flip(pitch_control)
    # Compute "possession value"
    return np.array(xT_grid*pitch_control).sum()
    
