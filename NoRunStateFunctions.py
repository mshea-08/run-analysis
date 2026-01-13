#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 15:35:59 2025

@author: meredithshea
"""

import pandas as pd
import numpy as np
import math
from OrganizationFunctions import create_player_positions_2

# ================================================
# Helper Functions
# ================================================

def position_on_line(
        x1,y1,      # starting point
        x2,y2,      # ending point
        distance=2, # distance new point is from start
        ):
    """
    Returns new point on the line (x1,y1)--(x2,y2) a 
    specified distance away from the first point.

    """
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L == 0:
        raise ValueError("Two points are identical.")
    ux, uy = dx / L, dy / L
    return x1 + distance * ux, y1 + distance * uy

def softmin(
        x,      # array of values
        T=6.0,  # soft-min parameter, small T -> min-like, large T -> avg-like
        ):

    x = np.asarray(x, float)
    a = -x / T
    a -= a.max()
    w = np.exp(a)
    w /= w.sum()
    return float((w * x).sum())

def grid_points_in_disk(
        cx,cy,                  # circle center
        radius,                 # circle radius
        spacing=0.5,            # grid spacing
        include_center=True,    # whether points include the center
        ):
    """
    Output is a Nx2 array, where each row is an (x,y) 
    coordinate pair within the indicated circle. 
    
    """
    # Ensure floats
    cx = float(cx)
    cy = float(cy)
    r = float(radius)
    s = float(spacing)

    # Integer offsets k such that k*spacing lies within [-r, r]
    kmin = int(np.ceil(-r / s))
    kmax = int(np.floor(r / s))
    if kmax < kmin:
        # No lattice offsets fit
        return np.array([[cx, cy]]) if include_center else np.empty((0, 2))

    x_offsets = np.arange(kmin, kmax + 1, dtype=int) * s
    y_offsets = np.arange(kmin, kmax + 1, dtype=int) * s
    Xo, Yo = np.meshgrid(x_offsets, y_offsets, indexing="xy")

    X = cx + Xo.ravel()
    Y = cy + Yo.ravel()

    # Keep points within the (closed) disk, small epsilon for floating tolerance
    eps = 1e-12
    inside = (X - cx) ** 2 + (Y - cy) ** 2 <= (r + eps) ** 2
    X_in = X[inside]
    Y_in = Y[inside]

    if X_in.size == 0:
        return np.array([[cx, cy]]) if include_center else np.empty((0, 2))

    # Sort by x then y 
    order = np.lexsort((Y_in, X_in))
    pts = np.column_stack((X_in[order], Y_in[order]))
    
    return pts


def avg_near_seg(
        df,             # df containing positions
        x_col,          # col in df with x positions
        y_col,          # col in df with y positions
        x0,             # x coord of vertical line
        y1,             # starting y val
        y2,             # ending y val
        max_dist=20.0,  # dist away from line segment we check for points
        ):
    """
    Return a weighted average of y for points within max_dist of the vertical
    segment x = x0, y in [y1, y2].

    - If no points are within max_dist, return the midpoint (y1 + y2)/2.
    - If there are nearby points, weight each point by its distance to the
      center of the segment.
    
    """
    # Make sure y values are ordered
    ymin, ymax = sorted([y1, y2])
    y_center = 0.5 * (ymin + ymax)

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    # Distance to the segment
    dx = x - x0
    within = (y >= ymin) & (y <= ymax)
    below = y < ymin
    above = y > ymax

    dy_below = y - ymin
    dy_above = y - ymax

    dist = np.empty_like(x, dtype=float)
    dist[within] = np.abs(dx[within])
    dist[below]  = np.sqrt(dx[below]**2 + dy_below[below]**2)
    dist[above]  = np.sqrt(dx[above]**2 + dy_above[above]**2)

    # Points close enough to the segment
    near_mask = dist <= max_dist

    if not near_mask.any():
        # No nearby points: return average of endpoints
        return y_center

    x_near = x[near_mask]
    y_near = y[near_mask]
    
    x_near = np.append(x_near,x0-1)
    y_near = np.append(y_near,(y1+y2)/2)

    # Distance to the center of the segment
    d_center = np.sqrt((x_near - x0)**2 + (y_near - y_center)**2)

    # Weights
    weights = np.maximum(max_dist - d_center, 0.0)

    # If all weights are zero, fall back to simple mean
    if np.all(weights == 0):
        return y_near.mean()

    return np.average(y_near, weights=weights)


def is_ball_in_attacking_final_third(
        ball_x,         # downfield position of ball
        pitch_length,   # length of pitch (skillcorner convention)
        attacking_side, # string, 'right_to_left' or 'left_to_right'
        ):
    """
    Returns True if the ball is in the attacking team's final third
    (closest third to the opponent's goal) along the x-axis.
     
    """
    half = pitch_length / 2.0

    if attacking_side == "left_to_right":
        # Attacking goal at +half
        # Final third: x >= half - pitch_length/3
        return ball_x >= (half - pitch_length / 3.0)
    else:  # "right_to_left"
        # Attacking goal at -half
        # Final third: x <= -half + pitch_length/3
        return ball_x <= (-half + pitch_length / 3.0)


def closest_defender_to_point(
        defender_positions, # df of all defender positions
        x, 
        y,               
        ):
    """
    Return the player_id of the defender closest to (x, y),
    or None if defender_positions is empty.
    """
    if defender_positions is None or defender_positions.empty:
        return None

    dists = np.hypot(
        defender_positions["x"].to_numpy() - x,
        defender_positions["y"].to_numpy() - y
    )
    idx = dists.argmin()
    return defender_positions["player_id"].iloc[idx]


def filter_points_near_vertical_segment(
        points_xy, # array, col 0 = x, col 1 = y
        x0, 
        y1, 
        y2, 
        max_dist,
        ):
    """
    Keep only points within max_dist of the vertical segment
    x = x0, y in [y1, y2].

    Returns array.
    """
    if points_xy.size == 0:
        return points_xy

    x = points_xy[:, 0]
    y = points_xy[:, 1]

    ymin, ymax = (y1, y2) if y1 <= y2 else (y2, y1)

    dx = x - x0

    within = (y >= ymin) & (y <= ymax)
    below  = (y < ymin)
    above  = (y > ymax)

    dist = np.empty_like(dx, dtype=float)
    
    # straight horizontal distance if projection lies on segment
    dist[within] = np.abs(dx[within])
    
    # distance to lower endpoint
    dy_below = y[below] - ymin
    dist[below] = np.sqrt(dx[below]**2 + dy_below**2)
    
    # distance to upper endpoint
    dy_above = y[above] - ymax
    dist[above] = np.sqrt(dx[above]**2 + dy_above**2)

    mask = dist <= max_dist
    return points_xy[mask]

# ================================
# Main Functions
# ================================

def move_runner(
        runner_id,                  # player_id for runner
        attacker_positions_end,     # DataFrame of attacker end positions (format of create_player_positions)
        defender_positions_end,     # DataFrame of defender end positions
        attacker_positions_start,   # DataFrame of attacker start positions
        defender_positions_start,   # DataFrame of defender start positions
        attacking_side,             # attacking team direction, "right_to_left" or "left_to_right"
        pitch_length,
        run_time,                   # duration of run (in seconds)
        ):
    """
    The goal of this function is to move the runner to the 
    correct no run state position. 
    
    The output is the new x and y coordinates of the runner. 
    """
    
    # Get runners initial position
    runner_x = attacker_positions_start.loc[
        attacker_positions_start.player_id == runner_id, "x"].iloc[0]
    runner_y = attacker_positions_start.loc[
        attacker_positions_start.player_id == runner_id, "y"].iloc[0]
    
    # Remove the runner from the attacker dfs
    attacker_positions_start = attacker_positions_start.loc[
        attacker_positions_start.player_id != runner_id]
    attacker_positions_end = attacker_positions_end.loc[
        attacker_positions_end.player_id != runner_id]
    
    # Compute the average change (down field) in other attackers
    vert_change = attacker_positions_end["x"] - attacker_positions_start["x"]
    vert_change_avg = vert_change.mean()
    
    # Define new variable for altered position
    runner_x_new = runner_x + vert_change_avg
    runner_y_new = runner_y
    
    # Check how runner compares to their teams def line height 
    # Alter runners position if they are last defender
    if attacking_side == "left_to_right":
        line_height = attacker_positions_end["x"].nsmallest(2).iloc[-1]
        if runner_x_new < line_height:
            runner_x_new = line_height
    else:
        line_height = attacker_positions_end["x"].nlargest(2).iloc[-1]
        if runner_x_new > line_height:
            runner_x_new = line_height  
            
    # Check how runnner compares to opponets def line height 
    # Alter position if they are offside
    if attacking_side == "left_to_right":
        line_height_opp = defender_positions_end["x"].nlargest(2).iloc[-1]
        if runner_x_new > line_height_opp:
            runner_x_new = line_height_opp - 1 # the -1 keeps them safely onside
    else:
        line_height_opp = defender_positions_end["x"].nsmallest(2).iloc[-1]
        if runner_x_new < line_height_opp:
            runner_x_new = line_height_opp + 1   
            
    # Mover runner horizontally if too close to teammates
    max_shift = min(3*run_time, 10.0)  # change

    if runner_x_new < -pitch_length/6 or runner_x_new > pitch_length/6:
        threshold = 5.0
    else:
        threshold = 8.0

    # teammates (runner already removed earlier)
    teammates_xy = attacker_positions_end[["x", "y"]].to_numpy()

    # default: stay put if no teammates or no movement allowed
    runner_y_new = runner_y

    if len(teammates_xy) > 0 and max_shift > 0:
        # Explicit segment: within +/- max_shift of original y
        y_min = runner_y - max_shift
        y_max = runner_y + max_shift

        # Candidate y positions along the line x = runner_x_new
        # (tune num for resolution)
        y_candidates = np.linspace(y_min, y_max, num=41)

        # Precompute dx to teammates
        dx = runner_x_new - teammates_xy[:, 0]  # shape (n_teammates,)

        min_dists = []
        for y_c in y_candidates:
            dy = y_c - teammates_xy[:, 1]
            dists = np.sqrt(dx**2 + dy**2)
            min_dists.append(dists.min())
        min_dists = np.array(min_dists)

        # Candidates that meet the spacing threshold
        ok = np.where(min_dists >= threshold)[0]

        if ok.size > 0:
            # Among those, pick the one closest to original runner_y
            best_idx = ok[np.argmin(np.abs(y_candidates[ok] - runner_y))]
        else:
            # No perfect spot; pick the y with the largest minimum distance
            best_idx = np.argmax(min_dists)

        runner_y_new = float(y_candidates[best_idx])

    return runner_x_new, runner_y_new

def position_score(
        runner_x,               # runner position x (no run position)
        runner_y,               # runner position y (no run position)
        ball_x,                 # ball position x
        ball_y,                 # ball position y
        cand_def_x,             # candidate position x
        cand_def_y,             # candidate position y
        attacking_side,         # "right_to_left" or "left_to_right"
        main_def_id,            # player_id of main defender
        defender_positions_end, # positions of all defenders at end
        pitch_length,           # length of pitch (width not needed)
        ):
    """
    The goal of this function is to score potential positions for
    the defender designated as the "main defender."
    
    Takes a candidate defender position and test it on three
    elements:
        - how close they are to the optimal defensive position,
        - how close they are to the passing lane, and
        - how much does the position keep the integrity of the 
          overall shape. 
    These three factors are given a score function, and weighed
    differently depending on multiple factors. 
    
    The output is a positional "score". The larger the score, the
    better. 
    """
    
    # Determine the goal location
    goal_y = 0
    if attacking_side == "left_to_right":
        goal_x = pitch_length/2
    else: 
        goal_x = -pitch_length/2
        
    # Compute what optimal is (depends on distance to goal)
    dist_runner_to_goal = math.hypot(runner_x - goal_x, runner_y-goal_y)
    optimal_dist = min(5, 0.05*dist_runner_to_goal)
    optimal_x, optimal_y = position_on_line(
        runner_x,
        runner_y,
        goal_x,
        goal_y,
        distance=optimal_dist
        )
    # Compute how far cand position is from optimal and invert
    dist_to_optimal = math.hypot(cand_def_x-optimal_x,cand_def_y-optimal_y)
    optimal_score = 1/(1+dist_to_optimal) # value between 0 and 1
    
    # Compute distance to point in passing lane
    lane_dist = min(5, 0.05*dist_runner_to_goal)
    pass_lane_target_x, pass_lane_target_y = position_on_line(
        runner_x, 
        runner_y, 
        ball_x, 
        ball_y,
        distance=lane_dist
        )
    # Compute how far cand position is from lane point and invert
    dist_to_lane = math.hypot(
        pass_lane_target_x-cand_def_x,
        pass_lane_target_y-cand_def_y)
    passing_lane_score = 1/(1+dist_to_lane)
    
    # Determine if main_def_id is a "defender"
    defs = ["RCB", "LCB", "RB", "LB", "CB"]
    main_def_role = defender_positions_end.loc[
        defender_positions_end.player_id == main_def_id, "position"].iloc[0]
    role = ("def" if main_def_role in defs else "not_def")
    # Create DataFrame of all other defender 
    other_def_end = defender_positions_end.loc[
        (defender_positions_end.player_id != main_def_id)]

    if role == "def":
        # If a player is a defender, we get def line component
        # Compute other defenders defensive line height
        other_def_def_end = other_def_end.loc[
            (other_def_end.position.isin(defs))]
        def_line_height = other_def_def_end["x"].mean()
        # Compute how far player is from line
        line_error = (def_line_height - cand_def_x)**2 # using square here
    else:
        line_error = 0
    # All players get a measurement of how equally spaced they are
    other_def_end["dist_to_main"] = np.sqrt(
        (other_def_end["x"]-cand_def_x)**2 + 
        (other_def_end["y"]-cand_def_y)**2
        )
    # Compute error based on the distance to 3 closest teammates
    close_vect = other_def_end["dist_to_main"].nsmallest(3).to_numpy()
    close_error = np.exp(-close_vect).sum() # decays exp
    # Combine the close_error and line_error into one score
    shape_score = 1/(1+line_error+close_error)
    
    # Combine into single score
    # Weights for components depend on position
    if role == "def":
        w_near = np.array([0.5, 0.0, 0.5]) 
        w_far = np.array([0.2, 0.1, 0.7])  
    else:
        w_near = np.array([0.6, 0.2, 0.2])
        w_far = np.array([0.3, 0.5, 0.2])
    # Compute how far the ball and runner are to goal
    ball_to_goal = math.hypot(ball_x-goal_x,ball_y-goal_y)
    runner_to_goal = math.hypot(runner_x - goal_x, runner_y - goal_y)
    dist_goal = softmin([ball_to_goal, runner_to_goal])
    # Determine wts by how far away the action is from goal
    far_mult = 3 # creates far cutoff
    D0 = 20 # near cutoff
    g = 1.0 - dist_goal / (far_mult * D0)
    g = max(0.0, min(1.0, g))
    w = g*w_near + (1-g)*w_far # creates wts array
    
    # Computes score 
    position_score = (
        w[0]*optimal_score + 
        w[1]*passing_lane_score +
        w[2]*shape_score
        )
    
    return position_score


def main_def_id(
        runner_x,
        runner_y,
        defender_positions_start,   # df of just def team positions
        attacking_side,
        pitch_length,
        ):
    """
    Determines the ID of the main defender, if any. If no 
    main def, returns None.
    
    """
    
    # Compute how close defenders are to runnner.
    defender_positions_start["dist_to_runner"] = (
        np.sqrt((defender_positions_start["x"] - runner_x)**2 + 
                (defender_positions_start["y"] - runner_y)**2))
    
    # Determine optimal defensive positioning.
    # Positioning depends on attacking_side
    if attacking_side == "left_to_right":
        goal_x = pitch_length/2
        goal_y = 0
    else: 
        goal_x = -pitch_length/2
        goal_y = 0
    # Compute optimal position. 
    optimal_def_x, optimal_def_y = position_on_line(
        runner_x, runner_y, goal_x, goal_y, distance=2)
    
    # Find which defender is closest to optimal. 
    dx = defender_positions_start["x"] - optimal_def_x
    dy = defender_positions_start["y"] - optimal_def_y
    d  = np.hypot(dx, dy)
    condition = defender_positions_start["dist_to_runner"] <= 8
    defender_positions_start["dist_to_optimal"] = np.where(condition, d, np.nan)
    
    # Determine who the main defender is.
    if defender_positions_start["dist_to_optimal"].isna().all():
        main_def = None
    else:
        main_def = defender_positions_start.loc[
            defender_positions_start["dist_to_optimal"].idxmin(), 'player_id']
    
    return main_def
    
def no_run_state_ball(
        run_info,   # series containing advance run infomation (from skillcorner)
        ):
    """
    Always gives the ball to the designated ball carrier.
    
    """
    # Determine ball carrier
    ball_carrier_id = run_info["player_in_possession_id"]
    match_data = run_info["match_data"]
    frame_end_data = run_info["frame_end_data"]
    # compute end of run state
    player_positions = create_player_positions_2(
        match_data, frame_end_data)

    x = player_positions.loc[
        player_positions["player_id"] == ball_carrier_id]["x"].iloc[0]
    y = player_positions.loc[
        player_positions["player_id"] == ball_carrier_id]["y"].iloc[0]


    return x, y


def classify_attackers_marking(
        attackers_end,  # df of attackers positions at end of run
        defenders_end,  # df of defenders positions at end of run
        r_mark=7.0,     # marking maximum distance
        goal_x=None,    # defending team goal x coord
        ):
    """
    Split attackers into 'unmarked' and 'marked' based on distance to the
    nearest goal-side defender in defenders_end.

    A defender only counts as a marker if they lie between the attacker
    and the goal along the x-axis (i.e., goal-side).

    Returns array of positions of marked and unmarked defenders.
    
    """
    if attackers_end.empty or defenders_end.empty or goal_x is None:
        return np.empty((0, 2)), np.empty((0, 2))

    attackers_xy = attackers_end[["x", "y"]].to_numpy()
    defenders_xy = defenders_end[["x", "y"]].to_numpy()

    # Pairwise distances between attackers and defenders
    dx_ad = attackers_xy[:, 0][:, None] - defenders_xy[:, 0][None, :]
    dy_ad = attackers_xy[:, 1][:, None] - defenders_xy[:, 1][None, :]
    dist_ad = np.sqrt(dx_ad**2 + dy_ad**2)  

    # Goal-side mask:
    # Just take x coords so we can broadcast
    att_x = attackers_xy[:, 0][:, None]   
    def_x = defenders_xy[:, 0][None, :]   

    # For each attacker: is goal to the left (smaller x) or right (larger x)?
    goal_left = (goal_x < att_x)          

    # If goal is to LEFT of attacker: goal_x <= def_x <= att_x
    valid_left  = (def_x >= goal_x) & (def_x <= att_x)
    # If goal is to RIGHT of attacker: att_x <= def_x <= goal_x
    valid_right = (def_x <= goal_x) & (def_x >= att_x)

    goal_side_mask = np.where(goal_left, valid_left, valid_right)  

    # Invalidate defenders that are not goal-side for a given attacker
    dist_ad[~goal_side_mask] = np.inf

    # Now nearest goal-side defender distance
    min_dists = dist_ad.min(axis=1)  # (A,)

    # Marked if a goal-side defender is within r_mark
    unmarked_mask = min_dists > r_mark

    unmarked_xy = attackers_xy[unmarked_mask]
    marked_xy   = attackers_xy[~unmarked_mask]

    return unmarked_xy, marked_xy


def choose_secondary_defender_y(
        sd_start_y,                         # potential y range for secondary defender
        sd_end_y,                           
        sd_end_x,                           # fixed x position of sec def
        runner_x_new,
        runner_y_new,
        ball_x,
        ball_y,
        teammates_new,
        attackers_end,
        defenders_end,
        sigma_action=15.0,                  # action score parameter, larger -> more wt
        sigma_coverage=10.0,                # coverage score parameter, larger => more wt
        alpha_marked=0.3,                   # smaller wt for marker def
        w_action=0.40,                      # how much action is weighted
        w_coverage=0.60,                    # how much coverage is weighted
        r_mark=7.0,                         # threshold used to determine marked/unmarked attackers
        num_candidates=31,                  # number of test pts
        max_unmarked_dist_to_segment=8.0,   # only attackers within this dist to segment are considered
        in_final_third=False,
        goal_x=None,           
        min_spacing_default=8.0,            # teammate spacing cutoffs 
        min_spacing_final_third=4.0, 
    ):
    """
    Choose the best y-position for the secondary defender along
    the segment from sd_start_y to sd_end_y, keeping x fixed at sd_end_x.

    Balances:
      - spacing from teammates (hard cutoff: 8m normally, 4m in final third)
      - staying near the 'action' (ball + runner)
      - coverage of attackers near this vertical segment.

    If no positions satisfy the spacing cutoff, we ignore spacing and
    pick the best position using only action + coverage.
    """
    # Candidate y-values along the segment
    y_candidates = np.linspace(sd_start_y, sd_end_y, num=num_candidates)

    # Teammates (defensive team) in the new state, excluding second_def 
    if teammates_new is None or teammates_new.empty:
        teammates_xy = np.empty((0, 2))
    else:
        teammates_xy = teammates_new[["x", "y"]].to_numpy()

    # --- Restrict attackers to those near this vertical segment (within 8m by default) ---
    if attackers_end is None or attackers_end.empty or defenders_end is None or defenders_end.empty or goal_x is None:
        unmarked_attackers_xy = np.empty((0, 2))
        marked_attackers_xy   = np.empty((0, 2))
    else:
        attackers_xy = attackers_end[["x", "y"]].to_numpy()
        attackers_xy_near = filter_points_near_vertical_segment(
            attackers_xy,
            x0=sd_end_x,
            y1=sd_start_y,
            y2=sd_end_y,
            max_dist=max_unmarked_dist_to_segment,
        )

        if attackers_xy_near.size == 0:
            # No attackers near the segment -> no coverage signal
            unmarked_attackers_xy = np.empty((0, 2))
            marked_attackers_xy   = np.empty((0, 2))
        else:
            # Build a minimal DF just with x,y for classification.
            attackers_near_df = pd.DataFrame(attackers_xy_near, columns=["x", "y"])
            unmarked_attackers_xy, marked_attackers_xy = classify_attackers_marking(
                attackers_near_df, defenders_end, r_mark=r_mark, goal_x=goal_x
            )

    # Spacing threshold for the secondary defender vs teammates
    min_spacing = (
        min_spacing_final_third if in_final_third else min_spacing_default
    )

    # ---------- Precompute action+coverage for all candidates once ----------
    base_scores = np.empty_like(y_candidates, dtype=float)
    cand_x = sd_end_x  # x is fixed

    for i, cand_y in enumerate(y_candidates):
        # --- (1) ACTION AWARENESS: close to ball AND runner (sum distance) ---
        d_ball = math.hypot(cand_x - ball_x,     cand_y - ball_y)
        d_runner = math.hypot(cand_x - runner_x_new, cand_y - runner_y_new)
        d_action = d_ball + d_runner
        action_score = math.exp(-d_action / sigma_action)

        # --- (2) COVERAGE: prefer being near unmarked attackers, but
        #                   also get smaller positive credit for marked ones ---
        if (unmarked_attackers_xy.size == 0) and (marked_attackers_xy.size == 0):
            coverage_score = 0.0
        else:
            unmarked_score = 0.0
            marked_score = 0.0

            if unmarked_attackers_xy.size > 0:
                du = np.sqrt(
                    (unmarked_attackers_xy[:, 0] - cand_x)**2 +
                    (unmarked_attackers_xy[:, 1] - cand_y)**2
                )
                # larger when closer to unmarked attackers
                unmarked_score = float(np.exp(-du / sigma_coverage).sum())

            if marked_attackers_xy.size > 0:
                dm = np.sqrt(
                    (marked_attackers_xy[:, 0] - cand_x)**2 +
                    (marked_attackers_xy[:, 1] - cand_y)**2
                )
                # marked still positive, but will get down-weighted by alpha_marked
                marked_score = float(np.exp(-dm / sigma_coverage).sum())

            # NO penalty for marked; they still help but less:
            coverage_raw = unmarked_score + alpha_marked * marked_score
            coverage_score = math.tanh(coverage_raw)  # squash to [0, 1) effectively

        base_scores[i] = (
            w_action  * action_score +
            w_coverage * coverage_score
        )

    # Degenerate safety: if everything is nan/inf, just keep end-of-run y
    if not np.isfinite(base_scores).any():
        return sd_end_x, float(sd_end_y)

    # ---------- PASS 1: enforce spacing cutoff ----------
    if teammates_xy.size > 0:
        # Vectorized spacing: distance from each candidate y to closest teammate
        dx_t = sd_end_x - teammates_xy[:, 0]        # (T,)
        dx_t2 = dx_t**2
        # (C, T) of y differences
        dy_all = y_candidates[:, None] - teammates_xy[None, :, 1]
        d_all = np.sqrt(dx_t2[None, :] + dy_all**2) # (C, T)
        min_dists = d_all.min(axis=1)               # (C,)
        spacing_ok = (min_dists >= min_spacing)
    else:
        spacing_ok = np.ones_like(y_candidates, dtype=bool)

    if spacing_ok.any():
        scores_strict = np.where(spacing_ok, base_scores, -np.inf)
        best_idx = int(np.argmax(scores_strict))
        if np.isfinite(scores_strict[best_idx]):
            sd_new_x = sd_end_x
            sd_new_y = float(y_candidates[best_idx])
            return sd_new_x, sd_new_y

    # ---------- PASS 2: no one passed spacing -> ignore spacing entirely ----------
    best_idx = int(np.argmax(base_scores))
    if not np.isfinite(base_scores[best_idx]):
        # ultra-degenerate fallback
        sd_new_y = float(sd_end_y)
    else:
        sd_new_y = float(y_candidates[best_idx])

    sd_new_x = sd_end_x
    return sd_new_x, sd_new_y

def no_run_state(
        match_data, 
        frame_start_data,
        frame_end_data, 
        run_info,
        ):
    """
    Returns player positions dataframe with all updated positions.

    """
    # STEP 0: Preliminary definitions. 
    
    # From run_info.
    frame_start = run_info["frame_start"]
    frame_end = run_info["frame_end"]
    att_team_id = run_info["team_id"]
    runner_id = run_info["player_id"]
    attacking_side = run_info["attacking_side"]
    
    # Pitch length.
    pitch_length = match_data["pitch_length"]
    
    # defenders goal
    half_L = pitch_length / 2.0
    if attacking_side == "left_to_right":
        # Attackers go to +half_L, defenders protect -half_L
        defenders_goal_x = -half_L
    else:  # "right_to_left"
        # Attackers go to -half_L, defenders protect +half_L
        defenders_goal_x = half_L

    
    # Create DataFrame player_positions for start and end frame. 
    player_positions_start = create_player_positions_2(
        match_data, frame_start_data)
    player_positions_end = create_player_positions_2(
        match_data, frame_end_data)
    
    # Get team_id for attacking and defending sides. 
    team_ids = player_positions_start.team_id.unique()
    def_team_id = (team_ids[1] if team_ids[0] == att_team_id else team_ids[0])
    
    # Attacking side at beginning and end.
    attacker_positions_start = player_positions_start[
        player_positions_start["team_id"] == att_team_id]
    attacker_positions_end = player_positions_end[
        player_positions_end["team_id"] == att_team_id]
    
    # Original runner position (at start of run).
    runner_x = player_positions_start.loc[
        player_positions_start.player_id == runner_id, "x"].iloc[0]
    runner_y = player_positions_start.loc[
        player_positions_start.player_id == runner_id, "y"].iloc[0]
    
    # Runner position at end of run.
    runner_x_end = player_positions_end.loc[
        player_positions_end.player_id == runner_id, "x"].iloc[0]
    runner_y_end = player_positions_end.loc[
        player_positions_end.player_id == runner_id, "y"].iloc[0]
    
    # Defending side at beginning and end. 
    defender_positions_start = player_positions_start[
        player_positions_start.team_id == def_team_id]
    defender_positions_end = player_positions_end[
        player_positions_end.team_id == def_team_id]
    
    # Time length of run.
    run_time = (frame_end - frame_start) * 0.1  # in seconds
    
    # Ball position at end. 
    #!!! In the no-run state we assume the ball carrier maintains possession
    ball_x, ball_y = no_run_state_ball(run_info)
    
    # DataFrame where we will store the new state. 
    player_positions_new = player_positions_end.copy()
    
    # STEP 1: Runner Positioning.
    runner_x_new, runner_y_new = move_runner(
        runner_id, 
        attacker_positions_end, 
        defender_positions_end, 
        attacker_positions_start,
        defender_positions_start,
        attacking_side,
        pitch_length,
        run_time)
    
    # Add position to DF. 
    player_positions_new.loc[
        player_positions_new.player_id == runner_id, "x"] = runner_x_new
    player_positions_new.loc[
        player_positions_new.player_id == runner_id, "y"] = runner_y_new
    
    # STEP 2: Identify main and secondary defenders.
    
    # 2.1: Identify the main defender. 
    main_def = main_def_id(
        runner_x, 
        runner_y, 
        defender_positions_start, 
        attacking_side, 
        pitch_length)  # returns None or player ID
    
    
    # 2.2: Identify secondary defender at the end of the run.
    second_def = main_def_id(
        runner_x_end, 
        runner_y_end, 
        defender_positions_end, 
        attacking_side, 
        pitch_length)
    
    # Eliminate second_def if same as main
    if second_def == main_def:
        second_def = None
    
    # Eliminate second_def if they are closer to the ball carrier than to the runner
    if second_def is not None:
        sd_x_end = player_positions_end.loc[
            player_positions_end.player_id == second_def, "x"].iloc[0]
        sd_y_end = player_positions_end.loc[
            player_positions_end.player_id == second_def, "y"].iloc[0]
        
        # distance from secondary defender to ball carrier
        dist_sd_ball = math.hypot(sd_x_end - ball_x, sd_y_end - ball_y)
        # distance from secondary defender to runner (end of run)
        dist_sd_runner = math.hypot(sd_x_end - runner_x_end, sd_y_end - runner_y_end)

        if dist_sd_ball < dist_sd_runner:
            second_def = None
            
    
    # 2.3: Check how close main def is to ball_carrier at end of frame.
    # If main_def is closer to ball_carrier and within 10 m leave them there
    if main_def is not None:
        md_x = player_positions_end.loc[
            player_positions_end.player_id == main_def]["x"].iloc[0]
        md_y = player_positions_end.loc[
            player_positions_end.player_id == main_def]["y"].iloc[0]
        
        md_to_bc_dist = math.hypot(ball_x-md_x,ball_y-md_y)
        md_to_r_dist = math.hypot(runner_x_end-md_x,runner_y_end-md_y)
        

        if (md_to_bc_dist <= 10) and (md_to_bc_dist < md_to_r_dist):
            if second_def is not None:
                main_def = second_def
            else:
                main_def = None 
    
    # STEP 3: Move defenders. 
    jog_speed = 3
    
    # 3.1: Alter main defender positioning. 
    if main_def is not None:
        
        # Compute grid of points within the ball. 
        pts = grid_points_in_disk(
            runner_x_new, runner_y_new, 
            min(run_time * jog_speed, 12)
        )
        # Iterate over grid points and compute positional score. 
        pos_score = np.zeros(len(pts))
        for i in range(len(pts)):
            pt = pts[i]
            x = pt[0]
            y = pt[1]
            # Compute position score
            pos_score[i] = position_score(
                runner_x_new, runner_y_new, 
                ball_x, ball_y,
                x, y, 
                attacking_side, 
                main_def, 
                defender_positions_end,
                pitch_length)
        
        # Take defender position with maximal position score.
        max_pos_score_idx = pos_score.argmax()
        main_def_x = pts[max_pos_score_idx][0]
        main_def_y = pts[max_pos_score_idx][1]
        
        # Add new main defender location to DF.
        player_positions_new.loc[
            player_positions_new.player_id == main_def, "x"] = main_def_x
        player_positions_new.loc[
            player_positions_new.player_id == main_def, "y"] = main_def_y
    
    # Create attacker and defender positions
    defenders_new = player_positions_new.loc[
        player_positions_new.team_id == def_team_id
        ]
    attackers_new = player_positions_new.loc[
        player_positions_new.team_id == att_team_id
        ]
    
    # 3.2: Alter secondary defender's position (if present).
    if second_def is not None:
        # First, check if they are the closest defender to the ball carrier.
        closest_def_id = closest_defender_to_point(
            defenders_new, ball_x, ball_y)

        if second_def == closest_def_id:
            # Primary ball presser: keep their end-of-run position.
            second_def = None
            pass
        
        else:
            # Pull basic info for second_def
            sd_end_x = player_positions_end.loc[
                player_positions_end.player_id == second_def, "x"
            ].iloc[0]
            sd_end_y = player_positions_end.loc[
                player_positions_end.player_id == second_def, "y"
            ].iloc[0]
            
            # Compute range of movement
            sd_move = min(8,jog_speed*run_time)

            # Teammates on defensive team in new state (excluding second_def)
            teammates_new = player_positions_new.loc[
                (player_positions_new.team_id == def_team_id) &
                (player_positions_new.player_id != second_def)
            ]


            in_final_third = is_ball_in_attacking_final_third(
                ball_x, pitch_length, attacking_side
                )
            

            # Use helper to choose best y along [sd_start_y, sd_end_y]
            sd_new_x, sd_new_y = choose_secondary_defender_y(
                sd_start_y=sd_end_y + sd_move,
                sd_end_y=sd_end_y - sd_move,
                sd_end_x=sd_end_x,
                runner_x_new=runner_x_new,
                runner_y_new=runner_y_new,
                ball_x=ball_x,
                ball_y=ball_y,
                teammates_new=teammates_new,
                attackers_end=attackers_new,
                defenders_end=defenders_new,
                in_final_third=in_final_third,
                goal_x=defenders_goal_x,
            )

            player_positions_new.loc[
                player_positions_new.player_id == second_def, "x"
            ] = sd_new_x
            player_positions_new.loc[
                player_positions_new.player_id == second_def, "y"
            ] = sd_new_y

    # STEP 4: Update velocities (average from start -> new state)
    if run_time > 0:
        # 4a) Runner
        runner_start = player_positions_start.loc[
            player_positions_start.player_id == runner_id].iloc[0]
        runner_new = player_positions_new.loc[
            player_positions_new.player_id == runner_id].iloc[0]
        runner_vx = (runner_new["x"] - runner_start["x"]) / run_time
        runner_vy = (runner_new["y"] - runner_start["y"]) / run_time
        player_positions_new.loc[
            player_positions_new.player_id == runner_id, ["v_x", "v_y"]] = [
                runner_vx, runner_vy]
        
        # 4b) Main defender (if any)
        if main_def is not None:
            main_start = player_positions_start.loc[
                player_positions_start.player_id == main_def].iloc[0]
            main_new = player_positions_new.loc[
                player_positions_new.player_id == main_def].iloc[0]
            main_vx = (main_new["x"] - main_start["x"]) / run_time
            main_vy = (main_new["y"] - main_start["y"]) / run_time
            player_positions_new.loc[
                player_positions_new.player_id == main_def, ["v_x", "v_y"]] = [
                    main_vx, main_vy]
        
        # 4c) Secondary defender (if distinct and present)
        if (second_def is not None) and (second_def != main_def):
            sec_start = player_positions_start.loc[
                player_positions_start.player_id == second_def].iloc[0]
            sec_new = player_positions_new.loc[
                player_positions_new.player_id == second_def].iloc[0]
            sec_vx = (sec_new["x"] - sec_start["x"]) / run_time
            sec_vy = (sec_new["y"] - sec_start["y"]) / run_time
            player_positions_new.loc[
                player_positions_new.player_id == second_def, ["v_x", "v_y"]] = [
                    sec_vx, sec_vy]
    
    return player_positions_new

