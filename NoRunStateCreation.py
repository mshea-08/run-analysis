#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 14:40:50 2025

@author: meredithshea
"""

import numpy as np
import TrackingFunctions as track
from math import hypot

def position_on_line(x1,y1,x2,y2,distance=2):
    dx, dy = x2 - x1, y2 - y1
    L = hypot(dx, dy)
    if L == 0:
        raise ValueError("Direction undefined because the two points are identical.")
    ux, uy = dx / L, dy / L
    return x1 + distance * ux, y1 + distance * uy


def main_def_possible_position_triangle(x1, y1, x2, y2):
    """
    Parameters
    ----------
    (x1,y1) -- initial defender position
    (x2,y2) -- final defender position

    Returns
    -------
    b1 = (x,y) -- one triangle vertex
    b2 = (x,y) -- another triangle vertex
    apex = (x2,y2)

    """
    dx, dy = x2 - x1, y2 - y1
    L = hypot(dx, dy)
    if L == 0:
        raise ValueError("Height undefined: (x1,y1) and (x2,y2) are identical.")
    # Unit vector perpendicular to the height (rotate by +90°)
    ux_perp, uy_perp = -dy / L, dx / L
    # Base length equals height length L; half-base vector:
    hx, hy = (L / 2) * ux_perp, (L / 2) * uy_perp

    b1 = (x1 + hx, y1 + hy)
    b2 = (x1 - hx, y1 - hy)
    apex = (x2, y2)
    
    return b1, b2, apex

def softmin(x, T=6.0):
    x = np.asarray(x, float)
    a = -x / T
    a -= a.max()
    w = np.exp(a)
    w /= w.sum()
    return float((w * x).sum())

def grid_points_in_triangle(v1, v2, v3, spacing=0.5, include_boundary=True, eps=1e-9):
    """
    Return an (M,2) array of grid points spaced `spacing` apart that lie inside
    the triangle with vertices v1, v2, v3. 

    Parameters
    ----------
    v1, v2, v3 : (2,) array-like
        Triangle vertices (x, y).
    spacing : float
        Grid spacing in same units as vertices (e.g., meters).
    include_boundary : bool
        If True, include points on the triangle edges.
    eps : float
        Tolerance for boundary inclusion with floating point arithmetic.

    Returns
    -------
    pts : ndarray shape (M, 2)
        Each row is (x,y)
    """
    v1 = np.asarray(v1, float)
    v2 = np.asarray(v2, float)
    v3 = np.asarray(v3, float)

    # Bounding box aligned to the grid (deterministic)
    xmin = min(v1[0], v2[0], v3[0])
    xmax = max(v1[0], v2[0], v3[0])
    ymin = min(v1[1], v2[1], v3[1])
    ymax = max(v1[1], v2[1], v3[1])

    gx_min = np.floor(xmin / spacing) * spacing
    gx_max = np.ceil (xmax / spacing) * spacing
    gy_min = np.floor(ymin / spacing) * spacing
    gy_max = np.ceil (ymax / spacing) * spacing

    xs = np.arange(gx_min, gx_max + eps, spacing)
    ys = np.arange(gy_min, gy_max + eps, spacing)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    grid = np.c_[X.ravel(), Y.ravel()]

    # Vectorized barycentric membership test
    a, b, c = v1, v2, v3
    v0 = c - a
    v1v = b - a
    v2v = grid - a  # shape (N,2)

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1v)
    d11 = np.dot(v1v, v1v)
    d20 = (v2v * v0).sum(axis=1)
    d21 = (v2v * v1v).sum(axis=1)
    denom = d00 * d11 - d01 * d01

    # Handle degenerate triangles (colinear points)
    if abs(denom) < eps:
        return np.empty((0, 2))  # no area → no interior grid points

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    if include_boundary:
        mask = (u >= -eps) & (v >= -eps) & (w >= -eps)
    else:
        mask = (u > eps) & (v > eps) & (w > eps)

    pts = grid[mask]

    # Deterministic ordering (by y, then x)
    if len(pts):
        order = np.lexsort((pts[:, 0], pts[:, 1]))
        pts = pts[order]
    return pts

def position_score(
        runner_x,
        runner_y,
        ball_x,
        ball_y,
        cand_def_x,
        cand_def_y,
        attacking_side,
        main_def_id, # player_id of main defender
        defender_positions_end, # positions of all defenders
        pitch_length
        ):
    # Compute three elements:
        # How far from "optimal" in the def position? 
        # How close is the defender to the passing lane?
        # How much does the defender keep shape integrity?
    # Where is the goal.
    goal_y = 0
    if attacking_side == "left_to_right":
        goal_x = pitch_length/2
    else: 
        goal_x = -pitch_length/2
        
    # STEP 1: Optimal score.
    # Compute what optimal is (depends on distance to goal).
    dist_runner_to_goal = hypot(runner_x - goal_x, runner_y-goal_y)
    optimal_dist = min(5, 0.05*dist_runner_to_goal)
    optimal_x, optimal_y = position_on_line(
        runner_x,runner_y,goal_x,goal_y,distance=optimal_dist)
    # Compute how far runner in from optimal and invert. 
    dist_to_optimal = hypot(cand_def_x-optimal_x,cand_def_y-optimal_y)
    optimal_score = 1/(1+dist_to_optimal) # value between 0 and 1
    
    # STEP 2: Passing lane score. 
    pass_lane_target_x, pass_lane_target_y = position_on_line(
        runner_x, runner_y, 
        ball_x, ball_y,
        distance=3)
    dist_to_lane = hypot(
        pass_lane_target_x-cand_def_x,
        pass_lane_target_y-cand_def_y)
    passing_lane_score = 1/(1+dist_to_lane)
    
    # STEP 3: Shape integrity score.
    # Is the main defender's role Def?
    defs = ["RCB", "LCB", "RB", "LB", "CB"]
    main_def_role = defender_positions_end.loc[
        defender_positions_end.player_id == main_def_id, "position"].iloc[0]
    role = ("def" if main_def_role in defs else "not_def")
    # Create DataFrame of the other defenders (not the main).
    other_def_end = defender_positions_end.loc[
        (defender_positions_end.player_id != main_def_id)]
    # Defenders get a defensive line error.
    if role == "def":
        # Compute other defenders defensive line height.
        other_def_def_end = other_def_end.loc[
            (other_def_end.position.isin(defs))]
        def_line_height = other_def_def_end["x"].mean()
        # Compute how far player is from line.
        line_error = (def_line_height - cand_def_x)**2 # using square here
    else:
        line_error = 0
    # All players get a measurement of how equally spaced they are.
    other_def_end["dist_to_main"] = np.sqrt(
        (other_def_end["x"]-cand_def_x)**2 + 
        (other_def_end["y"]-cand_def_y)**2)
    close_vect = other_def_end["dist_to_main"].nsmallest(3).to_numpy()
    close_error = np.exp(-close_vect).sum()
    # Combine the close_error and line_error into one score.
    shape_score = 1/(1+line_error+close_error)
    
    # STEP 4: Combine scores into single score. 
    w_near = np.array([0.6, 0.2, 0.2])
    w_far = np.array([0.3, 0.35, 0.35])
    ball_to_goal = hypot(ball_x-goal_x,ball_y-goal_y)
    runner_to_goal = hypot(runner_x - goal_x, runner_y - goal_y)
    dist_goal = softmin([ball_to_goal, runner_to_goal])
    far_mult = 3 # creates far cutoff
    D0 = 12 # near cutoff
    g = 1.0 - dist_goal / (far_mult * D0)
    w = g*w_near + (1-g)*w_far
    position_score = w[0]*optimal_score + w[1]*passing_lane_score + w[2]*shape_score
    
    return position_score
    
def no_run_state(
        match_data, 
        match_tracking_data, # tracking data for whole game with velocities
        frame_start, # starting frame of run
        frame_end, # ending frame of run
        runner_id, # player_id of runner
        attacking_side # right_to_left or left_to_right, stored in run DataFrame
        ):
    # STEP 0: Preliminary definitions. 
    # Pitch length.
    pitch_length = match_data["pitch_length"]
    # Create DataFrame player_positions for start and end frame. 
    player_positions_start = track.create_player_positions(
        match_data, match_tracking_data, frame_start)
    player_positions_end = track.create_player_positions(
        match_data, match_tracking_data, frame_end)
    # Get team_id for attacking and defending sides. 
    team_ids = player_positions_start.team_id.unique()
    att_team_id = player_positions_start.loc[
        player_positions_start.player_id == runner_id, "team_id"].iloc[0]
    def_team_id = (team_ids[1] if team_ids[0] == att_team_id else team_ids[0])
    # Ball position at end.
    ball_x, ball_y = track.ball_position(match_tracking_data, frame_end)
    # STEP 1: Runner Positioning.
    # Ideas:
        # (1) Runner is frozen at beginning position of run. (DONE)
        # (2) Check that runner does not become last defender, if they do align them
            # with defensive line height. (DONE)
        # (3) Runner is allowed to separate from ball carrier. 
    # Get runners initial position
    runner_x = player_positions_start.loc[
        player_positions_start.player_id == runner_id, "x"].iloc[0]
    runner_y = player_positions_start.loc[
        player_positions_start.player_id == runner_id, "y"].iloc[0]
    # Define new variables where runner position can be modified from.
    runner_x_final = runner_x
    runner_y_final = runner_y
    # Compute defensive line height for attacking team at end of run
    attacker_positions_end = player_positions_end[
        player_positions_end["team_id"] == att_team_id]
    # Check how runner compares to defensive line height. 
    # Alter runners position if frozen state becomes last defender. 
    if attacking_side == "left_to_right":
        line_height = attacker_positions_end["x"].nsmallest(2).iloc[-1]
        if runner_x < line_height:
            runner_x_final = line_height
    else:
        line_height = attacker_positions_end["x"].nlargest(2).iloc[-1]
        if runner_x > line_height:
            runner_x_final = line_height    
    # TODO: Allow runner to separate from ball carrier. 
    # Replace runners position in end frame. 
    player_positions_end.loc[
        player_positions_end.player_id == runner_id, "x"] = runner_x_final
    runner_y = player_positions_end.loc[
        player_positions_end.player_id == runner_id, "y"] = runner_y_final 
    
    # STEP 2: Adjust main defender.
    # Idea:
        # (1) Identify main defender (might be none). (DONE)
        # (2) Compute triangle of posible positions. 
    # Create DataFrame of just defenders (at start and end).
    defender_positions_start = player_positions_start[
        player_positions_start.team_id == def_team_id]
    defender_positions_end = player_positions_end[
        player_positions_end.team_id == def_team_id]
    # Compute how close defenders are to runnner.
    defender_positions_start["dist_to_runner"] = (
        np.sqrt((defender_positions_start["x"] - runner_x)**2 + 
                (defender_positions_start["y"] - runner_y)**2))
    # Determine optimal defensive positioning.
    if attacking_side == "left_to_right":
        goal_x = pitch_length/2
        goal_y = 0
    else: 
        goal_x = -pitch_length/2
        goal_y = 0
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
    # Alter main defender's position.
    if main_def is not None:
        # Compute the main defenders possible positions triangle.
        x1 = player_positions_start.loc[
            player_positions_start.player_id == main_def, "x"].iloc[0]
        y1 = player_positions_start.loc[
            player_positions_start.player_id == main_def, "y"].iloc[0]
        x2 = player_positions_end.loc[
            player_positions_end.player_id == main_def, "x"].iloc[0]
        y2 = player_positions_end.loc[
            player_positions_end.player_id == main_def, "y"].iloc[0]
        t1, t2, t3 = main_def_possible_position_triangle(x1, y1, x2, y2)
        # Iterate over possible positions in the triangle and maximize position_score.
        pts = grid_points_in_triangle(t1, t2, t3)
        pos_score = np.zeros(len(pts))
        for i in range(len(pts)):
            pt = pts[i]
            x = pt[0]
            y = pt[1]
            pos_score[i] = position_score(
                runner_x_final, runner_y_final, 
                ball_x, ball_y,
                x, y, 
                attacking_side, 
                main_def, 
                defender_positions_end,
                pitch_length)
        max_pos_score_idx = pos_score.argmax()
        main_def_x = pts[max_pos_score_idx][0]
        main_def_y = pts[max_pos_score_idx][1]
        player_positions_end.loc[
            player_positions_end.player_id == main_def, "x"] = main_def_x
        player_positions_end.loc[
            player_positions_end.player_id == main_def, "y"] = main_def_y
            
    ## TODO: Other influenced defenders.
    # Decide who they are
    # Decide how to adjust
    
    return player_positions_end




