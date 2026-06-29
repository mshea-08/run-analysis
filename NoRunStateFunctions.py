#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 15:35:59 2025

@author: meredithshea
"""

import numpy as np
import math
from OrganizationFunctions import create_player_positions_2

# ================================================
# Helper Functions
# ================================================

def position_on_line(
        x1,y1,       # starting point
        x2,y2,       # ending point
        distance=2,  # distance new point is from start
    ):
    """
    Returns new point on the line (x1,y1) -- (x2,y2) a 
    specified distance away from (x1, y1).

    """
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L == 0:
        raise ValueError("Two points are identical.")
    
    ux, uy = dx / L, dy / L
    
    return x1 + distance * ux, y1 + distance * uy

def softmin(
        x,      # array of values
        T=6.0,  # soft-min parameter
    ):
    """
    Computes a "soft min" that depends on parameter T.
    
    small T -> value is close to the list min
    large T -> value is close to the list average
    
    """

    x = np.asarray(x, float)
    a = -x / T
    a -= a.max()
    w = np.exp(a)
    w /= w.sum()
    return float((w * x).sum())

def avg_near_seg(
        df,             # df with positions
        x_col,          # col name with x positions
        y_col,          # col name with y positions
        x0,             
        y1,            
        y2,             
        max_dist=20.0,  # dist away checked
    ):
    """
    Return a weighted average of y for points within 
    max_dist of the line segment (x0,y1) -- (x0,y2).
    
    Each point is weighted each by its distance to the 
    center of the segment (closer = more weight).

    If no points are within max_dist, returns the 
    midpoint. 
    
    """
    ymin, ymax = sorted([y1, y2])
    y_center = 0.5 * (ymin + ymax)

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    dx = x - x0
    
    within = (y >= ymin) & (y <= ymax) 
    below = y < ymin
    above = y > ymax

    dy_below = y - ymin
    dy_above = y - ymax

    dist = np.empty_like(x, dtype=float)
    # Compute distances from points to line. 
    # Dist for point with ymin < y < ymax
    dist[within] = np.abs(dx[within])
    # Dist for points with y < ymin
    dist[below]  = np.sqrt(dx[below]**2 + dy_below[below]**2)
    # Dist for points with y > ymax
    dist[above]  = np.sqrt(dx[above]**2 + dy_above[above]**2)    
    
    near_mask = dist <= max_dist

    if not near_mask.any():
        return y_center

    x_near = x[near_mask]
    y_near = y[near_mask]
    
    x_near = np.append(x_near, x0-1)
    y_near = np.append(y_near, (y1+y2)/2)

    d_center = np.sqrt((x_near - x0)**2 + (y_near - y_center)**2)
    # Weights for points 
    weights = np.maximum(max_dist - d_center, 0.0)

    if np.all(weights == 0):
        return y_near.mean()

    return np.average(y_near, weights=weights)

def is_ball_in_attacking_final_third(
        ball_x,           # ball x 
        pitch_length,     # sk convention
        attacking_side,   # "right_to_left" or "left_to_right"
    ):
    """
    Returns True if the ball is in the attacking team's 
    final third.
     
    """
    half = pitch_length / 2.0

    if attacking_side == "left_to_right":
        return ball_x >= (half - pitch_length / 3.0)
    else:
        return ball_x <= (-half + pitch_length / 3.0)
    
def offside_check(
        attacking_side,   # "right_to_left" or "left_to_right"
        def_pos,          # df of defender positions
        runner_x,         # runner x 
        buffer_dist,      # buffer dist 
    ):
    """
    Moves runner to onside position.
    """
    if attacking_side == "left_to_right":
        # offside line 
        offside = def_pos["x"].nlargest(2).iloc[-1]
        if offside < 0:
            offside = 0.0
        if runner_x > offside:
            runner_x = offside - buffer_dist 
        
        return runner_x
    
    else: # right_to_left case
        # offside line
        offside = def_pos["x"].nsmallest(2).iloc[-1]
        if offside > 0:
            offside = 0.0
        if runner_x < offside:
            runner_x = offside + buffer_dist 
        
        return runner_x
        
def offside_line(
        attacking_side,    # direction of play
        def_pos,           # df of defender positions
    ):
    """
    Computes offside line.
    
    """
    if attacking_side == "left_to_right":
        offside = def_pos["x"].nlargest(2).iloc[-1]
        if offside > 0:
            return offside
        else:
            return 0
    
    else:
        offside = def_pos["x"].nsmallest(2).iloc[-1]
        if offside < 0:
            return offside
        else:
            return 0
        
def last_def_check(
        attacking_side,
        att_pos,            # df of attacker positions
        runner_x,
        buffer_dist,
    ):
    """
    Moves runner to slightly above defensive line.

    """
    if attacking_side == "left_to_right":
        line_height = att_pos["x"].nsmallest(2).iloc[-1]
        if runner_x < line_height:
            runner_x = line_height + buffer_dist
        return runner_x
    else:
        line_height = att_pos["x"].nlargest(2).iloc[-1]
        if runner_x > line_height:
            runner_x = line_height - buffer_dist
        return runner_x
        
def last_def_line(
        attacking_side,
        att_pos,
    ):
    """
    Compute def line height.

    """
    if attacking_side == "left_to_right":
        line_height = att_pos["x"].nsmallest(2).iloc[-1]
        return line_height
    else:
        line_height = att_pos["x"].nlargest(2).iloc[-1]
        return line_height

def ellipse_points(
        center, 
        a,                  # horizontal semi-axis
        b,                  # vertical semi-axis
        point_dist = 0.5,   # grid spacing
    ):
    """
    Returns (N,2) array of points evenly sampled 
    within the ellipse. 
    
    """
    cx, cy = center

    xs = np.arange(cx - a, cx + a + point_dist, point_dist)
    ys = np.arange(cy - b, cy + b + point_dist, point_dist)
    xx, yy = np.meshgrid(xs, ys)

    mask = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0

    points = np.column_stack([xx[mask], yy[mask]])
    
    return points

def max_min_dist(
        x, 
        y, 
        candidates,         # potential positions (N,2)
        fixed_points,       # points to measure dist to (M,2)
        threshold=10.0
    ):
    """
    Picks the candidate point whose min dist to any 
    fixed point is more than the threshold and that is
    closest to the original location (x,y). 
    
    If no candidate points meet the threshold, function
    returns the maximizer (of min dist).

    """
    # Compute distance from candidates to fixed points
    diff = candidates[:, np.newaxis, :] - fixed_points[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    # Compute min distance for each candidate
    min_dists = dists.min(axis=1)

    valid_mask = min_dists >= threshold
    # No candidate meets the threshold
    if not valid_mask.any():
        return candidates[min_dists.argmax()]
    
    valid_candidates = candidates[valid_mask]

    # Pick the candidate closest to the original location    
    origin = np.array([x, y])
    origin_dists = np.sqrt(((valid_candidates - origin) ** 2).sum(axis=1))
    best_idx = origin_dists.argmin()

    return valid_candidates[best_idx]

def is_marking(
        x,
        y,
        att_pos,
        attacking_side,
    ):
    """
    Returns True if the defender at position (x,y) is 
    determined to be marking an attacker, else returns False. 
    
    """
    if attacking_side == "left_to_right":
        marks = att_pos[
            (att_pos["x"] < x) &
            (att_pos["x"] > x-6) &
            (att_pos["y"] < y+3) &
            (att_pos["y"] > y-3)]
    else:
        marks = att_pos[
            (att_pos["x"] > x) &
            (att_pos["x"] < x+6) &
            (att_pos["y"] < y+3) &
            (att_pos["y"] > y-3)]
    
    if len(marks) > 0:
        return True
    else:
        return False


def intersect_disk_with_reachable(
        tactical_pts,       # candidate def points    
        def_start_x,        # def start position
        def_start_y,
        run_time,           # duration of run
        def_vx=0.0,
        def_vy=0.0,
        def_speed=5.0,      # max speed
        acceleration=3.0,
        ):
    """
    Filters tactical candidate points to only those physically 
    reachable by the defender.

    If no points satisfy physical condition, function 
    returns the point in tactical_pts closest to the 
    defender's start position.
    """
    # Dist from tactical points to def starting location
    dx = tactical_pts[:, 0] - def_start_x
    dy = tactical_pts[:, 1] - def_start_y
    dists = np.sqrt(dx**2 + dy**2)

    
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = np.where(dists > 0, dx / dists, 0.0)
        uy = np.where(dists > 0, dy / dists, 0.0)

    # Velocity projected onto the direction of the run
    v_proj = def_vx * ux + def_vy * uy
    v_proj = np.clip(v_proj, -def_speed, def_speed)

    accel_time = np.maximum(0.0, (def_speed - v_proj) / acceleration)
    accel_time = np.minimum(accel_time, run_time)

    dist_accel = v_proj * accel_time + 0.5 * acceleration * accel_time**2
    dist_cruise = def_speed * np.maximum(0.0, run_time - accel_time)

    max_reach = dist_accel + dist_cruise

    mask = dists <= max_reach
    if mask.any():
        return tactical_pts[mask]
    return tactical_pts[[dists.argmin()]]

# ================================
# Main Functions
# ================================

def move_runner(
        runner_id,
        att_pos_end,
        def_pos_end,
        att_pos_start,
        def_pos_start,
        attacking_side,
        pitch_length,
        pitch_width,
        run_time,
        jog_speed = 2.0,
        close_tm_dist = 7.0,        # threshold for teammate being too close to runner
        max_vert_dist = 7.0,        # max vertical change in runner position tuning
        max_horizontal_dist = 3.0,  # max horizontal change in runner position tuning
        close_dist = 10.0,          # threshold for runner begin too close to teammates
        ):
    """
    Computes the no-run-state position for the runner and,
    if applicable, a close teammate who fills the space 
    vacated by the runner.

    A teammate is only moved if:
      (1) they are within close_tm_dist of the runner's original 
          position, and
      (2) the runner's new no-run position is at most close_tm_dist 
          away from the original position.

    Returns: runner_x_new, runner_y_new, cls_tm_id, cls_tm_x, cls_tm_y
    """

    # ========================================
    # Step 1: Runner horizontal displacement
    # ========================================

    runner_x = float(att_pos_start.loc[
        att_pos_start.player_id == runner_id, "x"].iloc[0])
    runner_y = float(att_pos_start.loc[
        att_pos_start.player_id == runner_id, "y"].iloc[0])

    att_pos_start = att_pos_start.loc[
        att_pos_start.player_id != runner_id].copy()
    att_pos_end = att_pos_end.loc[
        att_pos_end.player_id != runner_id].copy()

    # Compute average vertical change for teammates 
    vert_change = att_pos_end["x"].values - att_pos_start["x"].values
    vert_change_avg = vert_change.mean()

    # Intermediate runner position
    runner_x_new = runner_x + vert_change_avg
    runner_y_new = runner_y

    # Check runner is not the last defender or offside after change
    runner_x_new = last_def_check(
        attacking_side, 
        att_pos_end, 
        runner_x_new, 
        max_horizontal_dist
    )
    runner_x_new = offside_check(
        attacking_side, 
        def_pos_end, 
        runner_x_new, 
        max_horizontal_dist
    )

    pitch_up   =  pitch_width / 2
    pitch_down = -pitch_width / 2

    # =============================================
    # Step 2: Close teammate check
    # =============================================

    # Dist from teammates end position to runners starting position   
    att_pos_end["dist_to_runner_start"] = np.sqrt(
        (att_pos_end["x"] - runner_x) ** 2
        + (att_pos_end["y"] - runner_y) ** 2)

    min_dist_to_start = att_pos_end["dist_to_runner_start"].min()

    # Dist to runner new position
    runner_new_to_start = math.hypot(
        runner_x_new - runner_x,
        runner_y_new - runner_y,
    )

    # Conditions for a "close teammate":
    # (1) teammate ends close to runners starting spot
    # (2) the runners new position is also close to starting spot
    
    # Logic: in this instance the teammate is running into 
    # space created by the runner. If the runner no longer 
    # generates this space, then the teammate needs to find
    # new space to occupy. 
    
    if (min_dist_to_start <= close_tm_dist) and (runner_new_to_start <= close_tm_dist):

        # Identify close teammate
        cls_tm_idx = att_pos_end["dist_to_runner_start"].idxmin()
        cls_tm_id  = att_pos_end.loc[cls_tm_idx, "player_id"]

        # positions of everyone but cls_tm        
        cls_tm_delete = att_pos_end.drop(cls_tm_idx)
        tm_pos = cls_tm_delete[["x", "y"]].to_numpy()
        tm_pos = np.append(tm_pos, [[runner_x_new, runner_y_new]], axis=0)

        # Intermediate cls_tm position
        cls_tm_x = float(att_pos_start.loc[cls_tm_idx, "x"]) + vert_change_avg
        cls_tm_y = float(att_pos_start.loc[cls_tm_idx, "y"])

        # Create list of candidate points for cls_tm
        r = jog_speed * run_time
        disk_pts = ellipse_points((cls_tm_x, cls_tm_y), r, r)

        offside = offside_line(attacking_side, def_pos_end)
        line_height = last_def_line(attacking_side, att_pos_end)

        # Make sure teammate is not offsides or last def  
        if attacking_side == "left_to_right":
            cand_pts = disk_pts[
                (disk_pts[:, 0] <= offside)
                & (disk_pts[:, 0] >= line_height)]
        else:
            cand_pts = disk_pts[
                (disk_pts[:, 0] >= offside)
                & (disk_pts[:, 0] <= line_height)]

        # Make sure teammate is on the pitch
        cand_pts = cand_pts[
            (cand_pts[:, 1] <= pitch_up)
            & (cand_pts[:, 1] >= pitch_down)]

        if len(cand_pts) == 0:
            cand_pts = np.array([[cls_tm_x, cls_tm_y]])
        
        # Pick a suitable new position for the cls_tm
        diff = cand_pts[:, np.newaxis, :] - tm_pos[np.newaxis, :, :]
        min_dists_to_tm = np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)
        valid_mask = min_dists_to_tm >= close_tm_dist

        origin = np.array([cls_tm_x, cls_tm_y])
        origin_dists = np.sqrt(((cand_pts - origin) ** 2).sum(axis=1))

        if valid_mask.any():
            origin_dists_valid = np.where(valid_mask, origin_dists, np.inf)
            best_idx = int(origin_dists_valid.argmin())
            new_x, new_y = cand_pts[best_idx]
        else:
            new_x, new_y = max_min_dist(cls_tm_x, cls_tm_y, cand_pts, tm_pos)

        cls_tm_x = new_x
        cls_tm_y = new_y

    else: # no cls_tm
        cls_tm_id = None
        cls_tm_x  = None
        cls_tm_y  = None

    # =============================================
    # Step 3: Runner vertical fine-tuning
    # =============================================

    # Time for downfield adjustment
    t_h = abs(vert_change_avg) / jog_speed

    # Allowable amount of fine-tuning adjustment
    max_vert_shift  = min(
        max(jog_speed * (run_time - t_h), 1.0), max_vert_dist)
    max_horizontal_shift = min(
        max(jog_speed * (run_time - t_h), 1.0), max_horizontal_dist)

    if cls_tm_id is not None:
        tm_end = att_pos_end.copy()
        tm_end.loc[tm_end.player_id == cls_tm_id, "x"] = cls_tm_x
        tm_end.loc[tm_end.player_id == cls_tm_id, "y"] = cls_tm_y
    else:
        tm_end = att_pos_end

    tm_end["dist_to_runner_new"] = np.sqrt(
        (tm_end["x"] - runner_x_new) ** 2
        + (tm_end["y"] - runner_y_new) ** 2)

    min_dist_to_new = tm_end["dist_to_runner_new"].min()

    # Fine tuning
    if min_dist_to_new <= close_dist:
        tm_pos = tm_end[["x", "y"]].to_numpy()

        ellipse_pts = ellipse_points(
            (runner_x_new, runner_y_new),
            max_horizontal_shift,
            max_vert_shift,
        )
        cand_pts = ellipse_pts[
            (ellipse_pts[:, 1] <= pitch_up)
            & (ellipse_pts[:, 1] >= pitch_down)
        ]

        new_x, new_y = max_min_dist(
            runner_x_new, runner_y_new, cand_pts, tm_pos)

        runner_x_new = new_x
        runner_y_new = new_y

    return runner_x_new, runner_y_new, cls_tm_id, cls_tm_x, cls_tm_y


def position_score(
        runner_x,               
        runner_y,               
        ball_x,                 
        ball_y,                 
        candidates,         # (N,2) array of posible positions
        attacking_side,         
        main_def_id,            
        def_pos_end,            
        pitch_length,           
        def_behind=3,       # safe number of defenders behind
    ):
    """
    Returns the position with the "best" defensive positioning
    score. 
    
    """
    cand_x = candidates[:, 0]
    cand_y = candidates[:, 1]

    goal_y = 0
    goal_x = pitch_length / 2 if attacking_side == "left_to_right" else -pitch_length / 2

    # ===========================
    # Defensive positioning score
    # ===========================
    dist_runner_to_goal = math.hypot(
        runner_x - goal_x, 
        runner_y - goal_y
    )
    
    # Optimal position of defender
    # Dist away depends on how far from goal the players are 
    optimal_dist = min(5, max(2, 0.05 * dist_runner_to_goal))
    optimal_x, optimal_y = position_on_line(
        runner_x, 
        runner_y, 
        goal_x, 
        goal_y, 
        distance=optimal_dist
    )
    
    # Optimal score
    dist_to_optimal = np.hypot(cand_x - optimal_x, cand_y - optimal_y)
    optimal_score = 1.0 / (1.0 + dist_to_optimal)

    # =========================
    # Passing lane score
    # =========================

    lane_dist = min(5, 0.05 * dist_runner_to_goal)
    if (runner_x == ball_x) and (runner_y == ball_y):
        passing_lane_score = np.zeros(len(cand_x))
    else:
        pass_lane_x, pass_lane_y = position_on_line(
            runner_x, 
            runner_y, 
            ball_x, 
            ball_y, 
            distance=lane_dist
        )
        
        dist_to_lane = np.hypot(cand_x - pass_lane_x, cand_y - pass_lane_y)
        passing_lane_score = 1.0 / (1.0 + dist_to_lane)

    # =========================
    # Shape score
    # =========================
    
    # Define general "def" position
    defs = ["RCB", "LCB", "RB", "LB", "CB"]
    main_def_role = def_pos_end.loc[
        def_pos_end.player_id == main_def_id, "position"].iloc[0]
    role = "def" if main_def_role in defs else "not_def"

    other_def_end = def_pos_end.loc[def_pos_end.player_id != main_def_id]
    other_xy = other_def_end[["x", "y"]].to_numpy()

    # If def, part of shape score is maintaining def line height
    if role == "def":
        other_def_line = other_def_end.loc[other_def_end.position.isin(defs)]
        def_line_height = other_def_line["x"].mean()
        line_error = (def_line_height - cand_x) ** 2
    else:
        line_error = np.zeros(len(cand_x))

    # Other part of shape score is not being too close to teammates
    if other_xy.shape[0] > 0:
        dx_om = cand_x[:, None] - other_xy[None, :, 0]
        dy_om = cand_y[:, None] - other_xy[None, :, 1]
        dists_om = np.sqrt(dx_om ** 2 + dy_om ** 2)

        k = min(3, other_xy.shape[0])
        closest = np.sort(dists_om, axis=1)[:, :k]
        close_error = np.exp(-closest).sum(axis=1)
    else:
        close_error = np.zeros(len(cand_x))

    # Total shape score 
    shape_score = 1.0 / (1.0 + line_error + close_error)

    # ===========================
    # Combine component scores 
    # ===========================
    
    if attacking_side == "right_to_left":
        numbers_behind = len(other_def_end[other_def_end["x"] < runner_x])
    else:
        numbers_behind = len(other_def_end[other_def_end["x"] > runner_x])

    if numbers_behind <= def_behind:
        # no defensive help, be conservative
        w_near = np.array([0.9, 0.0, 0.1])
        w_far  = np.array([0.7, 0.0, 0.3])
    elif role == "def":
        # defenders should be more focused on shape
        w_near = np.array([0.5, 0.0, 0.5])
        w_far  = np.array([0.2, 0.1, 0.7])
    else:
        w_near = np.array([0.6, 0.2, 0.2])
        w_far  = np.array([0.3, 0.5, 0.2])

    ball_to_goal   = math.hypot(ball_x - goal_x, ball_y - goal_y)
    runner_to_goal = math.hypot(runner_x - goal_x, runner_y - goal_y)
    dist_goal = softmin([ball_to_goal, runner_to_goal])

    far_mult = 3
    D0 = 20
    g = max(0.0, min(1.0, 1.0 - dist_goal / (far_mult * D0)))
    w = g * w_near + (1 - g) * w_far

    # Total score 
    scores = (
        w[0] * optimal_score +
        w[1] * passing_lane_score +
        w[2] * shape_score
    )
    
    best_idx = int(np.argmax(scores))
    
    return float(candidates[best_idx, 0]), float(candidates[best_idx, 1])

def def_id(
        attacker_id,
        def_pos,
        att_pos,
        attacking_side,
        pitch_length,
        ball_x = 0.0,
        ball_y = 0.0,
        def_threshold = 8.0,    # how close defender needs to be
        occupied_margin = 0.25, # proportion for determining whether other responsibilities take priorirty 
    ):
    """
    Determines the ID of the player defending attacker_id. If 
    no player is determined to be the defender, returns None. 
    
    Filter to defenders that are within def_threshold of the 
    optimal defensive position.
    
    Check that the defenders don't have anyone more 
    important to cover. We make two considerations:
        - if there is another attacker who is >25% closer
        - if there is an attacker who is closer (but less than
          25% so) and in a more dangerous position. 
        
    Pick the closest one under these conditions.
    
    """
    def_pos = def_pos.copy()
    att_pos = att_pos.copy()

    # Ignore GKs
    def_pos = def_pos[def_pos["position"] != "GK"]
    
    att_x = att_pos.loc[att_pos["player_id"]==attacker_id,"x"].iloc[0]
    att_y = att_pos.loc[att_pos["player_id"]==attacker_id,"y"].iloc[0]
    
    # Dist from all defenders to attacker
    def_pos["dist_to_runner"] = (
        np.sqrt((def_pos["x"] - att_x)**2 + 
                (def_pos["y"] - att_y)**2))
    
    if attacking_side == "left_to_right":
        goal_x = pitch_length / 2
        goal_y = 0
        def_goal_x = -pitch_length / 2
    else:
        goal_x = -pitch_length / 2
        goal_y = 0
        def_goal_x = pitch_length / 2

    # Optimal def position
    optimal_def_x, optimal_def_y = position_on_line(
        att_x, 
        att_y, 
        goal_x, 
        goal_y, 
        distance=2
    )

    def_pos = def_pos.reset_index(drop=True)
    # Dist from def to optimal defensive positioning
    dx = def_pos["x"] - optimal_def_x
    dy = def_pos["y"] - optimal_def_y
    d  = np.hypot(dx, dy)
    # Only include defenders close enough to attacker
    condition = def_pos["dist_to_runner"] <= def_threshold
    def_pos["dist_to_optimal"] = np.where(condition, d, np.nan)

    att_xy  = att_pos[["x", "y"]].to_numpy()
    def_xy  = def_pos[["x", "y"]].to_numpy()

    if att_xy.shape[0] > 0:
        # Dist from def to all attackers
        dists_da = np.sqrt(
            (def_xy[:, 0][:, None] - att_xy[None, :, 0]) ** 2 +
            (def_xy[:, 1][:, None] - att_xy[None, :, 1]) ** 2
        )
        nearest_att_idx = dists_da.argmin(axis=1)
        nearest_att_dist = dists_da.min(axis=1)
        dist_to_runner = def_pos["dist_to_runner"].to_numpy()
        
        # Check that defenders don't have someone more important to defend
        for i in range(len(def_pos)):
            # Def is not close to attacker
            if np.isnan(def_pos.loc[i, "dist_to_optimal"]):
                continue

            near_dist = nearest_att_dist[i]
            run_dist = dist_to_runner[i]

            # No other attackers are closer
            if near_dist >= run_dist:
                continue

            ratio = (run_dist - near_dist) / run_dist

            if ratio > occupied_margin:
                # Another attacker is significantly closer
                def_pos.loc[i, "dist_to_optimal"] = np.nan
            else:
                # Another attacker is only marginally closer 
                # Determine which attacker is the bigger threat 
                def_x_pos = def_xy[i, 0]
                final_third = pitch_length / 3
                in_final_third = (
                    def_x_pos <= def_goal_x + final_third
                    if attacking_side == "left_to_right"
                    else def_x_pos >= def_goal_x - final_third
                )

                if in_final_third:
                    dist_near_to_goal   = math.hypot(
                        att_xy[nearest_att_idx[i], 0] - def_goal_x,
                        att_xy[nearest_att_idx[i], 1])
                    dist_runner_to_goal = math.hypot(att_x - def_goal_x, att_y)
                    runner_more_dangerous = dist_runner_to_goal < dist_near_to_goal
                else:
                    dist_near_to_ball   = math.hypot(
                        att_xy[nearest_att_idx[i], 0] - ball_x,
                        att_xy[nearest_att_idx[i], 1] - ball_y)
                    dist_runner_to_ball = math.hypot(att_x - ball_x, att_y - ball_y)
                    runner_more_dangerous = dist_runner_to_ball < dist_near_to_ball

                if not runner_more_dangerous:
                    def_pos.loc[i, "dist_to_optimal"] = np.nan

    if def_pos["dist_to_optimal"].isna().all():
        main_def = None
    else:
        main_def = def_pos.loc[
            def_pos["dist_to_optimal"].idxmin(), "player_id"]

    return main_def
    
def no_run_state_ball(
        run,   # series containing advance run infomation (sk)
    ):
    """
    Always gives the ball to the designated ball carrier.
    
    """
    ball_carrier_id = run["player_in_possession_id"]
    match_data = run["match_data"]
    frame_end_data = run["frame_end_data"]
    player_positions = create_player_positions_2(
        match_data, frame_end_data)

    x = player_positions.loc[
        player_positions["player_id"] == ball_carrier_id]["x"].iloc[0]
    y = player_positions.loc[
        player_positions["player_id"] == ball_carrier_id]["y"].iloc[0]

    return x, y

def closest_defender_to_point(
        defender_pos,
        x, 
        y,               
    ):
    """
    Return the player_id of the defender closest to (x, y),
    or None if defender_positions is empty.
    """
    if defender_pos is None or defender_pos.empty:
        return None

    dists = np.hypot(
        defender_pos["x"].to_numpy() - x,
        defender_pos["y"].to_numpy() - y
    )
    idx = dists.argmin()
    return defender_pos["player_id"].iloc[idx]

def classify_attackers_marking(
        att_end,
        def_end,
        r_mark=7.0,     # max dist for marking
        goal_x=None,
        ):
    """
    Split attackers into unmarked and marked based on dist
    to the nearest goal-side defender in defenders_end.

    A defender only counts as a marker if:
      (1) they lie between the attacker and the goal (goal-side), and
      (2) the attacker is their nearest attacker, i.e. they 
          are not already occupied marking someone closer.

    Returns unmarked_df, marked_df.
    """
    if att_end.empty or def_end.empty or goal_x is None:
        return np.empty((0, 2)), np.empty((0, 2))

    att_xy = att_end[["x", "y"]].to_numpy()
    def_xy = def_end[["x", "y"]].to_numpy()

    # Dist from attackers to defenders
    # Shape = (A,D) 
    dx_ad = att_xy[:, 0][:, None] - def_xy[:, 0][None, :]
    dy_ad = att_xy[:, 1][:, None] - def_xy[:, 1][None, :]
    dist_ad = np.sqrt(dx_ad**2 + dy_ad**2)

    att_x = att_xy[:, 0][:, None]
    def_x = def_xy[:, 0][None, :]
    
    # Masks 
    goal_left   = (goal_x < att_x)
    valid_left  = (def_x >= goal_x) & (def_x <= att_x)
    valid_right = (def_x <= goal_x) & (def_x >= att_x)
    goal_side_mask = np.where(goal_left, valid_left, valid_right)

    nearest_att_per_def = dist_ad.min(axis=0)

    eps = 1e-8
    is_nearest = np.abs(dist_ad - nearest_att_per_def[None, :]) < eps
    
    # Mask identifying closest attacker where def is goal side
    valid_marker = goal_side_mask & is_nearest

    dist_valid = np.where(valid_marker, dist_ad, np.inf)
    min_dists = dist_valid.min(axis=1)

    unmarked_mask = min_dists > r_mark

    unmarked_df = att_end.loc[unmarked_mask, ["player_id", "x", "y"]].reset_index(drop=True)
    marked_df   = att_end.loc[~unmarked_mask, ["player_id", "x", "y"]].reset_index(drop=True)

    return unmarked_df, marked_df


def filter_points_near_vertical_segment(
        points_xy,  # (N,2) array of points
        x0, 
        y1, 
        y2, 
        max_dist,
        ):
    """
    Keep only points within max_dist of the vertical segment
    (x0,y1) -- (x0,y2).
    
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
    dist[within] = np.abs(dx[within])
    dy_below = y[below] - ymin
    dist[below] = np.sqrt(dx[below]**2 + dy_below**2)
    dy_above = y[above] - ymax
    dist[above] = np.sqrt(dx[above]**2 + dy_above**2)

    mask = dist <= max_dist
    return points_xy[mask]


def check_unmarked(
        x,
        y,
        r,
        att_pos,
        def_pos,
        def_id,
        goal_x,
    ):
    """
    Returns df of unmarked attackers within a radius r from
    the point (x,y) or None.
    
    Uses function classify_attackers_marking to identify 
    unmarked attackers. 
    
    """
    att_pos = att_pos.copy()
    def_pos = def_pos.copy()
    
    # Remove the defender begin placed
    other_def_pos = def_pos[def_pos["player_id"] != def_id]
    
    # Compute the unmarked attackers
    unmarked_attackers_df, marked_attackers_df = classify_attackers_marking(
        att_pos, 
        other_def_pos,  
        goal_x=goal_x
    )
    
    if len(unmarked_attackers_df) == 0:
        return None
    
    unmarked_attackers_df["dist_to_def"] = np.sqrt(
        (att_pos["x"] - x)**2
        +(att_pos["y"] - y)**2)
    
    # Determine attackers within zone 
    nearby_unmarked_att_pos = unmarked_attackers_df[unmarked_attackers_df["dist_to_def"] < r]
    nearby_unmarked_att_pos = nearby_unmarked_att_pos[["player_id","x","y"]]
    
    
    if len(nearby_unmarked_att_pos) > 0:
        return nearby_unmarked_att_pos
    else:
        return None
    
def ball_carrier_marker(
        carrier_id,
        def_pos,
        att_pos,
        goal_x,
        r_mark=8.0,
        r_close=3.0,
    ):
    """
    Returns the player_id of the defender marking the ball carrier,
    or None if no defender qualifies.
    
    """
    # Carrier location
    carrier_x = att_pos.loc[att_pos.player_id == carrier_id, "x"].iloc[0]
    carrier_y = att_pos.loc[att_pos.player_id == carrier_id, "y"].iloc[0]
    
    # Dist to ball carrier
    def_xy = def_pos[["x", "y"]].to_numpy()
    dists = np.sqrt((def_xy[:, 0] - carrier_x) ** 2 +
                    (def_xy[:, 1] - carrier_y) ** 2)
    
    # Defenders need to be goal side or "close enough" determined by r_close
    if goal_x < carrier_x:
        goal_side = def_xy[:, 0] <= carrier_x
    else:
        goal_side = def_xy[:, 0] >= carrier_x
    
    within_range = dists <= r_mark
    close_enough = dists <= r_close
    valid = within_range & (goal_side | close_enough)
    
    if not valid.any():
        return None
    dists_valid = np.where(valid, dists, np.inf)
    best_idx = int(np.argmin(dists_valid))
    
    return def_pos.iloc[best_idx]["player_id"]


def fallback_main(
        end_x,
        end_y,
        y1, 
        y2,
        def_new,
        att_new,
        def_id,
        num_candidates=31,
        min_spacing=7.0,
        sigma=10.0,
        tm_sigma=5.0,
        central_weight=0.1,
    ):
    """
    Fallback positioning for main defender. New position is 
    (end_x,y) where y1 < y < y2. 
    
    """
    y_candidates = np.linspace(y1, y2, num=num_candidates)
    def_new = def_new[def_new["player_id"] != def_id]

    # If no attackers, return end position
    if att_new is None or att_new.empty:
        return end_x, end_y

    att_xy = att_new[["x", "y"]].to_numpy()
    
    # ====================================
    # Nearby attackers score
    # ====================================

    # Compute how far attackers are from line segment
    dx = np.abs(att_xy[:, 0] - end_x)
    y_lo, y_hi = min(y1, y2), max(y1, y2)
    dy = np.maximum(0, np.maximum(y_lo - att_xy[:, 1], att_xy[:, 1] - y_hi))
    dist_to_line = np.sqrt(dx ** 2 + dy ** 2)
    # Turn into weights
    att_weights = np.exp(-dist_to_line / sigma)

    # Use weighted average of attacker positions
    if att_weights.sum() == 0:
        target_y = end_y
    else:
        target_y = float(np.dot(att_weights, att_xy[:, 1]) / att_weights.sum())
        
    # Score possible candidate locations 
    # Positions closer to attackers close to the line will get larger scores
    att_score = -np.abs(y_candidates - target_y)
    att_score = att_score - att_score.min()
    att_range = att_score.max()
    if att_range > 0:
        att_score = att_score / att_range

    # ====================================
    # Teammate spacing score 
    # ====================================
    # Positions are punished for being close to teammates
    
    if def_new is not None and not def_new.empty:
        teammates_xy = def_new[["x", "y"]].to_numpy()
        dx_t   = (end_x - teammates_xy[:, 0]) ** 2
        dy_all = y_candidates[:, None] - teammates_xy[None, :, 1]
        # Distances from points to teammates
        d_all  = np.sqrt(dx_t[None, :] + dy_all ** 2)

        crowding = np.exp(-d_all / tm_sigma).sum(axis=1)
        # Bad spacing is punished
        spacing_score = -crowding
        spacing_score = spacing_score - spacing_score.min()
        sp_range = spacing_score.max()
        
        if sp_range > 0:
            spacing_score = spacing_score / sp_range

        min_dists  = d_all.min(axis=1)
        spacing_ok = min_dists >= min_spacing
    
    else:
        spacing_score = np.ones(num_candidates)
        spacing_ok    = np.ones(num_candidates, dtype=bool)

    # ==============================
    # Centrality score 
    # ==============================
    # Candidate points closer to the center of the field are favored
    central_score = -np.abs(y_candidates)
    central_score = central_score - central_score.min()
    c_range = central_score.max()
    if c_range > 0:
        central_score = central_score / c_range

    # ==============================
    # Combined score 
    # ==============================
    w_att = 1.0 - central_weight
    w_spacing = 0.3
    w_central = central_weight
    scores = w_att * att_score + w_spacing * spacing_score + w_central * central_score

    if spacing_ok.any():
        scores_strict = np.where(spacing_ok, scores, -np.inf)
        best_idx = int(np.argmax(scores_strict))
    else:
        best_idx = int(np.argmax(scores))

    return end_x, float(y_candidates[best_idx])

# !!!!! here!!
def fallback_second(
        end_x,
        end_y,
        r_x,                # horizontal semi-axis         
        r_y,                # vertical semi-axis
        def_new,            
        att_new,
        def_id,           
        goal_x,           
        r_mark=7.0,
        pitch_width=68.0,
        point_dist=0.5,
        ):
    """
    Zonal fallback for the secondary defender with no direct marking
    assignment. Searches an ellipse of reachable positions (r_x, r_y)
    centered on the no-run position and picks the candidate closest to
    the danger- and marking-adjusted weighted centroid of all attackers:
      - danger weight  = 1 / dist_to_goal  (closer to goal = higher)
      - marking weight = dist to nearest goal-side defender
                         (well marked = low weight, free = high weight)
    Hard minimum spacing from teammates enforced where possible.
    """
    other_defs = def_new[def_new["player_id"] != def_id].copy()

    if att_new is None or att_new.empty:
        return end_x, float(np.clip(end_y, -pitch_width / 2, pitch_width / 2))

    att_xy = att_new[["x", "y"]].to_numpy()
    att_x  = att_xy[:, 0]
    att_y  = att_xy[:, 1]

    # --- Danger weight: 1 / distance to goal ---
    dist_to_goal = np.sqrt((att_x - goal_x) ** 2 + att_y ** 2)
    dist_to_goal = np.maximum(dist_to_goal, 0.1)
    danger_w = 1.0 / dist_to_goal                                  # (A,)

    # --- Marking weight: dist to nearest goal-side defender ---
    if not other_defs.empty:
        def_xy     = other_defs[["x", "y"]].to_numpy()            # (D, 2)
        att_x_col  = att_x[:, None]                                # (A, 1)
        def_x_row  = def_xy[None, :, 0]                           # (1, D)
        goal_left  = goal_x < att_x_col
        goal_side_mask = np.where(
            goal_left,
            (def_x_row >= goal_x) & (def_x_row <= att_x_col),
            (def_x_row <= goal_x) & (def_x_row >= att_x_col),
        )                                                           # (A, D)

        dx_ad   = att_x[:, None] - def_xy[None, :, 0]
        dy_ad   = att_y[:, None] - def_xy[None, :, 1]
        dist_ad = np.sqrt(dx_ad ** 2 + dy_ad ** 2)                # (A, D)

        dist_goalside     = np.where(goal_side_mask, dist_ad, np.inf)
        min_goalside_dist = dist_goalside.min(axis=1)              # (A,)
        min_goalside_dist = np.where(
            np.isinf(min_goalside_dist), r_mark, min_goalside_dist
        )
        marking_w = min_goalside_dist
    else:
        marking_w = np.ones(len(att_xy)) * r_mark

    # --- Combined weight → danger-marking weighted centroid ---
    combined_w = danger_w * marking_w
    w_sum = combined_w.sum()
    if w_sum == 0:
        combined_w = np.ones(len(att_xy)) / len(att_xy)
    else:
        combined_w /= w_sum

    target_x = float(np.dot(combined_w, att_x))
    target_y = float(np.dot(combined_w, att_y))

    # --- Ellipse candidate points ---
    candidates = ellipse_points(
        (end_x, end_y), max(r_x, 0.5), max(r_y, 0.5), point_dist=point_dist
    )
    half_w = pitch_width / 2.0
    candidates = candidates[
        (candidates[:, 1] >= -half_w) &
        (candidates[:, 1] <=  half_w)
    ]
    if len(candidates) == 0:
        return end_x, float(np.clip(end_y, -half_w, half_w))

    cand_x = candidates[:, 0]
    cand_y = candidates[:, 1]
    dist_to_target = np.sqrt(
        (cand_x - target_x) ** 2 + (cand_y - target_y) ** 2
    )

    # --- Teammate spacing ---
    if not other_defs.empty:
        tm_xy  = other_defs[["x", "y"]].to_numpy()
        dx_t   = cand_x[:, None] - tm_xy[None, :, 0]
        dy_t   = cand_y[:, None] - tm_xy[None, :, 1]
        d_tm   = np.sqrt(dx_t ** 2 + dy_t ** 2)
        min_tm = d_tm.min(axis=1)
        spacing_ok = min_tm >= 7.0
    else:
        spacing_ok = np.ones(len(candidates), dtype=bool)

    # Pass 1: hard spacing enforced
    if spacing_ok.any():
        dist_strict = np.where(spacing_ok, dist_to_target, np.inf)
        best_idx = int(np.argmin(dist_strict))
    else:
        # Pass 2: relax spacing
        best_idx = int(np.argmin(dist_to_target))

    return float(candidates[best_idx, 0]), float(candidates[best_idx, 1])


def no_run_state(
        match_data, 
        frame_start_data,
        frame_end_data, 
        run,
        jog_speed = 2.0,
        run_speed = 4.0,
        def_speed = 5.0,
        max_r = 12.0
        ):
    """
    Returns player positions dataframe with all updated positions.
    """
    # STEP 0: Preliminary definitions. 
    
    frame_start = run["frame_start"]
    frame_end = run["frame_end"]
    att_team_id = run["team_id"]
    runner_id = run["player_id"]
    attacking_side = run["attacking_side"]
    carrier_id = run["player_in_possession_id"]
    
    pitch_length = match_data["pitch_length"]
    pitch_width = match_data["pitch_width"]
    
    half_L = pitch_length / 2.0
    if attacking_side == "left_to_right":
        goal_x = half_L
    else:
        goal_x = -half_L

    pos_start = create_player_positions_2(match_data, frame_start_data)
    pos_end   = create_player_positions_2(match_data, frame_end_data)
    
    team_ids = pos_start.team_id.unique()
    def_team_id = (team_ids[1] if team_ids[0] == att_team_id else team_ids[0])
    
    att_pos_start = pos_start[pos_start["team_id"] == att_team_id]
    att_pos_end   = pos_end[pos_end["team_id"] == att_team_id]
    
    runner_x_end = pos_end.loc[pos_end.player_id == runner_id, "x"].iloc[0]
    runner_y_end = pos_end.loc[pos_end.player_id == runner_id, "y"].iloc[0]
    
    def_pos_start = pos_start[pos_start.team_id == def_team_id]
    def_pos_end   = pos_end[pos_end.team_id == def_team_id]
    
    run_time = (frame_end - frame_start) * 0.1  # in seconds
    
    ball_x, ball_y = no_run_state_ball(run)
    
    pos_new = pos_end.copy()
    
    # STEP 1: Runner Positioning.
    runner_x_new, runner_y_new, cls_tm_id, cls_tm_x, cls_tm_y = move_runner(
        runner_id, 
        att_pos_end, 
        def_pos_end, 
        att_pos_start,
        def_pos_start,
        attacking_side,
        pitch_length,
        pitch_width,
        run_time)
    
    pos_new.loc[pos_new.player_id == runner_id, "x"] = runner_x_new
    pos_new.loc[pos_new.player_id == runner_id, "y"] = runner_y_new
    if cls_tm_id != None:
        pos_new.loc[pos_new.player_id == cls_tm_id, "x"] = cls_tm_x
        pos_new.loc[pos_new.player_id == cls_tm_id, "y"] = cls_tm_y
    
    def_pos_new = pos_new[pos_new["team_id"] == def_team_id]
    att_pos_new = pos_new[pos_new["team_id"] == att_team_id]

    if cls_tm_id is not None:
        cls_tm_x = float(pos_new.loc[pos_new.player_id == cls_tm_id, "x"].iloc[0])
        cls_tm_y = float(pos_new.loc[pos_new.player_id == cls_tm_id, "y"].iloc[0])
        
    
    # STEP 2: Identify and move main, sec, and close tm def.
    
    # 2.1: Identify the main defender. 
    main_def = def_id(
        runner_id,
        def_pos_start,
        att_pos_start,
        attacking_side,
        pitch_length,
        ball_x, ball_y)

    # 2.2: Identify secondary defender at the end of the run.
    second_def = def_id(
        runner_id,
        def_pos_end,
        att_pos_end,
        attacking_side,
        pitch_length,
        ball_x, ball_y)
    
    # Eliminate second_def if same as main
    if second_def == main_def:
        second_def = None
    
    # Eliminate second_def if they are closer to the ball carrier than to the runner
    if second_def is not None:
        sd_x_end = pos_end.loc[pos_end.player_id == second_def, "x"].iloc[0]
        sd_y_end = pos_end.loc[pos_end.player_id == second_def, "y"].iloc[0]
        
        dist_sd_ball   = math.hypot(sd_x_end - ball_x, sd_y_end - ball_y)
        dist_sd_runner = math.hypot(sd_x_end - runner_x_end, sd_y_end - runner_y_end)

        if dist_sd_ball < dist_sd_runner:
            second_def = None
            
    # 2.3: Don't move who is marking bc.
    bc_def = ball_carrier_marker(
        carrier_id, def_pos_end, att_pos_end, goal_x) 

    if bc_def == main_def:
        main_def = None
    elif bc_def == second_def:
        second_def = None
    
    # 2.4: Identify close tm def.
    if cls_tm_id != None:
        cls_def = def_id(
            cls_tm_id,
            def_pos_new,
            att_pos_new,
            attacking_side,
            pitch_length,
            ball_x, ball_y)
        
        if (cls_def == bc_def) and (cls_tm_id != carrier_id):
            cls_def = None
            
        if (cls_def == main_def) or (cls_def == second_def) or (cls_def == None):
            cls_def = def_id(
                cls_tm_id,
                def_pos_start,
                att_pos_start,
                attacking_side,
                pitch_length,
                ball_x, ball_y)
            
            if (cls_def == bc_def) and (cls_tm_id != carrier_id):
                cls_def = None
                
            if cls_def == None:
                cls_def_cand = def_id(
                    cls_tm_id,
                    def_pos_end,
                    att_pos_end,
                    attacking_side,
                    pitch_length,
                    ball_x, ball_y)
                
                if (cls_def_cand == bc_def) and (cls_tm_id != carrier_id):
                    cls_def_cand = None
                
                if cls_def_cand != None:
                    x_cdc = pos_start.loc[
                        pos_start.player_id == cls_def_cand, "x"].iloc[0]
                    y_cdc = pos_start.loc[
                        pos_start.player_id == cls_def_cand, "y"].iloc[0]
                
                    x_ct = pos_start.loc[
                        pos_start.player_id == cls_tm_id, "x"].iloc[0]
                    y_ct = pos_start.loc[
                        pos_start.player_id == cls_tm_id, "y"].iloc[0]
                
                    dist = np.sqrt((x_cdc-x_ct)**2 + (y_cdc-y_ct)**2)

                    if dist < run_time * run_speed:
                        cls_def = cls_def_cand
                
    else: 
        cls_def = None
        
    if cls_def == main_def:
        cls_def = None
    elif cls_def == second_def:
        second_def = None

    # -----------------------------------------------------------------------
    # Defender start positions for physical reachability checks (Step 3).
    # -----------------------------------------------------------------------
    if main_def is not None:
        _md_start = pos_start.loc[pos_start.player_id == main_def].iloc[0]
        main_def_sx = float(_md_start["x"])
        main_def_sy = float(_md_start["y"])
        main_def_vx = float(_md_start["v_x"])
        main_def_vy = float(_md_start["v_y"])

    if second_def is not None:
        _sd_start = pos_start.loc[pos_start.player_id == second_def].iloc[0]
        sec_def_sx = float(_sd_start["x"])
        sec_def_sy = float(_sd_start["y"])
        sec_def_vx = float(_sd_start["v_x"])
        sec_def_vy = float(_sd_start["v_y"])

    if cls_def is not None:
        _cd_start = pos_start.loc[pos_start.player_id == cls_def].iloc[0]
        cls_def_sx = float(_cd_start["x"])
        cls_def_sy = float(_cd_start["y"])
        cls_def_vx = float(_cd_start["v_x"])
        cls_def_vy = float(_cd_start["v_y"])

    # STEP 3: Move defenders.
    
    # 3.0: Check if someone is already in a good def pos.
    conv_def = def_id(
        runner_id,
        def_pos_new,
        att_pos_new,
        attacking_side,
        pitch_length,
        ball_x, ball_y)
    
    if conv_def is not None:
        conv_def_x = pos_new.loc[pos_new.player_id == conv_def, "x"].iloc[0]
        if attacking_side == "left_to_right":
            if conv_def_x < runner_x_new:
                conv_def = None
        elif attacking_side == "right_to_left":
            if conv_def_x > runner_x_new:
                conv_def = None

    if conv_def is not None:
        conv_def_x = pos_new.loc[pos_new.player_id == conv_def, "x"].iloc[0]
        conv_def_y = pos_new.loc[pos_new.player_id == conv_def, "y"].iloc[0]
        dist_conv_to_runner = math.hypot(
            conv_def_x - runner_x_new, conv_def_y - runner_y_new)
        att_xy = att_pos_new[["x", "y"]].to_numpy()
        nearest_att_dist = np.hypot(
            att_xy[:, 0] - conv_def_x,
            att_xy[:, 1] - conv_def_y).min()
        if nearest_att_dist < dist_conv_to_runner:
            conv_def = None
    
    # 3.1: Alter main defender positioning. 
    if main_def is not None:
        if (conv_def == None) or (conv_def == main_def):
            # No one is already defending the runner in no-run state
            # Thus it is the main def job
            
            # Tactical points near runner's new position
            pts = ellipse_points(
                (runner_x_new, runner_y_new), 
                min(run_time * jog_speed, max_r),
                min(run_time * jog_speed, max_r)
            )
            
            # Restrict tactical points to physically reachable 
            pts = intersect_disk_with_reachable(
                pts, main_def_sx, main_def_sy, run_time,
                main_def_vx, main_def_vy, def_speed)
            
            # determine the position with the best score
            main_def_x, main_def_y = position_score(
                runner_x_new, runner_y_new,
                ball_x, ball_y,
                pts,
                attacking_side,
                main_def,
                def_pos_end,
                pitch_length
            )
            
        else:
            # Someone else is defending the runner already
            
            # Check if someone else is unmarked near main def at the end 
            x_md = pos_end.loc[pos_end.player_id == main_def, "x"].iloc[0]
            y_md = pos_end.loc[pos_end.player_id == main_def, "y"].iloc[0]
            marking = is_marking(
                x_md, 
                y_md, 
                att_pos_new, 
                attacking_side
            )
            
            if marking == True:
                main_def_x = x_md
                main_def_y = y_md
            
            else:
                # If not marking someone use fallback_main positioning
                md_move = jog_speed * run_time
                y1 = max(-pitch_width / 2, y_md - md_move)
                y2 = min( pitch_width / 2, y_md + md_move)

                main_def_x, main_def_y = fallback_main(
                    x_md, 
                    y_md,
                    y1, 
                    y2,
                    def_pos_new, 
                    att_pos_new,
                    main_def
                )
            
        pos_new.loc[pos_new.player_id == main_def, "x"] = main_def_x
        pos_new.loc[pos_new.player_id == main_def, "y"] = main_def_y

        def_pos_new = pos_new[pos_new["team_id"] == def_team_id]
        att_pos_new = pos_new[pos_new["team_id"] == att_team_id]

    
    # 3.2: Alter secondary defender's position (if present).
    if second_def is not None:

        x_start_sd = pos_start.loc[pos_start.player_id == second_def, "x"].iloc[0]
        y_start_sd = pos_start.loc[pos_start.player_id == second_def, "y"].iloc[0]

        other_def_start = def_pos_start[def_pos_start.player_id != second_def]
        other_def_end   = def_pos_end[def_pos_end.player_id != second_def]
        avg_dx = float((other_def_end["x"].values - other_def_start["x"].values).mean())

        x_norun = x_start_sd + avg_dx
        y_norun = y_start_sd

        team_dist     = abs(avg_dx)
        residual_time = max(0.0, run_time - team_dist / jog_speed)

        r_x = min(residual_time * (jog_speed + 1), max_r)
        r_y = min(run_time * (jog_speed + 1), 20.0)

        r = r_y

        unmarked_attackers = check_unmarked(
            x_norun, y_norun, r,
            att_pos_new, def_pos_new,
            second_def, goal_x)

        if unmarked_attackers is None:
            # No unmarked attackers — use fallback_second
            sec_def_x, sec_def_y = fallback_second(
                x_norun, y_norun,
                r_x, r_y,
                def_pos_new, att_pos_new,
                second_def,
                goal_x=goal_x,
                pitch_width=pitch_width)

        elif len(unmarked_attackers) == 1:
            marking_assignment = unmarked_attackers.iloc[0]["player_id"]

            att_x = pos_end.loc[pos_end.player_id == marking_assignment, "x"].iloc[0]
            att_y = pos_end.loc[pos_end.player_id == marking_assignment, "y"].iloc[0]

            pts = ellipse_points((x_norun, y_norun), max(r_x, 0.5), max(r_y, 0.5))
            pts = intersect_disk_with_reachable(
                pts, sec_def_sx, sec_def_sy, run_time,
                sec_def_vx, sec_def_vy, def_speed)

            sec_def_x, sec_def_y = position_score(
                att_x, att_y,
                ball_x, ball_y,
                pts,
                attacking_side,
                second_def,
                def_pos_end,
                pitch_length
            )

        elif len(unmarked_attackers) > 1:

            unmarked_attackers["dist_to_goal"] = np.sqrt(
                (unmarked_attackers["x"] - goal_x)**2
                + unmarked_attackers["y"]**2
            )
            wts = 1 / unmarked_attackers["dist_to_goal"]
            target_x = x_norun
            target_y = float((unmarked_attackers["y"] * wts).sum() / wts.sum())
        
            pts = ellipse_points((x_norun, y_norun), max(r_x, 0.5), max(r_y, 0.5))
            pts = intersect_disk_with_reachable(
                pts, sec_def_sx, sec_def_sy, run_time,
                sec_def_vx, sec_def_vy, def_speed)
        
            sec_def_x, sec_def_y = fallback_second(
                target_x, target_y,
                r_x, r_y,
                def_pos_new, att_pos_new,
                second_def,
                goal_x=goal_x,
                pitch_width=pitch_width)
            pos_new.loc[pos_new.player_id == second_def, "x"] = sec_def_x
            pos_new.loc[pos_new.player_id == second_def, "y"] = sec_def_y
    
            def_pos_new = pos_new[pos_new["team_id"] == def_team_id]
            att_pos_new = pos_new[pos_new["team_id"] == att_team_id]
            
    # 3.3: Alter cls tm defender positioning. 
    if cls_def is not None:
        pts = ellipse_points(
            (cls_tm_x, cls_tm_y), 
            min(run_time * jog_speed, max_r),
            min(run_time * jog_speed, max_r)
        )
        pts = intersect_disk_with_reachable(
            pts, cls_def_sx, cls_def_sy, run_time,
            cls_def_vx, cls_def_vy, def_speed)
        
        cls_def_x, cls_def_y = position_score(
            cls_tm_x, cls_tm_y,
            ball_x, ball_y,
            pts,
            attacking_side,
            cls_def,
            def_pos_end,
            pitch_length
        )
        pos_new.loc[pos_new.player_id == cls_def, "x"] = cls_def_x
        pos_new.loc[pos_new.player_id == cls_def, "y"] = cls_def_y
    

    # STEP 4: Update velocities (average from start -> new state)
    if run_time > 0:
        # 4a) Runner
        runner_start = pos_start.loc[pos_start.player_id == runner_id].iloc[0]
        runner_new   = pos_new.loc[pos_new.player_id == runner_id].iloc[0]
        runner_vx = (runner_new["x"] - runner_start["x"]) / run_time
        runner_vy = (runner_new["y"] - runner_start["y"]) / run_time
        pos_new.loc[pos_new.player_id == runner_id, ["v_x", "v_y"]] = [
            runner_vx, runner_vy]
        
        # 4b) Main defender
        if main_def is not None:
            main_start = pos_start.loc[pos_start.player_id == main_def].iloc[0]
            main_new   = pos_new.loc[pos_new.player_id == main_def].iloc[0]
            main_vx = (main_new["x"] - main_start["x"]) / run_time
            main_vy = (main_new["y"] - main_start["y"]) / run_time
            pos_new.loc[pos_new.player_id == main_def, ["v_x", "v_y"]] = [
                main_vx, main_vy]
        
        # 4c) Secondary defender
        if (second_def is not None) and (second_def != main_def):
            sec_start = pos_start.loc[pos_start.player_id == second_def].iloc[0]
            sec_new   = pos_new.loc[pos_new.player_id == second_def].iloc[0]
            sec_vx = (sec_new["x"] - sec_start["x"]) / run_time
            sec_vy = (sec_new["y"] - sec_start["y"]) / run_time
            pos_new.loc[pos_new.player_id == second_def, ["v_x", "v_y"]] = [
                sec_vx, sec_vy]

        # 4d) Close teammate
        if cls_tm_id is not None:
            cls_tm_start = pos_start.loc[pos_start.player_id == cls_tm_id].iloc[0]
            cls_tm_new   = pos_new.loc[pos_new.player_id == cls_tm_id].iloc[0]
            cls_tm_vx = (cls_tm_new["x"] - cls_tm_start["x"]) / run_time
            cls_tm_vy = (cls_tm_new["y"] - cls_tm_start["y"]) / run_time
            pos_new.loc[pos_new.player_id == cls_tm_id, ["v_x", "v_y"]] = [
                cls_tm_vx, cls_tm_vy]

        # 4e) Close teammate's defender
        if cls_def is not None:
            cls_def_start = pos_start.loc[pos_start.player_id == cls_def].iloc[0]
            cls_def_new   = pos_new.loc[pos_new.player_id == cls_def].iloc[0]
            cls_def_vx = (cls_def_new["x"] - cls_def_start["x"]) / run_time
            cls_def_vy = (cls_def_new["y"] - cls_def_start["y"]) / run_time
            pos_new.loc[pos_new.player_id == cls_def, ["v_x", "v_y"]] = [
                cls_def_vx, cls_def_vy]
    
    return pos_new