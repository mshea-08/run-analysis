#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 10:11:58 2026

@author: meredithshea
"""

import numpy as np
import pandas as pd
from OrganizationFunctions import create_player_positions_2
from NoRunStateFunctions import no_run_state_ball
from NoRunStateFunctions import no_run_state
from PossessionValueFunctions import p_val_no_runner
from PossessionValueFunctions import team_pitch_control_matrix

# ==========================================
# Compute support score - metric component 3
# ==========================================

def support_value(
        player_positions,
        run_info,
        pitch_length,
        pitch_width,
        rows=12,
        cols=16,
        R=25.0,             # radius checked around ball carrier
        sigma=10.0,         # parameter for spacial wts
        sigma_press=3.0,    # parameter for immediate pressure
    ):
    """
    Support value around the ball carrier. For a given state, 
    penalized if the carrier is under heavy pressure.

    - The ball is assumed to be at the feet of the original ball carrier
      (run_info['player_in_possession_id']), regardless of who actually
      has the ball in the frame.
    - Uses team pitch control (runner INCLUDED) in a disk of radius R around
      that carrier, weighted by exp(-dist_to_carrier / sigma).

    Higher = more / better nearby supportive options for the original carrier,
    and the carrier is not under immediate pressure.

    """
    team_id = run_info["team_id"]
    ball_carrier_id = run_info["player_in_possession_id"]

    # 0) Find ball carrier's position in this state
    row = player_positions.loc[player_positions["player_id"] == ball_carrier_id]
    if row.empty:
        # Failsafe: if carrier is not on pitch in this frame, just return 0
        return 0.0

    ball_x = float(row["x"].iloc[0])
    ball_y = float(row["y"].iloc[0])

    # 1) Team pitch control (runner INCLUDED)
    team_pc = team_pitch_control_matrix(
        ball_x, 
        ball_y, 
        player_positions,
        team_id, 
        pitch_length, 
        pitch_width,
        rows=rows, 
        cols=cols,
        )
    team_pc = np.clip(np.nan_to_num(team_pc, nan=0.0), 0.0, 1.0)

    # 2) Build grid coordinates
    dy = pitch_width / rows
    dx = pitch_length / cols
    x_centers = -pitch_length / 2 + dx * (np.arange(cols) + 0.5)
    y_centers = -pitch_width / 2 + dy * (np.arange(rows) + 0.5)
    Xg, Yg = np.meshgrid(x_centers, y_centers)  # (rows, cols)

    X = Xg.ravel()
    Y = Yg.ravel()
    pc_flat = team_pc.ravel()

    # 3) Distance from each grid cell to the ball carrier
    d_ball = np.hypot(X - ball_x, Y - ball_y)

    # 4) Restrict to a disk of radius R
    mask = d_ball <= R
    if not np.any(mask):
        return 0.0

    # 5) Support weights: exp(-d_ball/sigma) * team_pitch_control
    spatial_weights = np.exp(-d_ball[mask] / sigma)
    weights = spatial_weights * pc_flat[mask]
    support_raw = float(weights.sum())

    # 6) Pressure on ball carrier
    # Opponents (defenders relative to the ball carrier's team)
    opponents = player_positions[player_positions["team_id"] != team_id]

    if opponents.empty:
        # No defenders -> no pressure
        unpressured_factor = 1.0
    else:
        opp_x = opponents["x"].to_numpy()
        opp_y = opponents["y"].to_numpy()
        d_def = np.hypot(opp_x - ball_x, opp_y - ball_y)

        # Pressure mass: close defenders contribute ~1, far defenders ~0
        pressure_raw = float(np.exp(-d_def / sigma_press).sum())

        # Convert to [0,1]: 1 when pressure_raw ~ 0, decreasing as pressure grows
        unpressured_factor = 1.0 / (1.0 + pressure_raw)

    # Final support: good structure around ball, but downweighted if carrier is swarmed
    support_final = support_raw * unpressured_factor

    return float(support_final)

# ======================================
# Final computation of metrics
# ======================================

def compute_run_metrics(
        row,        # row of run information, the positional data has been preloaded as columns here for speed
        xT_grid,    # locally saved xT matrix (same conventions as pitch control)
        ):
    """
    For a single run (row), compute:
      - delta_sm = p_val_no_runner(end) - p_val_no_runner(no_run)
      - delta_support  = support(end) - support(no_run)

    Returns (delta_sm, delta_support).
    """
    # Unpack from row
    match_data       = row.match_data
    frame_start_data = row.frame_start_data
    frame_end_data   = row.frame_end_data
    team_id          = row.team_id
    runner_id        = row.player_id
    attacking_side   = row.attacking_side
    ball_carrier     = row.player_in_possession_id

    pitch_length = match_data["pitch_length"]
    pitch_width  = match_data["pitch_width"]

    # Build run_info dict in the format your helpers expect
    run_info = {
        "frame_start": row.frame_start,
        "frame_end": row.frame_end,
        "team_id": team_id,
        "player_id": runner_id,
        "attacking_side": attacking_side,
        "player_in_possession_id": row.player_in_possession_id,
        "match_data": match_data,
        "frame_end_data": frame_end_data,
    }

    # ---------- END-OF-RUN STATE ----------
    player_positions_end = create_player_positions_2(
        match_data, frame_end_data
    )

    if player_positions_end is None or player_positions_end.empty:
        return np.nan, np.nan, np.nan

    # Ball position is chosen to be the ball carrier!!
    ball_x_end = player_positions_end.loc[
        player_positions_end.player_id == ball_carrier]["x"].iloc[0]
    ball_y_end = player_positions_end.loc[
        player_positions_end.player_id == ball_carrier]["y"].iloc[0]


    # p_val_no_runner (end)
    p_no_runner_end = p_val_no_runner(
        ball_x_end, ball_y_end,
        player_positions_end,
        runner_id,
        team_id,
        attacking_side,
        pitch_length, 
        pitch_width,
        xT_grid
        )
    

    # support_value (end)
    support_end = support_value(
        player_positions_end,
        run_info,
        pitch_length,
        pitch_width,
        rows=12,
        cols=16,
        R=25.0,
        sigma=10.0,
        )

    # ---------- NO-RUN STATE ----------
    player_positions_no_run = no_run_state(
        match_data,
        frame_start_data,
        frame_end_data,
        run_info
        )

    if player_positions_no_run is None or player_positions_no_run.empty:
        # If something went wrong building the no-run state
        return np.nan, np.nan, np.nan

    # Ball position for no-run state: your helper uses run_info
    ball_x_nr, ball_y_nr = no_run_state_ball(run_info)


    # p_val_no_runner (no-run)
    p_no_runner_nr = p_val_no_runner(
        ball_x_nr, 
        ball_y_nr,
        player_positions_no_run,
        runner_id,
        team_id,
        attacking_side,
        pitch_length, pitch_width,
        xT_grid
        )

    # support_value (no-run)
    support_no_run = support_value(
        player_positions_no_run,
        run_info,
        pitch_length,
        pitch_width,
        rows=12,
        cols=16,
        R=25.0,
        sigma=10.0,
        )

    # ---------- DIFFS (end - no_run) ----------
    delta_sm = p_no_runner_end - p_no_runner_nr
    delta_support  = support_end  - support_no_run

    return delta_sm, delta_support


def attach_run_metrics(
        runs_df, 
        xT_grid,
        ):
    """
    Adds metric computations directly to dataframe.

    """
    out = runs_df.apply(
        lambda row: pd.Series(compute_run_metrics(row, xT_grid),
                              index=["SM_score", "supp_score"]),
        axis=1
    )
    return pd.concat([runs_df, out], axis=1)

