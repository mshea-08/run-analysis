#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 14:50:42 2025

@author: meredithshea
"""

import pandas as pd
from skillcorner.client import SkillcornerClient
from io import BytesIO
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import TrackingFunctions as track
from VisualFunctions import frame_plot_NWSL

# Start up the Skillcorner client
client = SkillcornerClient(username=' ', password=' ')

# We will work with a random NWSL match
# Load the match_data and match_tracking_data
match_data = client.get_match(match_id=1501121)

match_tracking_data = client.get_match_tracking_data(
    match_id=1501121, 
    params={'data_version': 3}
    )

# Load off_ball_run_df
off_ball_runs_csv = client.get_dynamic_events_off_ball_runs(
    match_id=1501121, 
    params={'file_format':'csv', 'ignore_dynamic_events_check': False}
    )
off_ball_runs_df = pd.read_csv(BytesIO(off_ball_runs_csv))

# Limit runs to runs in behind that last at least 3 sec with no one else running
test_runs_df = off_ball_runs_df[
    (off_ball_runs_df.duration > 3) &
    (off_ball_runs_df.event_subtype == "run_ahead_of_the_ball") &
    (off_ball_runs_df.n_simultaneous_runs == 0)
    ]

# Pick a run and plot
for i in range(test_runs_df.frame_start.iloc[5],test_runs_df.frame_end.iloc[5]):
    frame_plot_NWSL(match_data, match_tracking_data, i)
    plt.show()
    

# Use starting run frame
frame = test_runs_df.frame_start.iloc[5]

# Add velocities to match_tracking_data
match_tracking_data = track.add_velocities(match_tracking_data)

# Create player_positions
player_positions = track.create_player_positions(
    match_data,
    match_tracking_data,
    frame
    )

# Record ball position
ball_x, ball_y = track.ball_position(match_tracking_data, frame)

# Compute pitch_control matrix for team in position (KC Current, id = 2329)
pitch_length = match_data["pitch_length"]
pitch_width = match_data["pitch_width"]

# Compute pitch control matrix
KC_mat = track.team_pitch_control_matrix(
    ball_x,
    ball_y,
    player_positions,
    2329,
    pitch_length,
    pitch_width
    )

# Plot team pitch control
fig, ax = frame_plot_NWSL(match_data, match_tracking_data, frame)
extent = (-pitch_length/2,pitch_length/2,-pitch_width/2,pitch_width/2)
plt.imshow(KC_mat, origin='lower', extent=extent, cmap='RdBu', alpha=0.75, interpolation='gaussian')
plt.title('Pitch Control')
plt.show()

frame2 = test_runs_df.frame_end.iloc[5]
player_positions2 = track.create_player_positions(
    match_data,
    match_tracking_data,
    frame2
    )

# Compute pitch control matrix
KC_mat2 = track.team_pitch_control_matrix(
    ball_x,
    ball_y,
    player_positions2,
    2329,
    pitch_length,
    pitch_width
    )

# Plot team pitch control
fig, ax = frame_plot_NWSL(match_data, match_tracking_data, frame2)
extent = (-pitch_length/2,pitch_length/2,-pitch_width/2,pitch_width/2)
plt.imshow(KC_mat2, origin='lower', extent=extent, cmap='RdBu', alpha=0.75, interpolation='gaussian')
plt.title('Pitch Control')
plt.show()

# Compute just the pitch control of E. Wheeler (#5, pid = 816481)
wheeler_mat = track.player_pitch_control_matrix(
    ball_x, 
    ball_y, 
    player_positions, 
    816481, 
    pitch_length, 
    pitch_width
    )

# Plot Wheeler pitch control 
fig, ax = frame_plot_NWSL(match_data, match_tracking_data, frame)
extent = (-pitch_length/2,pitch_length/2,-pitch_width/2,pitch_width/2)
plt.imshow(wheeler_mat, origin='lower', extent=extent, cmap='Blues', interpolation='gaussian')
plt.title('Pitch Control')
plt.show()

# Plot Wheeler's pitch control over entire run
for frame in range(test_runs_df.frame_start.iloc[5],test_runs_df.frame_end.iloc[5]):
    # Create player_positions
    player_positions = track.create_player_positions(
        match_data,
        match_tracking_data,
        frame
        )
    # Record ball position
    ball_x, ball_y = track.ball_position(match_tracking_data, frame)
    # Compute just the pitch control of E. Wheeler (#5, pid = 816481)
    wheeler_mat = track.player_pitch_control_matrix(
        ball_x, 
        ball_y, 
        player_positions, 
        816481, 
        pitch_length, 
        pitch_width
        )
    fig, ax = frame_plot_NWSL(match_data, match_tracking_data, frame)
    extent = (-pitch_length/2,pitch_length/2,-pitch_width/2,pitch_width/2)
    plt.imshow(wheeler_mat, origin='lower', extent=extent, cmap='Blues', interpolation='gaussian')
    plt.title('Pitch Control')
    plt.show()


# Another run
for frame in range(test_runs_df.frame_start.iloc[4],test_runs_df.frame_end.iloc[4]):
    # Create player_positions
    player_positions = track.create_player_positions(
        match_data,
        match_tracking_data,
        frame
        )
    # Record ball position
    ball_x, ball_y = track.ball_position(match_tracking_data, frame)
    # Compute just the pitch control of Castellanos (pid = 62954)
    castellanos_mat = track.player_pitch_control_matrix(
        ball_x, 
        ball_y, 
        player_positions, 
        62954, 
        pitch_length, 
        pitch_width
        )
    fig, ax = frame_plot_NWSL(match_data, match_tracking_data, frame)
    extent = (-pitch_length/2,pitch_length/2,-pitch_width/2,pitch_width/2)
    plt.imshow(wheeler_mat, origin='lower', extent=extent, cmap='Blues', interpolation='gaussian')
    plt.title('Pitch Control')
    plt.show()

