#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 12:28:20 2025

@author: meredithshea
"""

import pandas as pd
import numpy as np
from scipy.stats import logistic
from skillcorner.client import SkillcornerClient

# Start up the Skillcorner client
client = SkillcornerClient(username='USERNAME', 
                           password='PASSWORD')
# -----------------------------------------------------
# NWSL competition_edition = 800
# WSL copetition_edition = 892
# -----------------------------------------------------
# Look at the available matches from the season
matches = client.get_matches(params={'competition_edition': 800})
match_ids = [entry['id'] for entry in matches] # creates an array of match_ids only

# Pick a game
match_id = match_ids[100]

# Load match data 
match_data = client.get_match(match_id=match_id)

# Organize player data in DataFrame
match_players = match_data["players"]
players_df = pd.DataFrame(match_players)

position_df = pd.DataFrame(players_df.player_role.to_list())
position_df = position_df.rename(columns={'id': 'position_id'})

players_df = pd.concat([players_df,position_df],axis=1)
players_df = players_df.drop('player_role',axis=1)

# Load match tracking data
match_tracking_data = client.get_match_tracking_data(match_id=match_id, params={'data_version': 3})

# Let's look at a single time stamp
time_stamp_data = match_tracking_data[32500]
player_positions = pd.DataFrame(time_stamp_data['player_data'])

# ======================================================
# Test out the time stamp by plotting 
# ======================================================
pitch_length = match_data['pitch_length']
pitch_width = match_data['pitch_width']

import matplotlib.pyplot as plt
from mplsoccer import Pitch

# ------------------------------------------------------
# ------------------------------------------------------

pitch = Pitch(pitch_type='skillcorner',pitch_length = pitch_length,
              pitch_width = pitch_width)

# This is for organizing the players into their teams
away_team = pd.DataFrame(match_data['away_team'], 
                         index=[0])
home_team = pd.DataFrame(match_data['home_team'], 
                         index=[0])

bay_fc_c1 = "#0D2032"  # Bay
utah_royals_c1 = "#FDB71A"  # Royals Yellow

(fix, ax) = pitch.draw()
for i,player in player_positions.iterrows():
    pid = player.player_id
    x = player.x
    y = player.y
    # Determine the players team
    team = players_df.loc[players_df.id == pid, 'team_id'].iloc[0]
    if team == away_team.id[0]:
        ax.scatter(x,y, color=utah_royals_c1)
    elif team == home_team.id[0]:
        ax.scatter(x,y,color=bay_fc_c1)
# Lastly let's plot the ball
ball_x, ball_y = time_stamp_data['ball_data']['x'], time_stamp_data['ball_data']['y']
ax.scatter(ball_x,ball_y, s=10, color='black')
plt.show()
# ======================================================
# ======================================================

# Compute player velocity at the time stamp 
time_stamp_data_2 = match_tracking_data[32499]
player_positions_2 = pd.DataFrame(time_stamp_data_2['player_data'])

# Create a single DataFrame from the two position dfs
player_positions.rename(columns={'x': 'x_2', 'y':'y_2'}, 
                        inplace=True)
player_positions_2.rename(columns={'x': 'x_1', 'y':'y_1'}, 
                        inplace=True)
player_positions.drop(columns=['is_detected'], 
                      inplace=True)
player_positions_2.drop(columns=['is_detected'], 
                      inplace=True)
players_movement = player_positions.merge(player_positions_2, 
                             on='player_id')
"""
Each frame is 0.1 sec apart and the distance measurements
given by SKillcorner are in meters, so if I'm using the 
previous frame to compute velocity I get:
    v_x = (x_2 - x_1)/0.1 m/s
    v_y = (y_2 - y_1)/0.1 m/s
"""
players_movement['v_x'] = (players_movement.x_2 - players_movement.x_1)/0.1
players_movement['v_y'] = (players_movement.y_2 - players_movement.y_1)/0.1

# Add these columns back to the position DataFrame
player_positions['v_x'] = players_movement['v_x']
player_positions['v_y'] = players_movement['v_y']
player_positions.rename(columns={'x_2': 'x', 'y_2':'y'}, 
                        inplace=True)

# Add each player's team to player_positions
player_positions = player_positions.merge(
    players_df[['id', 'team_id']].rename(columns={'id':'player_id'}),
    on='player_id',
    how='left')

def pitch_control_at_coord(x,y,ball_x,ball_y,player_positions, attack_id, defend_id):
    """
    Parameters
    ---------------------
    x, y : The location we are computing the pitch control of.
    ball_x, ball_y : The location of the ball. 
    player_positions : The DataFrame formed above. Each row should
        contain-- player_id, team_id, x, y, v_x, v_y.

    Returns
    ----------------------
    attack_prob, defend_prob 

    """
    # Time for the ball to get to the location
    ball_speed = 15 # assuming speed of 15 m/s
    dist = np.sqrt((x-ball_x)**2+(y-ball_y)**2)
    ball_time = dist/ball_speed
    
    # Time for players to reach ball 
    # ---------------------------------------
    # The idea:
        # Players run along current trajectory for 0.7 s,
        # then players alter course and run towards ball at max speed
    # ---------------------------------------
    player_speed = 5
    player_positions['x1'] = (
        player_positions.x + 0.7*player_positions.v_x
        )
    player_positions['y1'] = (
        player_positions.y + 0.7*player_positions.v_y
        )
    player_positions['time_to_ball'] = (
        np.sqrt((x - player_positions.x1)**2 + (y - player_positions.y1)**2)/player_speed
        + 0.7
        )
    # Computing probabilities
    # --------------------------------------
    # The idea:
        # Arrival time is Logistic RV with mean time_to_ball 
        # and parameter s (NOT standard deviation).
        # Time to control ball is exponential RV with 
        # parameter l.
    # --------------------------------------
    s = 0.37
    l = 4.3
    dt = 0.04
    player_positions['p_tot'] = 0
    p_sum = 0
    i = 0
    # Loop to compute probabilities
    while (1-p_sum) > 0.01 and i < 2500:
        i += 1
        p_sum = player_positions['p_tot'].sum()
        player_positions['arrival_prob'] = (
            logistic.cdf(ball_time + (i+1)*dt,loc=player_positions.time_to_ball,scale=s)
            )
        player_positions[f'p_{i}'] = (
            (1-p_sum)*player_positions['arrival_prob']*l*dt
            )
        
        player_positions['p_tot'] += player_positions[f'p_{i}']
        player_positions.drop(columns=['arrival_prob', f'p_{i}'], 
                              inplace=True)
    # Return probabilities
    attack_prob = player_positions[player_positions.team_id == attack_id]['p_tot'].sum()
    defend_prob = player_positions[player_positions.team_id == defend_id]['p_tot'].sum()
    return attack_prob, defend_prob


# ======================================================
# Visualize an example
# ======================================================
probs = np.zeros((34,53))

for i in range(53):
    for j in range(34):
        x_c = -pitch_length/2 + 2*i + 1
        y_c = -pitch_width/2 + 2*j + 1
        
        attack_prob, defend_prob = pitch_control_at_coord(x_c,y_c,ball_x,ball_y,player_positions, 3484, 3485)
        
        probs[j,i] = attack_prob
        
(fix, ax) = pitch.draw()
extent = (-pitch_length/2,pitch_length/2,-pitch_width/2,pitch_width/2)
plt.imshow(probs, origin='lower', extent=extent, cmap='bwr', interpolation='gaussian')
for i,player in player_positions.iterrows():
    pid = player.player_id
    x = player.x
    y = player.y
    # Determine the players team
    team = players_df.loc[players_df.id == pid, 'team_id'].iloc[0]
    if team == away_team.id[0]:
        ax.scatter(x,y, color=utah_royals_c1)
        plt.text(x+1, y+1, str(i), ha='right', va='bottom')
    elif team == home_team.id[0]:
        ax.scatter(x,y,color=bay_fc_c1)
        plt.text(x+1, y+1, str(i), ha='right', va='bottom')
# Lastly let's plot the ball
ball_x, ball_y = time_stamp_data['ball_data']['x'], time_stamp_data['ball_data']['y']
ax.scatter(ball_x,ball_y, s=10, color='black')
plt.title('Pitch Control')
plt.show()
    

