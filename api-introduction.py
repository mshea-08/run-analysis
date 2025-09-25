#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 09:26:31 2025

@author: meredithshea
"""

import pandas as pd
from skillcorner.client import SkillcornerClient
from io import BytesIO
import matplotlib.pyplot as plt
from mplsoccer import Pitch


# Start up the Skillcorner client
client = SkillcornerClient(username='USERNAME', password='PASSWORD')

# Get the two available competition ids
comp_editions=pd.DataFrame(client.get_competition_editions(params={'user':'true'}))
comp_editions=comp_editions[['id','name']]

# Let's use the NWSL 2024 season for now
comp_id = comp_editions.id[0]

# Look at the available matches from the season
matches = client.get_matches(params={'competition_edition': comp_id})
match_ids = [entry['id'] for entry in matches] # creates an array of match_ids only

# Let's use a random game to look closer
match_id = match_ids[6]

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
time_stamp_data = match_tracking_data[10000]
player_positions = pd.DataFrame(time_stamp_data['player_data'])

# Let's use the mplsoccer package to put these on a soccer pitch
pitch_length = match_data['pitch_length']
pitch_width = match_data['pitch_width']

# in skillcorner the coordinates are:
    # -pitch_length/2 to pitch_length/2
    # -pitch_width/2 to pitch_width/2
pitch = Pitch(pitch_type='skillcorner',pitch_length = pitch_length,
              pitch_width = pitch_width)

# This is for organizing the players into their teams
away_team = pd.DataFrame(match_data['away_team'], 
                         index=[0])
home_team = pd.DataFrame(match_data['home_team'], 
                         index=[0])

(fix, ax) = pitch.draw()
for i,player in player_positions.iterrows():
    pid = player.player_id
    x = player.x
    y = player.y
    # Determine the players team
    team = players_df.loc[players_df.id == pid, 'team_id'].iloc[0]
    if team == away_team.id[0]:
        ax.scatter(x,y, color='#41B6E6')
    elif team == home_team.id[0]:
        ax.scatter(x,y,color='#5F249F')
# Lastly let's plot the ball
ball_x, ball_y = time_stamp_data['ball_data']['x'], time_stamp_data['ball_data']['y']
ax.scatter(ball_x,ball_y, s=10, color='black')
plt.show()

# Load off ball run and store as a DataFrame
off_ball_runs_csv = client.get_dynamic_events_off_ball_runs(
    match_id=match_id, 
    params={'file_format':'csv', 'ignore_dynamic_events_check': False}
    )
off_ball_runs_df = pd.read_csv(BytesIO(off_ball_runs_csv))

