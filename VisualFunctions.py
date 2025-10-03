#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 10:20:13 2025

@author: meredithshea
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mplsoccer import Pitch


# ------------------------------------------------------
# NWSL Plots
# ------------------------------------------------------

def players_from_match_data(match_data):
    
    players_df = pd.DataFrame(match_data["players"])
    # flatten positional data
    position_df = pd.DataFrame(players_df.player_role.to_list())
    position_df = position_df.rename(columns={"id": "position_id"})
    # add positional data to data frame
    players_df = pd.concat([players_df,position_df],axis=1)
    players_df = players_df.drop("player_role",axis=1)
    
    return players_df

def color_distance(hex1, hex2):
    """
    Compute Euclidean distance between two hex colors in RGB space
    
    """
    rgb1 = np.array(mcolors.to_rgb(hex1))
    rgb2 = np.array(mcolors.to_rgb(hex2))
    return np.linalg.norm(rgb1 - rgb2)


def frame_plot_NWSL(match_data,match_tracking_data,frame):
    
    players_df = players_from_match_data(match_data)

    # extract player positions from tracking data
    player_positions = pd.DataFrame(
        match_tracking_data[frame]['player_data']
        )
    
    # extract ball position from tracking data
    ball_x = match_tracking_data[frame]["ball_data"]["x"]
    ball_y = match_tracking_data[frame]["ball_data"]["y"]

    # input pitch specs
    pitch_length = match_data['pitch_length']
    pitch_width = match_data['pitch_width']
    pitch = Pitch(pitch_type='skillcorner',pitch_length = pitch_length,
                  pitch_width = pitch_width)

    # input team specs
    away_team = pd.DataFrame(
        match_data['away_team'],
        index=[0]
        )
    home_team = pd.DataFrame(
        match_data['home_team'],
        index=[0]
        )
    
    # input team color data
    team_colors = {
        "2338": {"main": "#F1B1A5", "alt": "#202121"}, # ACFC
        "3485": {"main":"#0D2032", "alt": "#FF5049"}, # Bay
        "2337": {"main": "#3AB5E8", "alt": "#102B45"}, # Stars
        "1831": {"main": "#FF6900", "alt": "#8AB7E9"}, # Dash
        "2329": {"main": "#CF3339", "alt": "#62CBC9"}, # Current
        "2335": {"main": "#A9F1FD", "alt": "#000000"}, # Gotham
        "1833": {"main": "#00416B", "alt": "#AB0033"}, # Courage
        "2334": {"main": "#61259E", "alt": "#00ABFF"}, # Pride
        "1832": {"main": "#99242B", "alt": "#000000"}, # Thorns
        "1830": {"main": "#C5B5F2", "alt": "#26140C"}, # Racing
        "2332": {"main": "#FC1896", "alt": "#21C6D9"}, # Wave
        "2331": {"main": "#2E407A", "alt": "#D0A66B"}, # Reign
        "3484": {"main": "#FDB71A", "alt": "#0E1735"}, # Royals
        "2333": {"main": "#000000", "alt": "#EDE939"} # Spirit
        }

    (fix, ax) = pitch.draw()
    for i,player in player_positions.iterrows():
        # record player info
        pid = player.player_id
        x = player.x
        y = player.y
        
        # determine player team
        team = players_df.loc[players_df.id == pid, "team_id"].iloc[0]
        
        # determine player jersey number
        number = players_df.loc[players_df.id == pid, "number"].iloc[0]
        
        # storing colors
        home_main = team_colors[str(home_team.id[0])]["main"]
        away_main = team_colors[str(away_team.id[0])]["main"]
        away_alt = team_colors[str(away_team.id[0])]["alt"]
        
        if team == home_team.id[0]:
            ax.scatter(x,y, c=home_main)
            plt.text(x+1, y+1, str(number), ha='right', va='bottom',c=home_main)
        elif team == away_team.id[0]:
            if color_distance(home_main, away_main) > 0.25:
                ax.scatter(x,y,c=away_main)
                plt.text(x+1, y+1, str(number), ha='right', va='bottom',c=away_main)
            else:
                ax.scatter(x,y,c=away_alt)
                plt.text(x+1, y+1, str(number), ha='right', va='bottom',c=away_alt)
            
    # lastly, let's plot the ball
    ax.scatter(ball_x,ball_y, s=10, color='black')
    
    plt.show()
    