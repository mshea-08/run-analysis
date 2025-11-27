import pandas as pd
import numpy as np
from skillcorner.client import SkillcornerClient
import SpaceManipulationFunctions as sm

# This is all the functions used for the logistic regression and some example code of how we were using it during our meetings

#--------------------------------------
# Functions from Prof Shea's code
#--------------------------------------
def convert_pitch_loc(l,w,x,y):
    """
    Takes Wyscout location and converts to skillcorner location.
    """
    x_new = -l/2 + (l/100)*x
    y_new = w/2 - (w/100)*y
    
    return x_new, y_new

def possession_direction(team_id, match_data, frame_data):
    """
    frame_data = match_tracking_data[frame]

    """
    period = frame_data["period"]
    home_team_id = match_data["home_team"]["id"]
    
    if period == 1:
        if team_id == home_team_id:
            direction = match_data["home_team_side"][0]
        else:
            direction = match_data["home_team_side"][1]
    else:
        if team_id == home_team_id:
            direction = match_data["home_team_side"][1]
        else:
            direction = match_data["home_team_side"][0]
    
    return direction


#--------------------------------------
# Helper functions I created/modified
#--------------------------------------
# new functions to get wyscout coords from skillcorner

def convert_pitch_loc_inverse(l, w, x_new, y_new):
    """
    Takes SkillCorner location (x_new, y_new) and converts back to Wyscout location (x, y).
    """
    x = 100 * (x_new + l/2) / l
    y = 100 * (w/2 - y_new) / w
    return x, y

def wyscout_from_skillcorner(x_sc, y_sc, pitch_length, pitch_width, direction):
    # SkillCorner -> Wyscout (standardized left-to-right)
    x_w, y_w = convert_pitch_loc_inverse(pitch_length, pitch_width, x_sc, y_sc)

    # Undo the directional standardization
    if direction == "right_to_left":
        x_w = 100 - x_w
        y_w = 100 - y_w

    return round(x_w), round(y_w)

# modified get team id

def get_team_id(pass_info):
    # pull some important info from series
    passer_id = pass_info["sk_player_id"]
    match_data = pass_info["match_data"]
    frame_data = pass_info["frame_data"]
    
    player_positions = sm.create_player_positions_2(
        match_data, frame_data)
    
    team_id = player_positions.loc[
        player_positions["player_id"] == passer_id]["team_id"]
    
    if team_id.empty:
            return None
    
    return team_id.iloc[0]


#--------------------------------------
# This is what we use for new length
#--------------------------------------

def add_receiver_end_position(pass_test_df):
   '''
   take in data frame of pass data and add columns containing (intended) pass recipient x and y coords in wyscout
   '''
   # create empty list to store x and y for receiver
   recipient_x = []
   recipient_y = []

   for idx, row in pass_test_df.iterrows():
      # get frame of pass
      frame = row['frame'] 

      # get frame data of pass
      frame_data = row['frame_data'] #holds player positions/velocity at frame

      # get player data (array) and turn into df
      player_data = pd.DataFrame(list(frame_data['player_data']))

      # pull match data from df
      match_data = row['match_data'] #holds match data from the pass (pitch length/width, players, positions, pids, etc...)

      # get pitch l and w from match data 
      l = match_data['pitch_length']
      w = match_data['pitch_width']

      # check if pass was completed (if complete use real recipient id // if incomplete use intended)
      if row['pass_accurate']:
         recipient_id = row['sk_pass_recipient_id']
      else:
         recipient_id = row['intended_receiver']

      # pull recipient team for direction of play (need for translation)
      recipient_team = get_team_id(row)

      if recipient_team is None:
         recipient_x.append(np.nan)
         recipient_y.append(np.nan)
         continue

      # determine direction
      direction = possession_direction(recipient_team, match_data, frame_data)

      mask = player_data["player_id"] == recipient_id
      rec_row = player_data.loc[mask, ["x", "y"]]

      if rec_row.empty:
         # handle missing detection
         recipient_x.append(np.nan)
         recipient_y.append(np.nan)
      else:
         recipient_x_sc = rec_row["x"].iloc[0]
         recipient_y_sc = rec_row["y"].iloc[0]

         #translate to wyscout coords
         receiver_x_wy, receiver_y_wy = wyscout_from_skillcorner(recipient_x_sc, recipient_y_sc, l, w, direction)
         recipient_x.append(receiver_x_wy)
         recipient_y.append(receiver_y_wy)

   pass_test_df['recipient_x'] = recipient_x
   pass_test_df['recipient_y'] = recipient_y
   
   return pass_test_df

def new_length(pass_test_df):
   '''
   Use recipient x,y to calculate new pass length
   '''
   # define columns needed for calculation
   cols = ['location_x', 'location_y', 'recipient_x', 'recipient_y']
   
   # pull columns from df
   passer_receiver_x_y = pass_test_df[cols]

   # create empty list to store new lengths
   new_lengths = []

   # iterrate through rows and calculate new length
   for _, row in passer_receiver_x_y.iterrows():
      length = np.sqrt((row['location_x'] - row['recipient_x'])**2 + (row['location_y'] - row['recipient_y'])**2)
      new_lengths.append(round(length, 0)) #round length bc easier to read

   # add new lengths as a column to the df
   pass_test_df['new_length'] = new_lengths

   return pass_test_df


#--------------------------------------
# Updated version of gaku's functions
#--------------------------------------

def pressure_on_receiver(pass_test_df, radius=5):
    pressures = []
    for idx, row in pass_test_df.iterrows():
        
        receiver_id = row['sk_pass_recipient_id'] if row['pass_accurate'] else row['intended_receiver']
        match_data = row['match_data']
        frame_data = row['frame_data']
        player_positions = sm.create_player_positions_2(match_data, frame_data)
        
        if player_positions is None or player_positions.empty:
            pressures.append(np.nan)
            continue
        receiver_row = player_positions[player_positions['player_id'] == receiver_id]
        
        if receiver_row.empty:
            pressures.append(np.nan)
            continue
        
        receiver_team = receiver_row['team_id'].values[0]
        rx = receiver_row['x'].values[0]
        ry = receiver_row['y'].values[0]

        defenders = player_positions[player_positions['team_id'] != receiver_team]
        dists = np.sqrt((defenders['x'] - rx)**2 + (defenders['y'] - ry)**2)
        pressures.append((dists <= radius).sum())
    # df_pass = pass_test_df.copy()
    pass_test_df['pressure_on_receiver'] = pressures
    return pass_test_df #df_pass


def pressure_on_passer(pass_test_df, radius=5):
    """
    Adds 'pressure_on_passer' to df_pass for passes in the match.
    """
    pressure_counts = []
        
    for _, row in pass_test_df.iterrows():
        # pull match and frame data for that pass
        match_data = row['match_data']
        frame_data = row['frame_data']
        
        # pull passer id
        passer_id = row['sk_player_id']

        # create player positions for that pass 
        player_positions = sm.create_player_positions_2(match_data, frame_data) 

        # Make sure we got a result
        if player_positions is None:
            pressure_counts.append(np.nan)
            continue

        # Get passer info
        passer_row = player_positions[player_positions['player_id'] == passer_id]
        if passer_row.empty:
            pressure_counts.append(np.nan)
            continue

        passer_team = passer_row['team_id'].values[0]
        passer_x = passer_row['x'].values[0]
        passer_y = passer_row['y'].values[0]

        # Filter defenders (opposing team)
        defenders = player_positions[player_positions['team_id'] != passer_team]

        # Compute distance to each defender at this frame
        dists = np.sqrt((defenders['x'] - passer_x)**2 + (defenders['y'] - passer_y)**2)
        pressure_counts.append((dists <= radius).sum())

    pass_test_df['pressure_on_passer'] = pressure_counts
    return pass_test_df

def in_cone(def_x, def_y, passer_x, passer_y, recipient_x, recipient_y, cone_theta_deg, cutoff):
    
    # vector from passer to recipient
    dx_PR = recipient_x - passer_x
    dy_PR = recipient_y - passer_y
    
    # vector from passer to defender
    dx_PD = def_x - passer_x
    dy_PD = def_y - passer_y
    
    # distances 
    dist_PR = np.sqrt(dx_PR**2 + dy_PR**2)
    dist_PD = np.sqrt(dx_PD**2 + dy_PD**2)
    
    # angle betweeen vectors
    dot = dx_PR * dx_PD + dy_PR * dy_PD
    
    if dist_PD == 0 or dist_PR == 0:
        angle = np.pi
    else:
        angle = np.arccos(dot/(dist_PR * dist_PD + 1e-9))  # add small term to avoid div by zero
        
    # Only up to cutoff
    within_cone_length = dist_PD <= min(cutoff, dist_PR)
    within_angle = angle <= np.deg2rad(cone_theta_deg) / 2
    return within_cone_length and within_angle

def in_cylinder(def_x, def_y, passer_x, passer_y, recipient_x, recipient_y, cone_theta_deg, cutoff):
    # Vector from passer to receiver
    dx_PR = recipient_x - passer_x
    dy_PR = recipient_y - passer_y
    pass_length = np.hypot(dx_PR, dy_PR)
    # Unit vector along pass direction
    ux = dx_PR / (pass_length + 1e-9)
    uy = dy_PR / (pass_length + 1e-9)
    # Calculate tube width from cone geometry
    theta_rad = np.deg2rad(cone_theta_deg)
    tube_width = 2 * cutoff * np.tan(theta_rad / 2)
    # Start/end for cylinder
    x1 = passer_x + ux * cutoff
    y1 = passer_y + uy * cutoff
    # x2 = recipient_x #do we need this? never mentioned again the the function
    # y2 = recipient_y
    # Project defender onto pass line, compute perpendicular distance
    dx = def_x - x1
    dy = def_y - y1
    # Project onto line (between cutoff and receiver)
    proj = ((def_x - x1) * ux + (def_y - y1) * uy)
    in_segment = 0 <= proj <= (pass_length - cutoff)
    # Perpendicular distance to line
    perp = np.abs(-uy * dx + ux * dy)
    return in_segment and (perp <= tube_width/2)

def in_semi_circle(def_x, def_y, recipient_x, recipient_y, cone_theta_deg, cutoff):
    # Calculate tube width from cone geometry
    theta_rad = np.deg2rad(cone_theta_deg)
    tube_width = 2 * cutoff * np.tan(theta_rad / 2)
    dist = np.hypot(def_x - recipient_x, def_y - recipient_y)
    # radius is half the tube width
    return dist <= tube_width/2

def defenders_in_passing_lane(df_pass, player_positions, cone_theta_deg=30, cutoff=20):
    """
    Adds 'defenders_in_lane' column: number of defenders in cone or semicircle for each completed pass with pass_length <= max_pass_length.
    """
    lane_counts = []
    passes = df_pass
    for idx, row in passes.iterrows():
        frame = row['frame']
        passer_id = row['sk_player_id']
        if df_pass.loc[idx, 'pass_accurate']: #if true
            recipient_id = row['sk_pass_recipient_id']
        else:
            recipient_id = row['intended_receiver']
        pos_frame = player_positions[player_positions['frame'] == frame]
        if passer_id not in pos_frame['player_id'].values or recipient_id not in pos_frame['player_id'].values:
            lane_counts.append(np.nan)
            continue
        passer = pos_frame[pos_frame['player_id'] == passer_id].iloc[0]
        receiver = pos_frame[pos_frame['player_id'] == recipient_id].iloc[0]
        passer_team = passer['team_id']
        px, py = passer['x'], passer['y']
        rx, ry = receiver['x'], receiver['y']
        defenders = pos_frame[pos_frame['team_id'] != passer_team]
        count = 0
        for _, def_row in defenders.iterrows():
            dx, dy = def_row['x'], def_row['y']
            if (
                in_cone(dx, dy, px, py, rx, ry, cone_theta_deg, cutoff)
                or in_cylinder(dx, dy, px, py, rx, ry, cone_theta_deg, cutoff)
                or in_semi_circle(dx, dy, rx, ry, cone_theta_deg, cutoff)
            ):
                count += 1
        lane_counts.append(count)
    passes = passes.copy()
    passes['defenders_in_lane'] = lane_counts
    return passes
   
def flip_passes_to_right(df, y_center=50, y_max=100, y_cols=['location_y', 'pass_end_location_y', 'recipient_y'], angle_col='pass_angle'): #right of y axis (l-r)
    df = df.copy()
    # Identify passes that need flipping
    mask_flip = df['location_y'] >= y_center
    for col in y_cols:
        if col in df.columns:
            df.loc[mask_flip, col] = y_max - df.loc[mask_flip, col]
    # Flip angles for those passes
    if angle_col in df.columns:
        df.loc[mask_flip, angle_col] = -df.loc[mask_flip, angle_col]
    return df


#---------------------------------------------
# function version of the full features loop
#---------------------------------------------

# test on random subset of k matches
import random
random.seed(67)

# Load the full pass DataFrame from parquet
df_pass_all = pd.read_parquet("pass_test.parquet")

def full_features(df_pass_all, k):
    '''
    Takes in a dataframe of pass data and some integer k
    k: number of matches to include in the random sample
    Outputs a dataframe with pressure on passer/receiver, defenders in passing lane, 
    (intended or actual) receiver x/y, length between passer and receiver
    '''


    # Build a df of available match IDs and match data
    match_id_and_data_unique = (
        df_pass_all[["sk_match_id", "match_data"]]
        .dropna(subset=["sk_match_id"])
        .assign(sk_match_id=lambda d: d["sk_match_id"].astype(int))
        .drop_duplicates(subset=["sk_match_id"])
        )
    
    just_match_ids = match_id_and_data_unique['sk_match_id'].to_list()
    
    k_or_min = min(k, len(just_match_ids))
    all_match_results = []

    for match_id in random.sample(just_match_ids, k_or_min):
        # print which is processing to see how fast the code is running
        print(f"Processing match {match_id}...")
        
        # Subset for this match only
        df_pass_match = df_pass_all[df_pass_all['sk_match_id'] == match_id].copy().reset_index(drop=True)

        # Load match and tracking data
        
        # match_data = match_id_and_data_unique[match_id_and_data_unique['sk_match_id'] == match_id, 'match_data']
        # Build player_positions for all frames once
        positions_list = []
        # get all passes from that match
        for _, row in df_pass_match.iterrows():
            match_data = row['match_data']
            frame_data = row['frame_data']
            pos = sm.create_player_positions_2(match_data, frame_data)
            if pos is not None and not pos.empty:
                pos['frame'] = row['frame']
                positions_list.append(pos)
        if not positions_list:
            continue  # No player positions found, skip
        player_positions = pd.concat(positions_list, ignore_index=True)
        
        # Add pressure and passing lane features
        df_pass_match = pressure_on_passer(df_pass_match, radius=5)
        df_pass_match = defenders_in_passing_lane(df_pass_match, player_positions, cone_theta_deg=30, cutoff=20)
        df_pass_match = pressure_on_receiver(df_pass_match, radius=5)
        df_pass_match = add_receiver_end_position(df_pass_match)
        df_pass_match = new_length(df_pass_match)
        

        all_match_results.append(df_pass_match)

    # Concatenate all results back into a single DataFrame
    return pd.concat(all_match_results, ignore_index=True)


# Use of it on 25 matches:
ff_1 = full_features(df_pass_all, 25)

#--------------------------------------
# Logistic regression!!
#--------------------------------------
# this is just how I was using it with the same ff_1 data from above

df_flipped = flip_passes_to_right(ff_1, y_center=50, y_max=100)
df_model = df_flipped.dropna()

feature_cols = [
    "location_x",
    "location_y",
    "pass_angle",
    "new_length",
    "pressure_on_passer",
    "defenders_in_lane",
    'pressure_on_receiver',
]

Y = df_model['pass_accurate'].astype(int).copy()

X = df_model[feature_cols].copy()

from sklearn.preprocessing import StandardScaler

# Instantiate scaler and fit on features
scaler = StandardScaler()
scaler.fit(X)

# Transform features
X_scaled = scaler.transform(X.values)

from sklearn.model_selection import train_test_split

# Split data into train and test
X_train_scaled, X_test_scaled, Y_train, Y_test = train_test_split(X_scaled,
                                                                  Y,
                                                             train_size=.7,
                                                           random_state=67)

from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()

# Training the models 
logistic_regression.fit(X_train_scaled, Y_train)

# Making predictions with each model
log_reg_preds = logistic_regression.predict(X_test_scaled)

# Computing probabilities for log reg
log_reg_probs = logistic_regression.predict_proba(X_test_scaled)

log_reg_probs = pd.DataFrame(log_reg_probs)
log_reg_probs['pass_accurate'] = Y_test.to_list()

# look at the coefficients
pd.Series(logistic_regression.coef_.ravel(), index=feature_cols)
