# Overview 

The goal of this project is to create a metric that evaluates the quality of off-ball runs made in a game of soccer. In particular, most means of measurement rely on the runner actually recieving a pass (and, secondarily, on the quality of that pass). We aim to produce a meaningful value that is independent of the result of the play sequence. 

The off-ball run value, ORV, will rely on the following soccer principles:
1. **Potential quality of the pass created (PR score).** This will consider both the feasibility of the intended pass as well as the danger created from the resulting pass.
2. **Space manipulation caused by the run (SM score).** This will consider the space created for other players due to the action of the run.
3. **Support (S score).**  This will consider the change in pressure and support around the ball carrier. 

This page is in some amount of working order.

## Working Table of Contents

[Mathematical Description of the Model](#mathematical-description)

[No-Run State](#no-run-state)

[Current Code Available](#current-code-available)

[Other Technical Descriptions](#other-technical-descriptions)

## Data Availability

Tracking data has been provided by SkillCorner. For freely available tracking data check out [sample data](https://github.com/metrica-sports/sample-data) provided by Metrica. Note that the code provided here most likely needs to be edited to work for other data sources. 

# Mathematical Description

## Potential Reception Score, PR(r)

Let $xPass(r)$ denote the *expected pass completion* of the pass from the ball carrier to the runner and let $xT(r)$ denote the *expected threat* of the runner's position. Then the potential reception score is, 

$$PR(r) = xT(r) \cdot xPass(r)$$

## Space Manipulation Score, SM(r)

This part will analyze the amount of space the runner creates for their teammates only. 

Let $s$ denote the state of play after the run has ended and let $\hat{s}$ denote the counterfactual state where no off-ball run is attempted. LEt $PV^{-r}(s)$ denote the *possession value* of a state with the runner removed from the state. Space manipulation is computed as, 

$$PV^{-r}(s) - PV^{-r}(\hat{s})$$

For the code presented here, are possession value is given by, 

$$PV^{-r}(s) = \sum_{p \in G} xT(p) \cdot PC^{-r}(s;p)$$

where $G$ is a discretization of the pitch and $PC^{-r}(s)$ is the *runner removed pitch control*.

## Support Score, S(r)

The support score takes a radial neighborhood of the ball carrier and computes a weighted sum of the pitch control in this area. After that it looks at immediate pressure on the ball carrier and scales the score inversely with the pressure (more pressure $\implies$ smaller score).

## Understanding the Component Values

For all values, larger score is better. The potential reception score is always non-negative, however both the space manipulation score and support score may be negative. 

# No-Run State

The key to the space manipulation and support scores is the generation of the **no-run counterfactual state**. The idea behind the no-run state is to produce a state where most players behave in the same way, however the runner does not attempt the run. Anyone whose behavior was *influenced* by the run should behave appropriately given the new circumstances. 

The no-run state is built via a deterministic, principles based algorithm. We use basic concepts from physics--e.g. reaction time, maximum speed, etc--alongside soccer specific principles--e.g. goal side defending, lane blocking, defensive shape and spacing, etc--to create a series of position optimizing functions. The no-run state is only dependent on the **intial state** and **end state** of the run. For ease of computation it does not depend on intermediate frames. 

Below we will outline the main steps to the no-run state, although we will omit some detail. For a more detailed accounting see, *upload paper*.

## Move the Runner

The first step is to move the runner. To move the runner, we begin with their initial position, $(x_i,y_i)$ and evolve their $x$ (downfield) position by the average downfield change in their teammates. We then adjust their $y$ positioning to satisfy minimal spacing requirement. The spacing requirement depend on where on the field the ball and runner are. 

There are a few checks on the runners position to check that their position is legal. For instance, we check that the runner is not offside or the player that creates the offside line (for their own team). 

## Identify Influenced Defenders 

There are two types of defenders the algorithm classifies:
- main defender: determined to be defending the runner in the initial state of the run.
- secondary defender: determined to be defending the runner in the final state of the run.
Note that the algorithm may determine that no one is the main and/or secondary defender. If they are determined to be the same player we say that there is no secondary defender.

The main and secondary defenders are then given a *region of allowed movement*, which is determined by their starting and ending positions as well as the duration of the run. Within the region of movement an optimization function is used to produce the best location for the player. The optimization function for the main defender takes into account:
- the position of the ball and runner on the field,
- the defenders role (fwd/mid/def),
- any potential pass that could be made to the runner,
- and the defenders proximity to teammates.
The optimization function for the secondary defender takes into account:
- the position of the ball and runner on the field,
- other marker and unmarked attackers near the defender,
- and the proximity to teammates.

## Potential Improvements to the No-Run State

Below I would like to suggest some potential improvements that can be made to the no-run state. I also would like to encourage whoever is reading this to produce their own modifications and ideas!
- Hard cutoffs to keep the runner in the bounds of the field. Surprisingly there were no examples where this was an issue, but there is the potential to "bump" the runner off the pitch given the constraints on proximity to teammates. This can be easily added, I just haven't.
- Define and move *influenced teammates*. A good example of this would be the case where a runner moves downfield and a teammate moves to occupy the space they created. In the current set up the runner is moved to be some distance away from the teammate, but the teammate remains in this new position. It might be beneficial to give the teammate the ability to find another nearby pocket of space.
- Expand the region of movement for the second defender. Currently the second defenders potential movement options are defined only by a line. This probably should be extended (at the expense of some time), however the optimization function may need modifications. It may be interested to look at more frame and determine the *time of contact* of the second defender with the runner.
- Literally anything that makes it run fast. Per example it is not that slow, but for larger data analysis it would benefit from improvements. 

# Current Code Available

Here I give a description of the code available and how to use it. There are quite a few helper functions along the way and I won't describe all of their uses. 

## Organization Functions

This will discuss the code contained in OrganizationFunctions.py. The purpose of this file is mainly to organize and prepare the skillcorner data for our analysis. 

The function `add_velocities` adds velocities to all of tracking data. The functions `create_player_positions` and `create_player_positions_2` create the frame specific data frames that include relevant player information and their positions at the specified frame. These data frames are used throughout the code provided here. The functions `ball_position` and `ball_position_2` extract the location of the ball at a given fame. Below is a snipped of how these codes may be used. 

```
from skillcorner.client import SkillcornerClient
import pandas as pd
from io import BytesIO
from OrganizationFunction import add_velocities
from OrganizationFunction import create_player_positions_2

client = SkillcornerClient(username="username", password="password")

# load NWSL match ids 
matches = client.get_matches(params={'competition_edition': 800})
match_ids = [entry['id'] for entry in matches] # creates an array of match_ids only

match_id = # pick some match
match_data = client.get_match(match_id=match_id) # match_data contains information about the teams, pitch, etc
match_tracking_data = client.get_match_tracking_data(match_id=match_id, params={'data_version': 3}) # match_tracking_data contains all player positional data

match_tracking_data = sm.add_velocities(match_tracking_data)

# load skillcorner off-ball run infomation for the match of interest
off_ball_runs_csv = client.get_dynamic_events_off_ball_runs(match_id=match_id, params={'file_format':'csv', 'ignore_dynamic_events_check': False})
runs_df = pd.read_csv(BytesIO(off_ball_runs_csv))

idx = # pick a run
run = runs_df.iloc[idx] # series for a singular run

frame_start = run["frame_start"] # frame denoting the beginning of the run
frame_end = run["frame_end"]     # frame denoting the end of the run

frame_start_data = match_tracking_data[frame_start]
frame_end_data = match_tracking_data[frame_start]

player_positions_start = create_player_positions_2(frame_start_data)
player_positions_end = create_player_positions_2(frame_end_data)
```

## No-Run State Functions

The file NoRunStateFunctions.py contains the main no-run state function, `no_run_state`, and a bunch of intermediate and helper function. The functions are organized into two categories "helper" and "main" to help the reader identify what is used for what. The explanation of a few of the more important pieces:
- `move_runner`: This function adjust the position of the runner to the correct no-run state position.
- `position_score`: This function is used to adjust the *main defender*. For a given position it computes a position score (which we aim to optimize).
- `classify_attackers_marking`: This function determines which attackers are *unmarked*. This is important for the positioning of the second defender, because the second defender will tend to position themselves closer to unmarked players.
- `choose_seconday_defender_y`: Computes the new position of the secondary defender.
The output of no-run state is a DataFrame with the same set up as the DataFrame created using the function `create_player_positions`.

The below code is in addition to the code in the previous section and shows how to implement the no-run state functions. 

## Possession Value Functions

The file PossessionValueFunctions.py contains the following functions:
- `team_pitch_control_matrix`: Computes a matrix (of specified dimensions) denoting the pitch control of the indicated team. The code currently implements one of the earliest and simplest algorithms of pitch control found in [this paper](https://www.researchgate.net/publication/315166647_Physics-Based_Modeling_of_Pass_Probabilities_in_Soccer) by Spearman and Bayse.
- `team_pitch_control_runner_removed_matrix`: This function computes a pitch control matrix in the same manner as above, but does not include the runner in the computation.
- `p_val_no_runner`: Computes the *runner removed possession value*. Note that the xT matrix used should be compatible with the pitch control matrix that is computed above. That means the dimensions and orientation should be the same. For this project we created an xT matrix and saved it locally using Wyscout data... I have yet to upload that file though.

The following code gives an example of how to use these functions. This should be *in addition* to the code from the previous sections. 

```
from OrganizationFunctions import ball_position_2
from PossessionValueFunction import team_pitch_control_runner_removed_matrix
from 
```



















