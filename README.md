# Overview 

The goal of this project is to create a metric that evaluates the quality of off-ball runs made in a game of soccer. In particular, most means of measurement rely on the runner actually recieving a pass (and, secondarily, on the quality of that pass). We aim to produce a meaningful value that is independent of the result of the play sequence. 

The off-ball run value, ORV, will rely on the following soccer principles:
1. **Potential quality of the pass created.** This will consider both the feasibility of the intended pass as well as the danger created from the resulting pass.
2. **Space manipulation caused by the run.** This will consider the space created for other players due to the action of the run.
3. **Risk.** This will evaluate the risk of exposure if the ball is turned over.

This project is currently in progress. On this page we will highlight the main mathematical principles as well as discuss the current files in the repository. 

## Working Table of Contents

[Mathematical Description of the Model](#mathematical-description)
[Current Code Available](#current-code-available)
[Other Technical Descriptions](#other-technical-descriptions)

## Data Availability

Tracking data has been provided by SkillCorner. For freely available tracking data check out [sample data](https://github.com/metrica-sports/sample-data) provided by Metrica. Note that the code provided here most likely needs to be edited to work for other data sources. 

# Mathematical Description

## Revieved Run Value

This part will analyze the quality of the pass created by the run. *Under construction.*

## Space Manipulation

This part will analyze the amount of space opened up by the run. 

Let $s$ denote the state of play after the run has ended and let $\hat{s}$ denote the counterfactual state where no off-ball run is attempted. Essentially, the space manipulation factor is the difference, 

$$EPV(s) - EPV(\hat{s})$$

where $EPV$ is the *expected possession value* of the states. 

### Counterfactual Run State

The key to this metric is the generation of the no off-ball run counterfactual state. The idea behind the no-run counter factual state is to produce a state where most players behave in the same way, however the runner does not attempt the run and anyone whose behavior was immediately influenced by the run should behave appropriately given the new circumstances. 

## Risk

This part will analyze the amount of potential risk. *Under construction.* 

# Current Code Available

## Pitch Control 

### Definition

Pitch control is a model that implements basic principles from physics to compute which player/team is most likely to control the ball if it is moved to a given location on the pitch. The code currently implements one of the earliest and simplest algorithms of pitch control found in [this paper](https://www.researchgate.net/publication/315166647_Physics-Based_Modeling_of_Pass_Probabilities_in_Soccer) by Spearman and Bayse. 

### Implementation Using the Code

In TrackingFunctions.py there are two main functions concerning the pitch control of a given state: team_pitch_control_mat() and player_pitch_control_mat(). The functions take the following inputs:
- the location of the ball (ball_x,ball_y),
- a DataFrame consisting of the locations and velocities of all players (player_positions),
- the size of the pitch (pitch_length, pitch_width),
- the size of the grid spaces used (dx, dy),
- and the team or player whos pitch control is to be computed (team_id or player_id).

The output is a matrix of probabilities, where the rows are indexed bottom -> top and the columns are index left -> right. 

*Under construction.*

# Other Technical Descriptions 

*Under construction.*
