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

## Identify Influenced Defenders 



# Current Code Available

*Under construction.*

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

