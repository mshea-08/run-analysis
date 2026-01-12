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

$$PV^{-r}(s) = \sum_{p \in \mathcal{G}} xT(p) \cdot PC^{-r}(s;p)$$

where $\mathcal{G}$ is a discretization of the pitch and $PC^{-r}(s)$ is the *runner removed pitch control*.

## Support Score, S(r)

The support score takes a radial neighborhood of the ball carrier and computes a weighted sum of the pitch control in this area. After that it looks at immediate pressure on the ball carrier and scales the score inversely with the pressure (more pressure $\implies$ smaller score).

## Understanding the Component Values

For all values, larger score is better. The potential reception score is always non-negative, however both the space manipulation score and support score may be negative. 

# No-Run State

The key to the space manipulation and support scores is the generation of the **no-run counterfactual state**. The idea behind the no-run state is to produce a state where most players behave in the same way, however the runner does not attempt the run. Anyone whose behavior was *influenced* by the run should behave appropriately given the new circumstances. 

Below we will outline the main steps to the no-run state, although we will omit some detail. For a more detailed accounting see, *upload paper*.

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

## No Run State

The no run state is a counterfactual state that assumes the runner did not actually make the run. The no-run state assumes most players are at their end of run positions. We only alter the location of the runner and any *influenced defenders*.

First, the runners location is frozen according to their initial location (at the start of the run). The runners location is only further adjusted if one of the following is true:
- The frozen runner becomes the last defender in the final state. In this case we move them up to be equal to their defensive line height.
- (*Not Included Yet*) The frozen runner gets too close to the ball carrier. We assume the runner stays a minimum of 4 m away from the ball carrier.
- (*Not Included Yet*) The frozen runner is offside in the final state. We adjust the runner to stay onside.

Next, we compute the *main defender* at the initial frame. Note that their may be no main defender. To determine the main defender, we look at all defenders within 8 m of the runner. We then compute which of these defenders is closest to the *optimal position*. The *optimal position* is 2 m away from the runner on the line from the runner to the goal. For the main defender we apply the following:
- Compute a region of potential placement. This region is formed from their initial and ending locations.
- Within that region we look to maximize a position_score. The position score consists of the following elements, which are weighted differently in different regions of the pitch,
  - Optimal positioning: Determines the ideal defensive positioning (independent of the other players). It takes into account where the goal is and where on the field the runner is.
  - Pass lane blocking: Determines the best position to block a pass to the player.
  - Overall defensive shape: If the player is a defender, this measures how much they align with the defensive line height. Additionally, for all positions, this measures how spaced out they are in relation to their teammates. 

# Other Technical Descriptions 

*Under construction.*
