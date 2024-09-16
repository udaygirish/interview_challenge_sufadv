# Transformers, robots in disguise

Hi! First of all, we really appreciate your interest in working with us! 
It's not lost on us that you're spending valuable time. We promise to make this fun.

There has recently been an explosion in imitation learning research and Transformers
have been key in some of these impressive policies. One piece of work we find interesting is
(Body Transformers)[https://arxiv.org/pdf/2408.06316], which adds an inductive bias to guide
the attention mechanism. We'd like for you to read the paper, implement the BoT-Mix version,
and run an inference loop (with untrained weights) on the given dataset of recorded expert demonstrations.

The goal of this model (as of most models in imitation learning) is to predict an action given
its current state. An action is the target state the robot will be commanded to go to.

## Context / Assumptions:

0. The embodiement is a 6-DoF robotic arm with a gripper, so 7-dim state and action spaces.
1. We'll use joint angles to represent the state of the robot.
2. There are no other sensors, such as cameras.
3. You will only need to predict one action at a time.

## Dataset:

The dataset is in zarr format and is structured as follows:

```
data/
    joints - timeseries of joint positions; 7-dim arrays representing the 6 joint angles (-3.14/+3.14, rads) and 1 gripper pose (0-1, open/close).
    action - timeseries of actions; same as the above, but the commands sent to the
meta/
    episode_end_idx - the last joints/action index in the timeseries for an episode.
```

If `episode_end_idx[0] = 34`, that indicates `joints[:34], action[:34]` belong to episode `0`.
The dataset contains 3 demonstrations of us puppeting the robot to pick up objects.

## Submitting

Feel free to make a branch on this repo to push your code to.