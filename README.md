# rl-sensory-exploration

## Task

Implementation of reinforcement learning algorithm for the sensory exploration of the rooms.

Simulation environment: [multi-room simulator](https://github.com/g-e0s/mapgen).

## Approcah

For the solution classical [PPO algorithm](https://github.com/ray-project/ray/blob/master/rllib/agents/ppo/ppo.py) from ray library was used. The reward function for the agent is defined as following:


```
reward = explored * (1 / max(np.log(step), 1))
```

Where:

```
- eplored: int # number of new cells explored on the current step
- step: int # the step number 
```

The idea behind the reward function is a simple time-discounting of new cells, so in order to maximize the reward, the agent should maximize the number of opened cells on early steps. 

## Usage

1. Clone the repository and install all the packages with `pip install -r requirements.txt`.
2. Login to `wandb` from cli and modify the parameters in the [`config.json`](config/config.json).
3. Run the script [train.py](train.py). 

Note: The script takes one optional argument – path to the config file, by deafult config from the repository is used. To run the script with custom config:

`python3 -m train -i <path_to_your_config>`

## Experiment setup

As for the experiment, the agent was trained with the following parameters:

**Parameter** | Value | 
---|---|
Number of itterations | 500
Entropy coefficient | 0.1
Lamda | 1 
Seed | 42
Max steps | 2000
Width & Height of a room | 20
Observation size | 11
Vision radius | 5
Activation function | `relu`
Cell types | UNK, FREE, OCCUPIED
Convolution layers | [[16, [3, 3], 2], [32, [3, 3], 2], [32, [3, 3], 1]]

## Results

Wanb report with the `mean_reward` and `mean_episode_len` charts can be found via the [link here](https://wandb.ai/daryoou_sh/sensory_exploration).
Gif example of the agent moves after 500 iteranion: 


<img src = gifs/iter_500.gif width="400" height="400">

## Conclusion

As the charts of the mean reward and the episode length show, the agent learns fast during the first 13 iterations, but stucks after reaching the result of ~1200 steps and doesn't aim to finish the exploration faster as time passes by. The problem might be, that the current reward function is to abstact. On the gif above it's clearly seen that agent stucks at the same cells and repeats the same moves in a loop. It might be a good idea, to take a trajectory into account, and stimulate agent not to repeat the same moves. 

