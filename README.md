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

Note: The script takes one optional argument -- path to the config file: `python3 -m train -i <path_to_your_config>`, by deafult config from the repository is used.



