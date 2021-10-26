import os
import sys

import numpy as np

from gym import spaces
from mapgen import Dungeon

sys.path.append(os.path.abspath("mapgen"))
os.environ["PYTHONPATH"] = os.path.abspath("mapgen")


class ModifiedDungeon(Dungeon):

    def __init__(self,
                 width: int = "width",
                 height: int = "height",
                 max_rooms: int = "max_rooms",
                 min_room_xy: int = "min_room_xy",
                 max_room_xy: int = "max_room_xy",
                 observation_size: int = "observation_size",
                 max_steps: int = "max_steps",
                 vision_radius: int = "vision_radius",
                 seed: int = "seed"
                 ):

        super().__init__(
            width=width,
            height=height,
            max_rooms=max_rooms,
            min_room_xy=min_room_xy,
            max_room_xy=max_room_xy,
            observation_size=observation_size,
            max_steps=max_steps,
            vision_radius=vision_radius
        )

        self.seed(seed)
        # cell types: UNK, FREE, OCCUPIED
        self.observation_space = spaces.Box(0, 1, [observation_size, observation_size, 3])
        self.action_space = spaces.Discrete(3)

    def _reward(self, info: dict) -> int:
        explored = info["new_explored"]
        reward = explored * (1 / max(np.log(info["step"]), 1))
        return reward

    def step(self, action):
        observation, reward, done, info = super().step(action)
        reward = self._reward(info)
        observation = observation[:, :, :-1]
        return observation, reward, done, info

    def reset(self):
        observation = super().reset()
        observation = observation[:, :, :-1]
        return observation
