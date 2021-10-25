from PIL import Image
from typing import Tuple

import numpy as np


def save_image(data: np.array, size: Tuple = (500, 500)) -> None:
    Image.fromarray(data).convert('RGB').resize(size, Image.NEAREST).save('tmp.png')


def quantize_frame(data: np.array, size: Tuple = (500, 500)):
    return Image.fromarray(data).convert('RGB').resize(size, Image.NEAREST).quantize()


def save_gif(n_actions: int, gif_name: str, agent, env, observation) -> None:

    frames = []

    for _ in range(n_actions):
        action = agent.compute_single_action(observation)
        data = env._map.render(env._agent)
        frame = quantize_frame(data)
        frames.append(frame)

        observation, reward, done, info = env.step(action)
        if done:
            break

    frames[0].save(gif_name, save_all=True, append_images=frames[1:], loop=0, duration=1000 / 60)
