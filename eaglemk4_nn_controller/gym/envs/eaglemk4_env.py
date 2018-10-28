# Copyright (c) 2018 Roma Sokolkov
# MIT License

'''
OpenAI gym interface to interact with the car.

Hijacked with VAE.

- Use Z vector as observation space.
- Store raw images in VAE buffer.

Problem that DDPG already well implemented in stable-baselines
and VAE integration will require full reimplementation of DDPG
codebase. Instead we hijack VAE into gym environment.
'''

import gym
import numpy as np

from eaglemk4_nn_controller.ros_node import DrivingNode
from gym import spaces


class EagleMK4Env(gym.Env):
    """
    OpenAI gym environment for Eagle MK4.
    """
    def __init__(self):
        self.z_size = 512

        print("starting gym env")
        self.node = DrivingNode()

        # steering
        # TODO(r7vme): Add throttle
        self.action_space = spaces.Box(low=np.array([-1.0]),
                                       high=np.array([1.0]))

        # VAE latent vector data
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                            high=np.finfo(np.float32).max,
                                            shape=(1, self.z_size),
                                            dtype=np.float32)

    def step(self, action):
        self.node.take_action(action)
        observation, reward, done, info = self._observe()
        return observation, reward, done, info

    def reset(self):
        self.node.reset()
        observation, reward, done, info = self._observe()
        return observation

    def render(self, mode="human", close=False):
        return

    def is_game_over(self):
        return self.node.is_game_over()

    def is_training(self):
        return True if self.node.task else False

    def is_testing(self):
        return not self.is_training()

    def is_autopilot(self):
        return True if self.node.autopilot else False

    def _observe(self):
        observation, reward, done, info = self.node.observe()
        # Solves chicken-egg problem as gym calls reset before we call set_vae.
        if not hasattr(self, "vae"):
            return np.zeros(self.z_size), reward, done, info
        # Store image in VAE buffer.
        self.vae.buffer_append(observation)
        return self.vae.encode(observation), reward, done, info

    def set_vae(self, vae):
        self.vae = vae
