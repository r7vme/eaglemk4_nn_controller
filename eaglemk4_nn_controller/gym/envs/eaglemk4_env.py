# Copyright (c) 2018 Roma Sokolkov
# MIT License

'''
OpenAI gym interface to interact with the car.
'''

import gym
import numpy as np

from eaglemk4_nn_controller.ros import DrivingNode
from gym import spaces


class EagleMK4Env(gym.Env):
    """
    OpenAI gym environment for Eagle MK4.
    """
    def __init__(self):
        print("starting gym env")
        self.node = DrivingNode()

        # steering
        # TODO(r7vme): Add throttle
        self.action_space = spaces.Box(low=np.array([-1.0]),
                                       high=np.array([1.0]))

        # camera sensor data
        self.observation_space = spaces.Box(0, 255,
                                            self.node.get_sensor_size(),
                                            dtype=np.uint8)

    def step(self, action):
        self.node.take_action(action)
        observation, reward, done, info = self.node.observe()
        return observation, reward, done, info

    def reset(self):
        self.node.reset()
        observation, reward, done, info = self.node.observe()
        return observation

    def render(self, mode="human", close=False):
        pass

    def is_game_over(self):
        return self.node.is_game_over()
