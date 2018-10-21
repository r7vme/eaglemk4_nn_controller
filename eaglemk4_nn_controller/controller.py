# Copyright (c) 2018 Roma Sokolkov
# MIT License

'''
Main controller loop.
'''

import time
import numpy as np

import gym
import eaglemk4_nn_controller.gym


class Controller:

    HZ = 20.0

    def __init__(self):
        self.env = gym.make('eaglemk4-v0')

    def run(self):
        obs = self.env.reset()
        while True:
            time.sleep(1.0 / self.HZ)
            obs, reward, done, info = self.env.step(np.array([0.5]))
            if done:
                self.env.reset()
            self.env.render()

    def close(self):
        return
