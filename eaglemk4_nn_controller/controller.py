# Copyright (c) 2018 Roma Sokolkov
# MIT License

'''
Main controller loop.
'''

import gym
import eaglemk4_nn_controller.gym


class Controller:
    '''
    '''
    def __init__(self):
        self.env = gym.make('eaglemk4-v0')

    def run(self):
        obs = self.env.reset()
        while True:
            print("go go go")
            obs, reward, done, info = self.env.step(1.0)
            if done:
                print("done")
                self.env.reset()
            self.env.render()

    def close(self):
        return
