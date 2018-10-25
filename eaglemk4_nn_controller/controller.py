# Copyright (c) 2018 Roma Sokolkov
# MIT License

'''
Main controller loop.
'''

import gym
import os
import numpy as np
import time
import rospy

import eaglemk4_nn_controller.gym
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from eaglemk4_nn_controller.models.ddpg_with_vae import DDPGWithVAE as DDPG
from eaglemk4_nn_controller.models.vae.controller import VAEController


class Controller:

    HZ = 20.0

    PATH_MODEL_DDPG = "ddpg.pkl"
    PATH_MODEL_VAE = "vae.json"

    def __init__(self):
        # Make sure model path exists.
        self.model_path = rospy.get_param('nn_controller/model_path',
                                          '/data/models')
        if not os.path.exists(self.model_path):
            raise Exception(self.model_path + ' does not exist.')
        self.ddpg_path = os.path.join(self.model_path, self.PATH_MODEL_DDPG)
        self.vae_path = os.path.join(self.model_path, self.PATH_MODEL_VAE)

        self.env = gym.make('eaglemk4-v0')

        # Initialize VAE model and add it to gym environment.
        # VAE does image post processing to latent vector and
        # buffers raw image for future optimization.
        self.vae = VAEController(buffer_size=100,
                                 image_size=(144, 176, 3),
                                 batch_size=32,
                                 epoch_per_optimization=1)
        self.env.unwrapped.set_vae(self.vae)

        # Don't run anything until human approves.
        print("EagleMK4 Neural Network Controller loaded!")
        print("1. Press triangle to select task.")
        print("2. Press right bumper to start task.")
        self._wait_autopilot()

        # Run infinite loop.
        self.run()

    def run(self):
        while True:
            # We have only two tasks at the moment:
            # - train - runs online training.
            # - test - evaluates trained models.
            if self.env.unwrapped.is_training():
                print("Training...")
                # TODO: Allow precompiled models.
                self.ddpg = self._init_ddpg()

                episode = 0
                skip_episodes = 10
                do_ddpg_training = False
                while self.env.unwrapped.is_training():
                    if episode > skip_episodes:
                        do_ddpg_training = True
                    self.ddpg.learn(vae=self.vae,
                                    do_ddpg_training=do_ddpg_training)
                    episode += 1

                # Finally save model files.
                self.ddpg.save(self.ddpg_path)
                self.vae.save(self.vae_path)
            elif self.env.unwrapped.is_testing():
                print("Testing...")

                if self._any_precompiled_models():
                    # Load models and run testing episodes.
                    self.ddpg = DDPG.load(self.ddpg_path)
                    self.vae.load(self.vae_path)
                    while self.env.unwrapped.is_testing():
                        self.run_test_episode()
                else:
                    print("No precompiled models found.",
                          "Please run training by pressing triange (switch task)",
                          "and press-keep right bumper (allow autopilot)."
                          "Unpressing right bumper stops the episode.")
                    while self.env.unwrapped.is_testing():
                        time.sleep(1.0 / self.HZ)

    def run_testing_episode(self):
        # Reset will wait for autopilot mode ("right bumper" pressed).
        obs = self.env.reset()
        while True:
            time.sleep(1.0 / self.HZ)
            action, _states = self.ddpg.predict(obs)
            obs, reward, done, info = self.env.step(np.array([0.5]))
            if done:
                print("Testing episode finished.")
                return

    def _any_precompiled_models(self):
        if os.path.exists(self.ddpg_path) and \
           os.path.exists(self.vae_path):
            return True
        return False

    def _init_ddpg(self):
        # the noise objects for DDPG
        n_actions = self.env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),
                theta=float(0.6) * np.ones(n_actions),
                sigma=float(0.2) * np.ones(n_actions)
                )

        return DDPG(LnMlpPolicy,
                    self.env,
                    verbose=1,
                    batch_size=64,
                    clip_norm=5e-3,
                    gamma=0.9,
                    param_noise=None,
                    action_noise=action_noise,
                    memory_limit=1000,
                    nb_train_steps=3000,
                    )

    # Make sure user pressed autopilot button.
    def _wait_autopilot(self):
        while True:
            time.sleep(1 / self.HZ)
            if self.env.unwrapped.is_autopilot():
                return

    def close(self):
        return
