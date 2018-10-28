# Copyright (c) 2018 Roma Sokolkov
# MIT License

"""
DDPGWithVAE inherits DDPG from stable-baselines
and reimplements learning method.
"""

import time

import numpy as np
from mpi4py import MPI

from stable_baselines import logger
from stable_baselines.ddpg.ddpg import DDPG


class DDPGWithVAE(DDPG):
    """
    Modified learn method from stable-baselines

    - Stop rollout on episode done.
    - More verbosity.
    - Add VAE optimization step.
    """
    def learn(self, callback=None, vae=None, do_ddpg_training=True):
        rank = MPI.COMM_WORLD.Get_rank()
        # we assume symmetric actions.
        assert np.all(np.abs(self.env.action_space.low) == self.env.action_space.high)

        self.episode_reward = np.zeros((1,))
        with self.sess.as_default(), self.graph.as_default():
            # Prepare everything.
            self._reset()
            episode_reward = 0.
            episode_step = 0
            episodes = 0
            step = 0
            total_steps = 0

            start_time = time.time()

            actor_losses = []
            critic_losses = []

            obs = self.env.reset()
            # Rollout one episode.
            while True:
                # Predict next action.
                action, q_value = self._policy(obs, apply_noise=True, compute_q=True)
                print(action)
                assert action.shape == self.env.action_space.shape

                # Execute next action.
                if rank == 0 and self.render:
                    self.env.render()
                new_obs, reward, done, _ = self.env.step(action * np.abs(self.action_space.low))

                step += 1
                total_steps += 1
                if rank == 0 and self.render:
                    self.env.render()
                episode_reward += reward
                episode_step += 1

                # Book-keeping.
                # Store transition only, if we started DDPG optimization.
                if do_ddpg_training:
                    self._store_transition(obs, action, reward, new_obs, done)
                obs = new_obs
                if callback is not None:
                    callback(locals(), globals())

                if done:
                    print("episode finished. Reward: ", episode_reward)
                    # Episode done.
                    episode_reward = 0.
                    episode_step = 0
                    episodes += 1

                    self._reset()
                    # Finish rollout on episode finish.
                    break

            print("Rollout finished")

            # Train VAE.
            train_start = time.time()
            print("VAE training: Started...")
            vae.optimize()
            print("VAE training duration:", time.time() - train_start)

            # Train DDPG.
            actor_losses = []
            critic_losses = []
            train_start = time.time()
            if do_ddpg_training:
                print("DDPG training: Started...")
                for t_train in range(self.nb_train_steps):
                    critic_loss, actor_loss = self._train_step(0, None, log=t_train == 0)
                    critic_losses.append(critic_loss)
                    actor_losses.append(actor_loss)
                    self._update_target_net()
                print("DDPG training duration:", time.time() - train_start)
