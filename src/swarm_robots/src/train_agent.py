#!/usr/bin/env python3

import os
import sys
import numpy as np
import random

import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

import gym
from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from typing import Callable

# Adding path to particle class
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, current_path+'/envs')
from particle_env import ParticleEnvRL


# Getting input from the launch file
try:
    mode = sys.argv[1]
except:
    mode = "test"

print(f"Loading in mode: {mode}")

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting rewards
    """

    def __init__(self, env):
        verbose = 0
        super(TensorboardCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        self.logger.record('reward', self.env.reward)
        self.logger.record('reward dist 2 goal', self.env.reward_distance_2_goal)
        self.logger.record('reward dist 2 particle', self.env.punishment_distance_2_particle)
        return True



def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


if __name__ == '__main__':


    rospy.init_node('swarm_node', anonymous=True)
    env = gym.make('Particle-v0')


    if mode == "train":
        reward_callback=TensorboardCallback(env = env)

        model = PPO(policy = 'MlpPolicy',
                    env = env,
                    verbose=1,
                    tensorboard_log= current_path+"/tensorboard/",
                    learning_rate = linear_schedule(0.0001),
                    seed=1,
                    gamma=0.97)


        model.learn(total_timesteps=100000, callback = reward_callback)
        model.save(current_path+"/models")


    elif mode == "test":
        model = PPO.load(current_path+"/models/" + "models")
        env.test = True
        
        for _ in range(100): # Test the trained agent 100 times
            obs = env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)