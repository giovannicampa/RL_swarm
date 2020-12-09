#!/usr/bin/env python3

import os
import sys
import numpy as np
import random
import datetime

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


# - Getting input from the launch file
# Mode \in ["train", "test"]
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
                    learning_rate = linear_schedule(0.00001),
                    seed=1,
                    gamma=0.9)


        model.learn(total_timesteps=500000, callback = reward_callback)

        now = datetime.datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S").strip(" ").replace("/", "_").replace(":", "_").replace(" ", "_")
        model.save(current_path+"/models/model_"+dt_string)


    elif mode == "test":

        total_test_episodes = 100

        model = PPO.load(current_path+"/models/" + "models")
        env.test = True

        for _ in range(total_test_episodes): # Test the trained agent
            
            obs = env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)