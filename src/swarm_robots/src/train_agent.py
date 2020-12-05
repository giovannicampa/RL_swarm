#!/usr/bin/env python3

import sys
import numpy as np
import random

import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

import gym
from gym import spaces

from stable_baselines3 import DDPG
from stable_baselines3.ddpg.policies import MlpPolicy

# import rospkg
# rospack = rospkg.RosPack()

# # list all packages, equivalent to rospack list
# rospack.list() 

# # get the file path for rospy_tutorials
# package_path = rospack.get_path('swarm_robots')


sys.path.insert(1, '/home/giovanni/ROS_workspaces/RL_swarm/src/swarm_robots/src/envs')
from particle_env import ParticleEnvRL



if __name__ == '__main__':


    rospy.init_node('swarm_node', anonymous=True)
    env = gym.make('Particle-v0')

    # grid_search_params = {  "gamma": [0.99, 0.97, 0.95],
    #                         "lr": [0.001, 0.0001],
    #                         "network": [[126, 126], [256, 256], [512, 512]]}


    # for hyperparameter in hyperparameter_permutations:

    #     gamma_i = hyperparameter["gamma"]
    #     actor_lr_i = hyperparameter["lr"]
    #     critic_lr_i = hyperparameter["network"]

    model = DDPG(policy = 'MlpPolicy',
                env = env,
                verbose=1,
                tensorboard_log= "/home/giovanni/ROS_workspaces/RL_swarm/src/swarm_robots/src/tensorboard/",
                learning_rate = 0.0001,
                gamma=0.99)


    model.learn(total_timesteps=100000)
    model.save("/home/giovanni/ROS_workspaces/RL_swarm/src/swarm_robots/src/models")
