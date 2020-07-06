#!/usr/bin/env python3

import rospy
# import matplotlib.pyplot as plt
import numpy as np
import random
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import gym
from gym import spaces
import sys
sys.path.insert(1, '/home/giovanni/ROS_workspaces/RL_swarm/src/swarm_robots/src/envs')
from particle_env import ParticleEnv

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy



if __name__ == '__main__':


    rospy.init_node('swarm_node', anonymous=True)
    env = gym.make('Particle-v0')

    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log= "/home/giovanni/ROS_workspaces/RL_swarm/src/swarm_robots/src/tensorboard")
    model.learn(total_timesteps=10000)
    model.save("/home/giovanni/ROS_workspaces/RL_swarm/src/swarm_robots/src/models")




