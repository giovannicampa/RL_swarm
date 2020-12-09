#!/usr/bin/env python3

# In this script we generate the environment of "stupid" particles, that the agent will interact with
from scipy.spatial.transform import Rotation
import rospy
from geometry_msgs.msg import PoseArray, PoseStamped
import numpy as np
import random
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import gym
from gym import spaces
import sys
sys.path.insert(1, '/home/giovanni/ROS_workspaces/RL_swarm/src/swarm_robots/src/envs')
from particle_env import ParticleEnv


pub_marker = rospy.Publisher('particles_markers', MarkerArray, queue_size=10)
pub_marker_velocity = rospy.Publisher('particles_velocity', MarkerArray, queue_size=10)
pub_poses = rospy.Publisher('particles_positions', PoseArray, queue_size=10)


# Getting input from the launch file
try:
    nr_particles = int(sys.argv[1])
    print(f"Nr input particles from launch file: {nr_particles}")
except:
    nr_particles = 10


# Generate dumb particles
particle_list = []
for i in range(nr_particles):

    p = ParticleEnv(particle_id = i)

    particle_list.append(p)


# Calculating distance between two particles
def distance_particles(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def update_world():
    

    while not rospy.is_shutdown():

        marker_array = MarkerArray()
        velocity_marker_array = MarkerArray()
        pose_array = PoseArray()


        # Updating all the particles
        for p in particle_list:

            # Checking if there are any close by particles
            distance_list = []
            for q in particle_list:

                distance_list.append(distance_particles(p,q))


            # Setting distance to self particle to 100 and finding the closest one
            distance_list[distance_list.index(0)] = 100
            p.dist_2_closest = distance_list[np.argmin(distance_list)]
            
            p.closest_particles_x = particle_list[np.argmin(distance_list)].x
            p.closest_particles_y = particle_list[np.argmin(distance_list)].y
            
            p.calc_ang_dist_2_closest()
            p.update_states_wb()

            # Position marker
            marker = Marker()
            marker.header.frame_id = "world"
            marker.id = p.particle_id
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 4
            marker.scale.y = 4
            marker.scale.z = 4
            marker.color.a = 1.0

            if(p.collision_path == False):
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0

            marker.pose.orientation.w = 1.0
            marker.pose.position.x = float(p.x)
            marker.pose.position.y = float(p.y)
            marker.pose.position.z = 0
            marker_array.markers.append(marker)
            

            # Velocity marker
            velocity_marker = Marker()
            velocity_marker.header.frame_id = "world"
            velocity_marker.id = p.particle_id
            velocity_marker.type = marker.ARROW
            velocity_marker.action = marker.ADD
            velocity_marker.scale.x = p.vel*10
            velocity_marker.scale.y = 1
            velocity_marker.scale.z = 1
            velocity_marker.color.a = 1.0
            velocity_marker.color.r = 1.0
            velocity_marker.color.g = 0.0
            velocity_marker.color.b = 0.0

            velocity_marker.pose.position = marker.pose.position

            quat = Rotation.from_euler("z", p.phi, degrees=False).as_quat()
            velocity_marker.pose.orientation.x = quat[0]
            velocity_marker.pose.orientation.y = quat[1]
            velocity_marker.pose.orientation.z = quat[2]
            velocity_marker.pose.orientation.w = quat[3]
            velocity_marker_array.markers.append(velocity_marker)

            # Pose array
            pose_array.poses.append(marker.pose)


        pub_marker.publish(marker_array)
        pub_marker_velocity.publish(velocity_marker_array)
        pub_poses.publish(pose_array)
        rospy.sleep(0.01)


# for each episode, we test the robot for nsteps
while True:
    rospy.init_node('swarm_node', anonymous=True)
    update_world()
