#!/usr/bin/env python

import rospy
import matplotlib.pyplot as plt
import numpy as np
import random
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import gym
from gym import spaces

from gym.envs.swarm_robotics import Particle


pub = rospy.Publisher('particles_markers', MarkerArray, queue_size=10)
rospy.init_node('swarm_node', anonymous=True)






# Calculating distance between two particles
def distance_particles(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)



# Generate particles
particle_list = []

for i in range(200):

    p = Particle(random.randint(-250,250),random.randint(-250,250), 0, 1, np.inf, np.inf, i)
    particle_list.append(p)



def movement():
    

    while not rospy.is_shutdown():

        markerArray = MarkerArray()

        # Updating all the particles
        for p in particle_list:


            # Checking if there are any close particles
            distance_list = []
            for q in particle_list:

                distance_list.append(distance_particles(p,q))


            # Setting distance to self particle to 100 and finding the closest one
            distance_list[distance_list.index(0)] = 100
            p.dist_2_closest = distance_list[np.argmin(distance_list)]
            
            p.closest_particles_x = particle_list[np.argmin(distance_list)].x
            p.closest_particles_y = particle_list[np.argmin(distance_list)].y
            
            p.calc_ang_dist_2_closest()
            p.update_states()


            marker = Marker()
            marker.header.frame_id = "world"
            marker.id = p.id
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 2
            marker.scale.y = 2
            marker.scale.z = 2
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

            markerArray.markers.append(marker)


        print(markerArray.markers[1].pose.position.x)


        pub.publish(markerArray)
        rospy.sleep(0.01)




if __name__ == "__main__":

    movement()