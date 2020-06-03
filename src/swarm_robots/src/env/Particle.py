import gym
from gym import spaces
import numpy as np
import random



class Particle(gym.Env):

    security_angle = 45*np.pi/180
    security_distance = 10
    border_x = 1250
    border_y = 1250
    angle_addition_low = 0.3
    angle_addition_high = 1


    # Init method
    def __init__(self, x, y, phi, vel, closest_particles_x, closest_particles_y, id_nr):
        
        # Inheriting from the superclass of Particle
        super(Particle, self).__init__()

        # Own attributes of the Particle class
        self.x = x
        self.y = y
        self.phi = phi
        self.vel = vel
        self.closest_particles_x = closest_particles_x
        self.closest_particles_y = closest_particles_y
        self.id = id_nr
        self.dist_2_closest = np.sqrt((self.x - self.closest_particles_x)**2 + (self.y - self.closest_particles_y)**2)
        self.collision_path = False


        # Modifying the attributes inherited from the gym.Env class
        self.observation_space = spaces.Box(low=np.array([-1.57, 0]), high=np.array([1.57, 10]))
        
    
    def step(self, action):
        self.phi = self.phi + action[0]
        self.vel = self.vel + action[1]


    def calc_ang_dist_2_closest(self):
        self.ang_dist_2_closest = np.arctan2((self.y - self.closest_particles_y), (self.x - self.closest_particles_x))


    def update_states(self):

        # Check closeness to other particles
        if((abs(self.ang_dist_2_closest) < self.security_angle) and (self.dist_2_closest < self.security_distance)):
            
            if(self.ang_dist_2_closest < self.security_angle):
                self.phi = self.phi - random.uniform(0.3, 1)

            elif(self.ang_dist_2_closest > self.security_angle):
                self.phi = self.phi + random.uniform(0.3, 1) 

            self.collision_path = True
            print("Particle in collision course!")

        else:
            self.collision_path = False


        # random.uniform(self.angle_addition_low, self.angle_addition_high)*
        angle_2_particle = np.arctan2(self.y, self.x)
        distance_from_origin = np.sqrt(self.x**2 + self.y**2)
        self.distance_from_origin = distance_from_origin


        radius = np.sqrt((self.border_x/1.4)**2 + (self.border_y/1.4)**2)


        # If particle is too far away
        if((distance_from_origin > radius) & (abs(self.phi - angle_2_particle) < np.pi/2)):

            self.phi = self.phi + np.sign(self.phi - angle_2_particle) * random.uniform(0,0.7)


        # Adding random variation to the angle
        else:
            self.phi = self.phi + random.uniform(-0.4, 0.4)


        # Bringing the angles in the right range
        if(self.phi < -np.pi):
            self.phi = self.phi + 2*np.pi
        
        if(self.phi > np.pi):
            self.phi = self.phi - 2*np.pi


        self.x = self.x + np.cos(self.phi)*self.vel
        self.y = self.y + np.sin(self.phi)*self.vel
    

    def calculate_reward(self):

        return 1/self.dist_2_closest + 1/self.distance_from_origin + self.vel