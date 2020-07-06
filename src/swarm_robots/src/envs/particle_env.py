import gym
from gym import spaces
import numpy as np
import random

from gym.envs.registration import register
 
register(
    id='Particle-v0', 
    entry_point='particle_env:ParticleEnv',
)



class ParticleEnv(gym.Env):

    # Particle characteristics
    security_angle = 45*np.pi/180
    security_distance = 10
    border_x = 500
    border_y = 500
    angle_addition_low = 0.3
    angle_addition_high = 1

    # Training characteristics
    n_episodes = 1000
    n_steps = 300


    # Init method
    # , x, y, phi, vel, closest_particles_x, closest_particles_y, id_nr = 1000, lim_x = 100, lim_y = 100, n_episodes = 1000, n_steps = 300
    def __init__(self):
        
        # Inheriting from the superclass of Particle
        super(ParticleEnv, self).__init__()

        x = random.randint(-250,250)
        y = random.randint(-250,250)
        phi = 0
        vel = 1
        closest_particles_x = np.inf
        closest_particles_y = np.inf
        id_nr = random.randint(1,1000)
        lim_x = 100
        lim_y = 100
        n_episodes = 1000
        n_steps = 300

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
        self.steps = 0
        self.lim_x = lim_x
        self.lim_y = lim_y
        self.n_episodes = n_episodes
        self.n_steps = n_steps


        # Modifying the attributes inherited from the gym.Env class [x, y, vel, phi, closest_particles_x, closest_particles_y]
        self.observation_space = spaces.Box(low=np.array([-lim_x, -lim_y, 0, -np.pi, 0, 0]), high=np.array([lim_x, lim_y, 2, +np.pi, lim_y*2, lim_x*2]))
        self.action_space = spaces.Box(low = np.array([-np.pi/10, -0.2]), high=np.array([np.pi/10, 0.2]))


    # Update function
    def step(self, action):

        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.phi = self.phi + action[0]
        self.vel = self.vel + action[1]


        observation = self.get_observation()

        reward = self.calculate_reward()

        done = self.is_done()

        information = []

        return observation, reward, done, information


    def calc_ang_dist_2_closest(self):
        self.ang_dist_2_closest = np.arctan2((self.y - self.closest_particles_y), (self.x - self.closest_particles_x))

    # Hard coded behaviour model
    def update_states(self):

        # Check closeness to other particles
        if((abs(self.ang_dist_2_closest) < self.security_angle) and (self.dist_2_closest < self.security_distance)):
            
            if(self.ang_dist_2_closest < self.security_angle):
                self.phi = self.phi - random.uniform(0.3, 1)

            elif(self.ang_dist_2_closest > self.security_angle):
                self.phi = self.phi + random.uniform(0.3, 1) 

            self.collision_path = True
            # print("Particle in collision course!")

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

        return 1/self.dist_2_closest + 1/self.distance_from_origin


    def reset(self):
        self.x = random.uniform(-10,10)
        self.y = random.uniform(-10,10)
        self.phi = 0.01
        self.vel = 1
        self.closest_particles_x = np.inf
        self.closest_particles_y = np.inf
        self.dist_2_closest = np.sqrt((self.x - self.closest_particles_x)**2 + (self.y - self.closest_particles_y)**2)
        self.collision_path = False


    def is_done(self):
        if(self.steps >= self.n_steps):
            return True
        else:
            return False


    def get_observation(self):
        
        return [self.x,self.y, self.vel, self.phi, self.closest_particles_x, self.closest_particles_y]