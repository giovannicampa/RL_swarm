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


    def __init__(self):
        
        super(ParticleEnv, self).__init__()

        # Motion parameters
        self.security_angle = 45*np.pi/180   # If abs(self.ang_dist_2_closest) is below this value, the particle changes direction
        self.security_distance = 10          # At this distance to the next particle, 
        self.len_half = 200                  # Border along x of the area where the particles can move


        # Own attributes of the Particle class
        self.x = random.randint(-self.len_half, self.len_half)  # Position x
        self.y = random.randint(-self.len_half, self.len_half)  # Position x
        self.phi = 0                                            # Heading angle
        self.vel = 1                                            # Velocity along the heading angle
        self.closest_particles_x = np.inf                       # Distance to closest particle along x
        self.closest_particles_y = np.inf                       # Distance to closest particle along y
        self.dist_2_closest = np.sqrt(
            (self.x - self.closest_particles_x)**2 +\
            (self.y - self.closest_particles_y)**2)             # Distance to the closest particle
        self.collision_path = False                             # Whether a particle is in collision with another one
        self.id = random.randint(1,1000)                        # Particle Id
        self.n_episodes = 1000                                  # Nr training episodes
        self.n_steps = 300                                      # Nr training steps per episode
        self.steps = 0                                          # Current amount of steps

        # Modifying the attributes inherited from the gym.Env class [x, y, vel, phi, closest_particles_x, closest_particles_y]
        self.observation_space = spaces.Box(low= np.array([-self.len_half,-self.len_half, 0, -np.pi, 0, 0]),
                                            high=np.array([ self.len_half, self.len_half, 2, +np.pi, self.len_half*2, self.len_half*2]))
        self.action_space = spaces.Box(low= np.array([-np.pi/10,-0.2]),
                                       high=np.array([ np.pi/10, 0.2]))


    # Update function
    def step(self, action):
        """ Step function required by the rl environment
        
        1. Gets action in desired range
        2. Apply action that modifies state
        3. Get observation of current state
        4. Calculate reward
        """

        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.phi = self.phi + action[0]
        self.vel = self.vel + action[1]

        observation = self.get_observation()

        reward = self.calculate_reward()

        done = self.is_done()

        information = []

        return observation, reward, done, information


    def calc_ang_dist_2_closest(self):
        """ Angular distance to next particle

        Defined as the angle between the heading direction of the particle and the line connecting the particle itself and the closest one
        """
        
        self.ang_dist_2_closest = np.arctan2((self.y - self.closest_particles_y), (self.x - self.closest_particles_x))


    def update_states_wb(self):
        """ Update position of the particle with the white box method

        The particle is moved in a collision free direction
        """

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


        angle_2_particle = np.arctan2(self.y, self.x)
        self.distance_from_origin = np.sqrt(self.x**2 + self.y**2) # Distance from the particle to (0,0)


        # Radius of a circle with centre in (0,0) that inscribes the square with length self.len_half
        radius = np.sqrt((self.len_half/1.4)**2 + (self.len_half/1.4)**2)


        # If particle is too far away
        if((self.distance_from_origin > radius) & (abs(self.phi - angle_2_particle) < np.pi/2)):

            self.phi = self.phi + np.sign(self.phi - angle_2_particle) * random.uniform(0,0.7)


        # Adding random variation to the angle
        else:
            self.phi = self.phi + random.uniform(-0.4, 0.4)


        # Bringing the angles in the right range
        if(self.phi < -np.pi):
            self.phi = self.phi + 2*np.pi
        
        if(self.phi > np.pi):
            self.phi = self.phi - 2*np.pi

        # Update position of particle
        self.x = self.x + np.cos(self.phi)*self.vel
        self.y = self.y + np.sin(self.phi)*self.vel
    

    def calculate_reward(self):
        """ Calculate reward for the current particle
        """
        return 1/self.dist_2_closest + 1/self.distance_from_origin


    def reset(self):
        """ Reset the states of the particle
        """

        self.x = random.uniform(-self.len_half, self.len_half)
        self.y = random.uniform(-self.len_half, self.len_half)
        self.phi = 0.01
        self.vel = 1
        self.closest_particles_x = np.inf
        self.closest_particles_y = np.inf
        self.dist_2_closest = np.sqrt((self.x - self.closest_particles_x)**2 + (self.y - self.closest_particles_y)**2)
        self.collision_path = False


    def is_done(self):
        """ Check if simulation is over
        
        Return true if the nr of steps is bigger thant the maximum allowed nr of steps per episode
        """

        if(self.steps >= self.n_steps):
            return True
        else:
            return False


    def get_observation(self):
        """ Return the current state of the particle
        """
        
        return [self.x,self.y, self.vel, self.phi, self.closest_particles_x, self.closest_particles_y]