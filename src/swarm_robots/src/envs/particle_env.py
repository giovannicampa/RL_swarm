import gym
from gym import spaces
import numpy as np
import random

from geometry_msgs.msg import PoseArray
import rospy
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation

from gym.envs.registration import register
 


class ParticleEnv(gym.Env):
    """ Particle class
    
    It contains all attributes of a moving particle. Its motion is defined by a white box model
    """

    def __init__(self, particle_id = 999):
        
        super(ParticleEnv, self).__init__()

        # Motion parameters
        self.security_angle = 45*np.pi/180   # If abs(self.ang_dist_2_closest) is below this value, the particle changes direction
        self.security_distance = 10          # At this distance to the next particle, 
        self.len_half = 200                  # Half length of the square inside which the particles are move


        # Own attributes of the Particle class
        self.x = random.randint(-self.len_half, self.len_half)  # Position x
        self.y = random.randint(-self.len_half, self.len_half)  # Position x
        self.phi = 0                                            # Heading angle
        self.vel = 1                                            # Velocity along the heading angle
        self.closest_particles_x = np.inf                       # Distance to closest particle along x
        self.closest_particles_y = np.inf                       # Distance to closest particle along y
        self.dist_2_closest = np.inf                            # Distance to the closest particle
        self.collision_path = False                             # Whether a particle is in collision with another one
        self.particle_id = particle_id #random.randint(1,1000)                        # Particle Id





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
        self.calculate_distance_from_origin()


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
    

    def calculate_distance_from_origin(self):

        self.distance_from_origin = np.sqrt(self.x**2 + self.y**2) # Distance from the particle to (0,0)



# ---------------------------------------------------------------------------------------------------------

# Needed to interface with the gym environment
register(
    id='Particle-v0', 
    entry_point='particle_env:ParticleEnvRL',
)


class ParticleEnvRL(ParticleEnv):
    """ Class of the RL particle
    
    It is used to train the RL algorithm
    """


    def __init__(self):

        super(ParticleEnvRL, self).__init__()
        
        self.subscriber = rospy.Subscriber("/particles_positions", PoseArray, self.update_distances_to_particles)
    
        self.pub_marker = rospy.Publisher('particle_learning_marker', Marker, queue_size=10)
        self.pub_marker_velocity = rospy.Publisher('particle_learning_velocity', Marker, queue_size=10)

        self.n_steps = 300                                      # Nr training steps per episode
        self.steps = 0                                          # Current amount of steps

        # Modifying the attributes inherited from the gym.Env class [x, y, vel, phi, closest_particles_x, closest_particles_y]
        self.observation_space = spaces.Box(low= np.array([-self.len_half,-self.len_half,-1, -np.pi,-2*self.len_half,-2*self.len_half]),
                                            high=np.array([ self.len_half, self.len_half, 1, +np.pi, 2*self.len_half, 2*self.len_half]))
        self.action_space = spaces.Box(low= np.array([-np.pi/10,-1]),
                                       high=np.array([ np.pi/10, 1]))
        self.reward = 0


    def calculate_reward(self):
        """ Calculate reward for the current particle
        """
        self.calculate_distance_from_origin()

        return 1/self.dist_2_closest + 1/self.distance_from_origin


    def reset(self):
        """ Reset the states of the particle
        """
        self.steps = 0
        self.x = random.uniform(-self.len_half, self.len_half)
        self.y = random.uniform(-self.len_half, self.len_half)
        self.phi = 0.01
        self.vel = 1
        # self.update_distances_to_particles()
        self.collision_path = False

        return self.get_observation()


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

        return [self.x, self.y, self.vel, self.phi, self.closest_particles_x, self.closest_particles_y]


    # Update function
    def step(self, action):
        """ Step function required by the rl environment
        
        1. Gets action in desired range
        2. Apply action that modifies state
        3. Get observation of current state
        4. Calculate reward
        """

        # print(action)
        self.steps += 1

        self.publish_position()


        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.phi = self.phi + action[0]
        self.vel = action[1]
        
        # Bringing the angles in the right range
        if(self.phi < -np.pi):
            self.phi = self.phi + 2*np.pi
        
        if(self.phi > np.pi):
            self.phi = self.phi - 2*np.pi


        self.x = self.x + self.vel*np.cos(self.phi)
        self.y = self.y + self.vel*np.sin(self.phi)

        observation = self.get_observation()

        self.reward = self.calculate_reward()

        done = self.is_done()

        information = {"Finished":done}

        rospy.sleep(0.01)

        return observation, self.reward, done, information


    def update_distances_to_particles(self, msg):
        """ Calculates distance to closest particle

        Iterates over all particles and finds the closest one to the current particle
        """
        distance = np.inf

        for particle in msg.poses:

            distance_2_particle = np.linalg.norm(np.array([particle.position.x - self.x, particle.position.y - self.y]))
            
            if distance_2_particle < distance:
                distance = distance_2_particle
                self.closest_particles_x = particle.position.x
                self.closest_particles_y = particle.position.y

        self.dist_2_closest = distance


    def publish_position(self):
        """ Publishes the marker for position and heading orientation of the particle
        """

        # Position marker
        marker = Marker()
        marker.header.frame_id = "world"
        marker.id = 999
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 4
        marker.scale.y = 4
        marker.scale.z = 4
        marker.color.a = 1.0

        if(self.collision_path == False):
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
        else:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0

        marker.pose.orientation.w = 1.0
        marker.pose.position.x = float(self.x)
        marker.pose.position.y = float(self.y)
        marker.pose.position.z = 0
        

        # Velocity marker
        velocity_marker = Marker()
        velocity_marker.header.frame_id = "world"
        velocity_marker.id = 999
        velocity_marker.type = marker.ARROW
        velocity_marker.action = marker.ADD
        velocity_marker.scale.x = self.vel*10
        velocity_marker.scale.y = 1
        velocity_marker.scale.z = 1
        velocity_marker.color.a = 1.0
        velocity_marker.color.r = 1.0
        velocity_marker.color.g = 0.0
        velocity_marker.color.b = 0.0

        velocity_marker.pose.position = marker.pose.position

        quat = Rotation.from_euler("z", self.phi, degrees=False).as_quat()
        velocity_marker.pose.orientation.x = quat[0]
        velocity_marker.pose.orientation.y = quat[1]
        velocity_marker.pose.orientation.z = quat[2]
        velocity_marker.pose.orientation.w = quat[3]


        self.pub_marker.publish(marker)
        self.pub_marker_velocity.publish(velocity_marker)
