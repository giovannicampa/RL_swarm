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
        self.x = 0                           # Position x
        self.y = 0                           # Position y
        self.phi = 0                         # Heading angle
        self.vel = 1                         # Velocity along the heading angle
        self.closest_particles_x = np.inf    # Distance to closest particle along x
        self.closest_particles_y = np.inf    # Distance to closest particle along y
        self.dist_2_closest = np.inf         # Distance to the closest particle
        self.collision_path = False          # Whether a particle is in collision with another one
        self.particle_id = particle_id       # Particle Id
        self.goal_x = 0                      # Position goal x
        self.goal_y = 0                      # Position goal y




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
    
        self.pub_marker_position = rospy.Publisher('particle_learning_marker', Marker, queue_size=10)
        self.pub_marker_velocity = rospy.Publisher('particle_learning_velocity', Marker, queue_size=10)
        self.pub_marker_goal = rospy.Publisher('goal_marker', Marker, queue_size=10)

        self.n_steps = 300                                      # Nr training steps per episode
        self.steps = 0                                          # Current amount of steps

        # [x, y, vel, phi, closest_particles_x, closest_particles_y, goal_x, goal_y, obstacle_x, obstacle_y]
        self.observation_space = spaces.Box(low= np.array([-self.len_half,-self.len_half,-self.len_half,-self.len_half,-self.len_half,-self.len_half]),
                                            high=np.array([ self.len_half, self.len_half, self.len_half, self.len_half, self.len_half, self.len_half]))

        self.action_space = spaces.Box(low= np.array([-5,-5]),
                                       high=np.array([ 5, 5]))
        
        self.distance_from_goal = 0
        self.distance_from_goal_previous = 0
        self.reward = 0


    def calculate_reward(self):
        """ Calculate reward for the current particle
        """

        self.punishment_distance_2_particle = -1 if self.dist_2_closest < 1 else -1/self.dist_2_closest
        self.reward_distance_2_goal = 10*(self.distance_from_goal_previous - self.distance_from_goal)/(self.distance_from_goal) if self.distance_from_goal > 1 else 10

        return self.reward_distance_2_goal + self.punishment_distance_2_particle


    def reset(self):
        """ Reset the states of the particle
        """
        self.steps = 0
        self.x = random.uniform(-self.len_half, self.len_half)
        self.y = random.uniform(-self.len_half, self.len_half)
        self.x_previous = self.x + random.uniform(-1,1)
        self.y_previous = self.y + random.uniform(-1,1)
        self.collision_path = False
        self.generate_goal()
        self.distance_from_goal_previous = self.distance_from_goal

        return self.get_observation()


    def is_done(self):
        """ Check if simulation is over
        
        Return true if the nr of steps is bigger thant the maximum allowed nr of steps per episode
        """


        if self.distance_from_goal < 1 or self.steps >= self.n_steps:
            if self.distance_from_goal < 1: print("Reached goal")
            return True
        else:
            return False


    def get_observation(self):
        """ Return the current state of the environment

        [position particle (x,y), position goal (x,y), position closest particle (x,y)]
        """

        return [self.x, self.y, self.goal_x, self.goal_y, self.closest_particles_x, self.closest_particles_y]

    # Update function
    def step(self, action):
        """ Step function required by the rl environment
        
        1. Gets action in desired range
        2. Apply action that modifies state
        3. Get observation of current state
        4. Calculate reward
        """

        self.x_previous, self.y_previous = self.x, self.y

        self.steps += 1

        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.x = self.x + action[0]
        self.y = self.y + action[1]

        self.phi = np.arctan2(self.y - self.y_previous, self.x - self.x_previous)
        self.vel = np.linalg.norm([self.x - self.x_previous, self.y - self.y_previous])

        self.publish_position()

        observation = self.get_observation()

        self.calculate_distance_from_goal()

        self.reward = self.calculate_reward()

        done = self.is_done()

        information = {"Finished":done}

        # rospy.sleep(0.01)

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
        marker_position = Marker()
        marker_position.header.frame_id = "world"
        marker_position.id = 999
        marker_position.type = marker_position.SPHERE
        marker_position.action = marker_position.ADD
        marker_position.scale.x = 4
        marker_position.scale.y = 4
        marker_position.scale.z = 4
        marker_position.color.a = 1.0

        if(self.collision_path == False):
            marker_position.color.r = 0.0
            marker_position.color.g = 0.0
            marker_position.color.b = 1.0
        else:
            marker_position.color.r = 1.0
            marker_position.color.g = 0.0
            marker_position.color.b = 1.0

        marker_position.pose.orientation.w = 1.0
        marker_position.pose.position.x = float(self.x)
        marker_position.pose.position.y = float(self.y)
        marker_position.pose.position.z = 0
        

        # Velocity marker
        marker_velocity = Marker()
        marker_velocity.header.frame_id = "world"
        marker_velocity.id = 999
        marker_velocity.type = marker_velocity.ARROW
        marker_velocity.action = marker_velocity.ADD
        marker_velocity.scale.x = self.vel*10
        marker_velocity.scale.y = 1
        marker_velocity.scale.z = 1
        marker_velocity.color.a = 1.0
        marker_velocity.color.r = 1.0
        marker_velocity.color.g = 0.0
        marker_velocity.color.b = 0.0

        marker_velocity.pose.position = marker_position.pose.position

        quat = Rotation.from_euler("z", self.phi, degrees=False).as_quat()
        marker_velocity.pose.orientation.x = quat[0]
        marker_velocity.pose.orientation.y = quat[1]
        marker_velocity.pose.orientation.z = quat[2]
        marker_velocity.pose.orientation.w = quat[3]


        self.pub_marker_position.publish(marker_position)
        self.pub_marker_velocity.publish(marker_velocity)


    def generate_goal(self):
        """ Generate a goal for the particle to go to
        """
        self.goal_x = random.randint(-self.len_half, self.len_half)  # Position goal x
        self.goal_y = random.randint(-self.len_half, self.len_half)  # Position goal y


        # Velocity marker
        marker_goal = Marker()
        marker_goal.header.frame_id = "world"
        marker_goal.id = 999
        marker_goal.type = marker_goal.SPHERE
        marker_goal.action = marker_goal.ADD
        marker_goal.scale.x = 10
        marker_goal.scale.y = 10
        marker_goal.scale.z = 10
        marker_goal.color.a = 1.0
        marker_goal.color.r = 1.0
        marker_goal.color.g = 0.0
        marker_goal.color.b = 1.0
        marker_goal.pose.position.x = float(self.goal_x)
        marker_goal.pose.position.y = float(self.goal_y)
        marker_goal.pose.position.z = 0
        marker_goal.pose.orientation.x = 0
        marker_goal.pose.orientation.y = 0
        marker_goal.pose.orientation.z = 0
        marker_goal.pose.orientation.w = 1


        self.pub_marker_goal.publish(marker_goal)

        self.calculate_distance_from_goal()

    def calculate_distance_from_goal(self):
        self.distance_from_goal_previous = self.distance_from_goal
        self.distance_from_goal = np.linalg.norm([self.x - self.goal_x, self.y - self.goal_y])