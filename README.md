# Path planning with collision avoidance

The goal of this project is to make an agent learn how to move in a way to reach a goal, and to avoid collision with particles in the surrounding environment.

## ROS setup

The simulation is launched with the __launch_swarm.launch__ file and is structured in two parts. The first one, named __swarm node__, simulates moving particles that can already avoid each other. This is done with a white box movement model. The second part is the __training node__ and is about a particle that learns to reach a set goal while avoiding surrounding particles.


## 1. Surrounding environment (moving particles)

The environment is populated with particles belonging to the **ParticleEnv** class. This class has a method that defines a position update for each particle in a way that it does not collide with other close particles and does not move to far away from the centre. The position of the closest particle to avoid is retrieved from inside the node. The position of the particles is then grouped in a __PoseArray__ message and published on the __/particles_position__ topic.


## 2. RL agent setup (learning particles)

The learning particle is a child class (**ParticleEnvRL**) of the particle class defined above. Additionaly it contains methods for the **Step**, **Reset** that define the movement and the beginning of a new learning episode.

### Reset
The reset method of the environment spawns the particle in a new position, defines new goal coordinates.

### Action
The agent's action space consists of two movements one in the x and one in the y direction. Each of these can vary between [-1, +1]. This action is added to the current position and defines the position at the beginning of the next step.

### Observation
The observation is defined as the carteisian coordinates (x,y) of the position of the ego particle, of the goal and the ones of the closest particle. While the first two positions are retrieved from attributes of the class, the position of the closest particle comes from subscribing to a ROS topic published by the __swarm node__.


## Results

The pictures below show the behaviour at the beginning and the end (1 million steps) of the training.
As can be see, the goal is reached

<table>
  <tr>
    <td>Beginning</td>
    <td>1M iterations</td>
  </tr>
  <tr>
    <td><img src="https://github.com/giovannicampa/RL_swarm/blob/master/src/pictures/Training_beginning.gif" width=270></td>
    <td><img src="https://github.com/giovannicampa/RL_swarm/blob/master/src/pictures/Training_reached_goal.gif" width=270></td>
  </tr>
 </table>


<p align="left">
  <img src="https://github.com/giovannicampa/RL_swarm/blob/master/src/pictures/training_results" width="700">
</p>

<p align="left">
  <em>Increasing rewards over the training process</em>
</p>
