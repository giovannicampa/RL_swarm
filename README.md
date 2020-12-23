# RL_swarm
The goal of this project is to make an agent learn how to move in a way to reach a goal, and to avoid collision with particles in the surrounding environment.

## ROS setup

The simulation is stru

## Environment

The environment is populated with particles belonging to the **ParticleEnv** class. This class has a method that defines a position update for each particle in a way that it does not collide with other close particles and does not move to far away from the centre. The particles in the environment are spawned as 

## Setup

The learning particle is a child class (**ParticleEnvRL**) of the particle class defined above. It contains methods for the **Step**, **Reset** that define the movement and the beginning of a new learning episode.

### Reset
The reset method of the environment spawns the particle in a new position, defines new goal coordinates.

### Action
The agent's action space consists of two movements one in the x and one in the y direction. Each of these can vary between [-1, +1]. This action is added to the current position and defines the position at the beginning of the next step.

### Observation
The observation is defined as the carteisian coordinates (x,y) of the position of the ego particle, of the goal and the ones of the closest particle.


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
