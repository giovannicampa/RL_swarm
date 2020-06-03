from gym.envs.registration import register
 
register(id='Particle-env', 
    entry_point='swarm_robots.scripts.BubbleShooterEnv', 
)