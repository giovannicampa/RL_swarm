from gym.envs.registration import register
 
register(
    id='Particle-v0', 
    entry_point='particle_env:ParticleEnv',
)