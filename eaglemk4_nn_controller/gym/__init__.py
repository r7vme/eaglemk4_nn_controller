from gym.envs.registration import register


register(
    id='eaglemk4-v0',
    entry_point='eaglemk4_nn_controller.gym.envs:EagleMK4Env',
    timestep_limit=1000000,
)
