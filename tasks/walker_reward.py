import mujoco

import numpy as np

def walker_reward(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """
    Classic walker reward:
    + forward velocity
    - small control cost
    """

    # Forward velocity of torso (body 1, by convention)
    # You may need to adjust: often torso is body 1 or body named "torso"
    torso_id = 1
    vel = data.cvel[torso_id]   # spatial velocity (6D: angular + linear)
    forward_vel = vel[3]        # linear x velocity

    # Control effort penalty
    ctrl_cost = 0.001 * np.sum(np.square(data.ctrl))

    reward = forward_vel - ctrl_cost
    return float(reward)