import mujoco

def walker_done(model: mujoco.MjModel, data: mujoco.MjData, t: int) -> bool:
    # Termination if torso falls too low or flips
    torso_id = 1
    height = data.xpos[torso_id][2]

    if height < 0.7:   # tune for your model
        return True

    return False