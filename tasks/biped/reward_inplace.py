def reward(model, data):
    vx = data.qvel[0]
    r_speed = 2.0 * vx
    r_backward = -3.0 * max(0.0, -vx)

    pelvis_z = data.qpos[2]
    target_height = 0.65
    r_height = np.exp(-20.0 * (pelvis_z - target_height)**2)

    pitch = data.qpos[3]
    r_upright = np.exp(-10.0 * pitch**2)

    pitch_vel = data.qvel[3]
    r_stability = np.exp(-5.0 * pitch_vel**2)

    # gait symmetry (hip difference)
    phase = np.sin(2 * (data.qpos[7] - data.qpos[10]))
    r_symmetry = 0.2 * phase

    # forward pelvis drift
    pelvis_x = data.qpos[0]
    r_forward_pose = 0.1 * pelvis_x

    total = (
        r_speed +
        r_backward +
        0.3 * r_height +
        0.3 * r_upright +
        0.4 * r_stability +
        r_symmetry +
        r_forward_pose
    )

    return float(total)
