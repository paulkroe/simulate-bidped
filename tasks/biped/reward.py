# tasks/biped/reward.py
import mujoco
import numpy as np

def reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    forward_reward_weight: float = 1.0,
    lateral_penalty_weight: float = 0.5,
    height_penalty_weight: float = 10.0,
    pitch_penalty_weight: float = 0.5,
    ctrl_cost_weight: float = 0.001,
    healthy_reward: float = 0.05,
    target_hip_height: float = 0.17,  # around initial hips z
) -> tuple[float, dict]:
    """
    Reward for the DIY biped:

    - forward_reward ~ x-velocity of the hips body
    - lateral_penalty penalizes y^2
    - height_penalty penalizes (z_hip - target_hip_height)^2
    - pitch_penalty penalizes torso pitch^2
    - ctrl_cost penalizes squared actions
    - healthy_reward if still in a reasonable configuration
    """

    # --- COM / body kinematics ---
    hips_id = model.body("hips").id
    # data.cvel: [nbody, 6], last 3 components are angular vel
    hips_cvel = data.cvel[hips_id]  # [6]
    hips_vel = hips_cvel[:3]       # (vx, vy, vz)

    vx = float(hips_vel[0])
    vy = float(hips_vel[1])

    forward_reward = forward_reward_weight * vx
    lateral_penalty = lateral_penalty_weight * (vy * vy)

    # --- height & pitch from body pose ---
    hips_pos = data.xpos[hips_id]  # [3]
    z_hip = float(hips_pos[2])

    height_error = z_hip - target_hip_height
    height_penalty = height_penalty_weight * (height_error * height_error)

    # Extract pitch from hips orientation.
    # data.xmat[hips_id] is a 3x3 rotation matrix in row-major form.
    # We'll roughly approximate pitch from this matrix.
    R = data.xmat[hips_id].reshape(3, 3)
    # Simple atan2-based pitch (rot around y axis):
    # assuming R = Rz * Ry * Rx; pitch ~ atan2(-R[2,0], sqrt(R[0,0]^2 + R[1,0]^2))
    pitch = float(np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)))
    pitch_penalty = pitch_penalty_weight * (pitch * pitch)

    # --- Control cost ---
    ctrl_cost = ctrl_cost_weight * float(np.sum(np.square(data.ctrl)))

    # --- Healthy bonus ---
    # Keep a simple check: no NaNs, reasonable height, moderate pitch
    is_healthy = (
        np.isfinite(data.qpos).all()
        and np.isfinite(data.qvel).all()
        and (0.05 < z_hip < 0.5)    # tweak as needed
        and (abs(pitch) < 1.0)
    )
    healthy_bonus = healthy_reward if is_healthy else 0.0

    total = (
        forward_reward
        - lateral_penalty
        - height_penalty
        - pitch_penalty
        - ctrl_cost
        + healthy_bonus
    )

    components = {
        "vx": vx,
        "vy": vy,
        "forward_reward": forward_reward,
        "lateral_penalty": lateral_penalty,
        "height_penalty": height_penalty,
        "pitch_penalty": pitch_penalty,
        "ctrl_cost": ctrl_cost,
        "healthy_bonus": healthy_bonus,
        "z_hip": z_hip,
        "pitch": pitch,
        "is_healthy": float(is_healthy),
    }
    return float(total), components
