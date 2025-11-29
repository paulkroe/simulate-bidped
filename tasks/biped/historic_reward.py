# tasks/biped/history_reward.py
from __future__ import annotations
from typing import Callable, Mapping, Tuple, Dict
import numpy as np
import mujoco

from core.mujoco_env import MujocoEnv, RewardReturn

# Callable that returns reference joint positions q_ref(t) (full qpos)
RefQFn = Callable[[float], np.ndarray]


def exp_reward(u: np.ndarray, v: np.ndarray, alpha: float) -> float:
    """
    r(u, v) = exp(-alpha * ||u - v||^2)
    Used for all components, as in Eq. (2) in the paper.
    """
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    diff = u - v
    return float(np.exp(-alpha * float(np.dot(diff, diff))))


def make_historic_reward(
    env: MujocoEnv,
    ref_q_fn: RefQFn,
    torso_body: str = "torso",
    v_des: float = 0.6,   # desired forward speed
) -> Callable[[mujoco.MjModel, mujoco.MjData, int, float, np.ndarray], RewardReturn]:
    """
    Build a Cassie-style reward for your Walker, simplified but structurally similar:

      - Motion tracking:
          * joint positions vs reference: r(q_m, q_m^r(t))
          * pelvis height vs nominal:     r(q_z, q_z^r)
      - Task completion:
          * forward velocity vs target:   r(v_x, v_des)
      - Smoothing:
          * small torques:                r(tau, 0)
          * small action changes:         r(a_t, a_{t-1})

    Combined as a weighted sum and normalized by L1 norm of weights, as in rt = (w / ||w||_1)^T r. 
    """
    model = env.model
    torso_id = model.body(torso_body).id

    # ------------------------------------------------------------------
    # Choose which joints we call "motors" (for q_m). For a floating base:
    #   nq = 7 + n_hinge, nv = 6 + n_hinge → offset = nq - nv = 1
    # Here we treat all non-base joints as motors.
    # If you want a stricter subset, you can adjust motor_q_idx.
    # ------------------------------------------------------------------
    nq = model.nq
    nv = model.nv
    offset = nq - nv
    motor_q_idx = np.arange(offset, nq, dtype=int)

    # ------------------------------------------------------------------
    # Weights (roughly inspired by Table III for walking) 
    # ------------------------------------------------------------------
    w_motion_q      = 15.0
    w_pelvis_height = 5.0
    w_pelvis_vel    = 15.0
    w_torque        = 3.0
    w_action_diff   = 3.0

    w_sum = (
        abs(w_motion_q)
        + abs(w_pelvis_height)
        + abs(w_pelvis_vel)
        + abs(w_torque)
        + abs(w_action_diff)
    )

    last_action = np.zeros(env.spec.act.shape[0], dtype=np.float32)

    def reward_fn(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: int,
        dt: float,
        action: np.ndarray,
    ) -> Tuple[float, Mapping[str, float]]:
        nonlocal last_action

        time_sec = t * dt

        q = data.qpos.copy()
        qd = data.qvel.copy()
        a = np.asarray(action, dtype=np.float32)

        # --------------------------------------------------------------
        # 1) Motion tracking: joint positions vs reference
        # --------------------------------------------------------------
        q_ref_full = np.asarray(ref_q_fn(time_sec), dtype=np.float32)
        q_m     = q[motor_q_idx]
        q_m_ref = q_ref_full[motor_q_idx]

        r_motion_q = exp_reward(q_m, q_m_ref, alpha=10.0)

        # --------------------------------------------------------------
        # 2) Motion tracking: pelvis height (global z)
        # --------------------------------------------------------------
        pelvis_pos = data.xpos[torso_id].copy()
        qz = float(pelvis_pos[2])

        # Use env.hip_height as nominal reference if available,
        # otherwise just treat current as reference (effectively no term).
        if env.hip_height is not None:
            qz_ref = float(env.hip_height)
        else:
            qz_ref = qz

        r_pelvis_height = exp_reward(
            np.array([qz], dtype=np.float32),
            np.array([qz_ref], dtype=np.float32),
            alpha=50.0,
        )

        # --------------------------------------------------------------
        # 3) Task completion: forward velocity vs target
        #    (pelvis vx in world frame)
        # --------------------------------------------------------------
        # MuJoCo cvel: [wx, wy, wz, vx, vy, vz] in world coordinates.
        vel_spatial = data.cvel[torso_id]
        vx = float(vel_spatial[3])

        r_pelvis_vel = exp_reward(
            np.array([vx], dtype=np.float32),
            np.array([v_des], dtype=np.float32),
            alpha=2.0,
        )

        # --------------------------------------------------------------
        # 4) Smoothing: torque magnitude r(tau, 0)
        # --------------------------------------------------------------
        r_torque = exp_reward(a, np.zeros_like(a), alpha=0.1)

        # --------------------------------------------------------------
        # 5) Smoothing: change of action r(a_t, a_{t-1})
        # --------------------------------------------------------------
        if t == 0:
            last_action = a.copy()
        delta_a = a - last_action
        r_action_diff = exp_reward(delta_a, np.zeros_like(delta_a), alpha=1.0)
        last_action = a.copy()

        # --------------------------------------------------------------
        # Weighted sum + normalization
        # --------------------------------------------------------------
        components_weighted = {
            "motion_q":      w_motion_q      * r_motion_q,
            "pelvis_height": w_pelvis_height * r_pelvis_height,
            "pelvis_vel":    w_pelvis_vel    * r_pelvis_vel,
            "torque":        w_torque        * r_torque,
            "action_diff":   w_action_diff   * r_action_diff,
        }

        total = sum(components_weighted.values()) / w_sum

        # For logging / streaming, expose *un-normalized* weighted terms
        return float(total), {k: float(v) for k, v in components_weighted.items()}

    return reward_fn