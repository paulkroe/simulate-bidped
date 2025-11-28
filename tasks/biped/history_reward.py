# tasks/walker_paper_reward.py
from __future__ import annotations
from typing import Dict, Any, Tuple

import numpy as np
import mujoco

from tasks.walker_reference_motion import WalkerReferenceMotion


def exp_tracking(u: np.ndarray, v: np.ndarray, alpha: float) -> float:
    diff = u - v
    return float(np.exp(-alpha * np.dot(diff, diff)))


class HistoryReward:
    """
    Implements a simplified version of Table III:
    - motion tracking: motor positions, pelvis height, foot height
    - task completion: pelvis velocity
    - smoothing: torque, motor velocity, change of action
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        ref_motion: WalkerReferenceMotion,
        w_motor_pos: float = 15.0,
        w_pelvis_height: float = 5.0,
        w_foot_height: float = 10.0,
        w_pelvis_vel: float = 15.0,
        w_torque: float = 3.0,
        w_motor_vel: float = 3.0,
        w_action_change: float = 3.0,
        alpha_motor_pos: float = 10.0,
        alpha_height: float = 50.0,
        alpha_vel: float = 5.0,
        alpha_torque: float = 2.0,
        alpha_motor_vel: float = 2.0,
        alpha_action_change: float = 2.0,
    ):
        self.model = model
        self.ref_motion = ref_motion

        # weights
        self.w_motor_pos = w_motor_pos
        self.w_pelvis_height = w_pelvis_height
        self.w_foot_height = w_foot_height
        self.w_pelvis_vel = w_pelvis_vel
        self.w_torque = w_torque
        self.w_motor_vel = w_motor_vel
        self.w_action_change = w_action_change

        # alphas
        self.alpha_motor_pos = alpha_motor_pos
        self.alpha_height = alpha_height
        self.alpha_vel = alpha_vel
        self.alpha_torque = alpha_torque
        self.alpha_motor_vel = alpha_motor_vel
        self.alpha_action_change = alpha_action_change

        # For change-of-action smoothing
        self._last_action = None

    def __call__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: int,
        dt: float,
        action: np.ndarray,
        command: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward and components.

        Args:
            t: environment step index
            dt: env timestep (model.opt.timestep * frame_skip)
        """
        time_sec = t * dt
        ref = self.ref_motion.get(time_sec, command=command)

        # Current quantities
        q = np.array(data.qpos, dtype=np.float32)
        qdot = np.array(data.qvel, dtype=np.float32)
        torque = np.array(data.qfrc_actuator, dtype=np.float32)

        # Pelvis state (assuming body 0 is pelvis; adjust indices as needed)
        pelvis_body = 0
        pelvis_pos = data.xpos[pelvis_body].copy()
        pelvis_vel = data.cvel[pelvis_body].copy()  # 6D spatial vel
        pelvis_vel_xy = pelvis_vel[3:5]            # world x,y linear vel
        pelvis_height = pelvis_pos[2]

        # Foot heights: adjust body names/ids to your model
        left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
        lf_height = data.xpos[left_foot_id][2]
        rf_height = data.xpos[right_foot_id][2]

        # Motion tracking components
        r_motor_pos = exp_tracking(q, ref["q_m_ref"], self.alpha_motor_pos)
        r_pelvis_height = exp_tracking(
            np.array([pelvis_height], dtype=np.float32),
            np.array([ref["pelvis_height_ref"]], dtype=np.float32),
            self.alpha_height,
        )
        r_foot_height = exp_tracking(
            np.array([lf_height, rf_height], dtype=np.float32),
            np.array(
                [ref["left_foot_height_ref"], ref["right_foot_height_ref"]],
                dtype=np.float32,
            ),
            self.alpha_height,
        )

        # Task completion: track pelvis velocity
        r_pelvis_vel = exp_tracking(pelvis_vel_xy, ref["pelvis_vel_ref"], self.alpha_vel)

        # Smoothing terms
        r_torque = exp_tracking(torque, np.zeros_like(torque), self.alpha_torque)
        r_motor_vel = exp_tracking(qdot, np.zeros_like(qdot), self.alpha_motor_vel)

        if self._last_action is None:
            r_action_change = 1.0  # neutral at first step
        else:
            r_action_change = exp_tracking(
                action, self._last_action, self.alpha_action_change
            )
        self._last_action = action.copy()

        # Weighted sum (optionally normalize as in the paper with ||w||_1)
        components = {
            "motor_pos": self.w_motor_pos * r_motor_pos,
            "pelvis_height": self.w_pelvis_height * r_pelvis_height,
            "foot_height": self.w_foot_height * r_foot_height,
            "pelvis_vel": self.w_pelvis_vel * r_pelvis_vel,
            "torque": self.w_torque * r_torque,
            "motor_vel": self.w_motor_vel * r_motor_vel,
            "action_change": self.w_action_change * r_action_change,
        }

        total = sum(components.values())
        return total, components