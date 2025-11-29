# core/biped_obs_wrapper.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np
import mujoco

from core.base_env import Env, StepResult
from core.specs import EnvSpec, SpaceSpec


@dataclass
class BipedSensorConfig:
    contact_height_thresh: float = 0.005


class BipedSensorWrapper(Env):
    """
    Wrap MujocoEnv for the DIY biped to expose a 'proprio-like' observation:

        obs = [
          torso_orientation_quat (4),
          torso_angular_velocity (3),
          joint_angles (6),
          joint_velocities (6),
          left_foot_contact (1),
          right_foot_contact (1),
        ]  => 21-dim vector
    """

    def __init__(self, env: Env, cfg: BipedSensorConfig | None = None):
        self.env = env
        self.cfg = cfg or BipedSensorConfig()

        model = self.env.model
        data = self.env.data

        self._hips_id = model.body("hips").id
        self._left_foot_id = model.body("left_foot").id
        self._right_foot_id = model.body("right_foot").id

        # Joint names for 6 actuated joints
        self._joint_names = [
            "left_hip_joint",
            "left_leg_joint",
            "left_foot_joint",
            "right_hip_joint",
            "right_leg_joint",
            "right_foot_joint",
        ]

        nq = model.nq
        nv = model.nv
        offset = nq - nv

        q_indices = []
        qd_indices = []
        for jname in self._joint_names:
            j_id = model.joint(jname).id
            q_idx = int(model.jnt_qposadr[j_id])
            v_idx = q_idx - offset
            q_indices.append(q_idx)
            qd_indices.append(v_idx)

        self._q_indices = np.array(q_indices, dtype=int)
        self._qd_indices = np.array(qd_indices, dtype=int)

        obs_dim = 4 + 3 + 6 + 6 + 2
        self.spec = EnvSpec(
            obs=SpaceSpec(shape=(obs_dim,), dtype=np.float32),
            act=self.env.spec.act,
        )

    # ---------- helpers ----------
    def _build_obs(self) -> np.ndarray:
        model = self.env.model
        data = self.env.data

        # torso orientation and angular velocity
        quat = data.xquat[self._hips_id]     # [4]
        cvel = data.cvel[self._hips_id]      # [6]
        ang_vel = cvel[3:]                   # [3]

        q = np.asarray(data.qpos, dtype=np.float32)
        qd = np.asarray(data.qvel, dtype=np.float32)

        q_joints = q[self._q_indices]        # [6]
        qd_joints = qd[self._qd_indices]     # [6]

        # binary "contact" via height threshold
        left_z = data.xpos[self._left_foot_id, 2]
        right_z = data.xpos[self._right_foot_id, 2]
        l_contact = float(left_z < self.cfg.contact_height_thresh)
        r_contact = float(right_z < self.cfg.contact_height_thresh)

        obs = np.concatenate(
            [
                quat.astype(np.float32),
                ang_vel.astype(np.float32),
                q_joints,
                qd_joints,
                np.array([l_contact, r_contact], dtype=np.float32),
            ],
            axis=0,
        )
        return obs

    def __getattr__(self, name):
        """
        Proxy all unknown attributes to the underlying MujocoEnv.
        This makes the wrapper fully transparent to the streaming server.
        """
        if name == "env":
            return super().__getattribute__("env")
        return getattr(self.env, name)

    # ---------- Env interface ----------
    def reset(self, seed: int | None = None) -> np.ndarray:
        _ = self.env.reset(seed=seed)
        return self._build_obs()

    def step(self, action: np.ndarray) -> StepResult:
        step_res = self.env.step(action)
        obs = self._build_obs()
        return StepResult(
            obs=obs,
            reward=step_res.reward,
            done=step_res.done,
            info=step_res.info,
            frame=step_res.frame,
        )

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> "BipedSensorWrapper":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
