# scripts/stream_biped_reference.py
from __future__ import annotations

import uvicorn
import numpy as np

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from streaming.mjpeg_server import create_app

from policies.reference_policy import (
    ReferenceWalkerPolicy,
    GaitParams,
    WalkerJointMap,
    LegJointIndices,
)
from control.ik_2r import Planar2RLegConfig
from control.pd import PDConfig

from tasks.biped.reward import reward as reward
from tasks.biped.done import done as done


def make_env() -> MujocoEnv:
    cfg = MujocoEnvConfig(
        xml_path="assets/biped/biped.xml",
        episode_length=5_000,
        frame_skip=5,
        ctrl_scale=0.1,
        reset_noise_scale=0.01,
        render=True,
        reward_fn=reward,
        done_fn=done,
        width=640,
        height=480,
        hip_site="base",
        left_foot_site="left_foot_ik",
        right_foot_site="right_foot_ik",
    )
    return MujocoEnv(cfg)


def make_policy(env: MujocoEnv):
    # ---- Gait parameters (Option A: small, safe gait) ----
    gait_params = GaitParams(
        step_length=0.02,      # small horizontal excursion
        step_height=0.01,      # low but non-zero clearance
        cycle_duration=1.25,   # ~1.25s per full L->R cycle
    )

    # 2R leg geometry (same for both legs)
    leg_geom_left = Planar2RLegConfig(
        L1=0.05,       # thigh length [m]
        L2=0.058,      # shank length [m]
        knee_sign=1.0,
        hip_offset=np.pi / 2,     # rotate plane into model frame
        knee_offset=0.0,
        ankle_offset=-np.pi / 2,
    )
    leg_geom_right = Planar2RLegConfig(
        L1=0.05,
        L2=0.058,
        knee_sign=1.0,
        hip_offset=np.pi / 2,
        knee_offset=0.0,
        ankle_offset=-np.pi / 2,
    )

    # PD gains (global for all joints)
    pd_cfg = PDConfig(
        kp=50.0,
        kd=1.0,
        torque_limit=None,
    )

    # Joint indices from inspect_joints.py:
    #
    #  free joint: qpos 0..6
    #  7  left_hip_roll
    #  8  left_hip_pitch
    #  9  left_leg_joint (knee)
    # 10  left_foot_joint (ankle)
    # 11  right_hip_roll
    # 12  right_hip_pitch
    # 13  right_leg_joint
    # 14  right_foot_joint
    joint_map = WalkerJointMap(
        left=LegJointIndices(
            hip_roll=7,
            hip_pitch=8,
            knee=9,
            ankle=10,
        ),
        right=LegJointIndices(
            hip_roll=11,
            hip_pitch=12,
            knee=13,
            ankle=14,
        ),
    )

    return ReferenceWalkerPolicy(
        env=env,
        gait_params=gait_params,
        joint_map=joint_map,
        left_leg_geom=leg_geom_left,
        right_leg_geom=leg_geom_right,
        pd_config=pd_cfg,
        desired_foot_angle=0.0,  # parallel to ground
    )


def main():
    device = "cpu"

    app = create_app(
        env_factory=make_env,
        policy_factory=make_policy,
        checkpoint_path=None,
        device=device,
    )

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
