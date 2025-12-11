# scripts/biped/record_biped_reference.py
from __future__ import annotations

import numpy as np

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from control.pd import PDConfig
from control.ik_2r import Planar2RLegConfig

from policies.reference_policy import (
    ReferenceWalkerPolicy,
    GaitParams,
)

from tasks.biped.reward import reward
from tasks.biped.done import done

from recording import CameraSettings, RecordingConfig, record_video

def make_env() -> MujocoEnv:
    cfg = MujocoEnvConfig(
        xml_path="assets/biped/biped.xml",
        episode_length=5_000,
        frame_skip=5,
        pd_cfg=PDConfig(kp=5.0, kd=1.0, torque_limit=5.0),
        reset_noise_scale=0.01,
        render=True,
        reward_fn=reward,
        done_fn=done,
        base_site="base",
        left_foot_site="left_foot_ik",
        right_foot_site="right_foot_ik",
        width=640,
        height=480,
    )
    return MujocoEnv(cfg)


def make_policy(env: MujocoEnv):
    # ---- Gait parameters ----
    gait_params = GaitParams(
        step_length=0.05,      # tune to be realistic for your tiny biped
        step_height=0.01,      # front foot height
        cycle_duration=1.25,   # 1 second per full step R->L
    )

    leg_geom_left = Planar2RLegConfig(
        L1=0.05,    # thigh length [m]
        L2=0.058,   # shank length [m]
        knee_sign=1.0,
        hip_offset=np.pi/2,
        knee_offset=0,
        ankle_offset=-np.pi/2,
    )
    leg_geom_right = Planar2RLegConfig(
        L1=0.05,
        L2=0.058,
        knee_sign=1.0,
        hip_offset=np.pi/2,
        knee_offset=0,
        ankle_offset=-np.pi/2,
    )

    pd_cfg = PDConfig(
        kp=5.0,
        kd=1.0,
        torque_limit=None,
    )

    return ReferenceWalkerPolicy(
        env=env,
        gait_params=gait_params,
        left_leg_geom=leg_geom_left,
        right_leg_geom=leg_geom_right,
        pd_config=pd_cfg,
        desired_foot_angle=0.0,  # parallel to ground
    )


def main():
    output_path = "recordings/biped_reference.mp4"
    num_steps = 1000  # Number of simulation steps to record
    device = "cpu"
    seed = None  # Set to an integer for reproducible recordings

    # Camera settings
    camera = CameraSettings(
        distance=1.0,
        azimuth=90.0,
        elevation=-90.0,
        lookat=(0.6, 0.0, 0.5),
    )
    # -------------------------------------------------------------------------

    config = RecordingConfig(
        output_path=output_path,
        num_steps=num_steps,
        camera=camera,
    )

    print(f"Recording {num_steps} steps to {output_path}...")

    stats = record_video(
        env_factory=make_env,
        policy_factory=make_policy,
        config=config,
        checkpoint_path=None,  # No checkpoint needed for reference policy
        device=device,
        seed=seed,
    )

    print("\nRecording complete!")
    print(f"  Output: {stats['output_path']}")
    print(f"  Frames: {stats['num_frames']}")
    print(f"  FPS: {stats['fps']:.1f}")
    print(f"  Duration: {stats['duration_seconds']:.2f} seconds")
    print(f"  Episodes: {stats['num_episodes']}")
    print(f"  Mean reward: {stats['mean_episode_reward']:.2f}")


if __name__ == "__main__":
    main()