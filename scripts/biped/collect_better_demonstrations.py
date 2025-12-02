# scripts/biped/collect_better_demonstrations.py
"""
Collect higher-quality demonstrations with MORE DATA.

The issue: NN learned leg motions but not weight shifting.
Solution: Collect WAY more demonstrations so NN sees the full 
         diversity of weight-shift situations.

Strategy:
- 100k demonstrations (was 10k)
- From complete walking cycles
- Include ALL phases of gait
"""
from __future__ import annotations
import numpy as np
import torch

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from policies.reference_policy import (
    ReferenceWalkerPolicy,
    GaitParams,
    WalkerJointMap,
    LegJointIndices,
)
from control.ik_2r import Planar2RLegConfig
from control.pd import PDConfig

from tasks.biped.reward_v4 import reward_v4
from tasks.biped.done_v2_strict import done_v2_strict


def make_env():
    """Create biped environment."""
    cfg = MujocoEnvConfig(
        xml_path="assets/biped/biped.xml",
        episode_length=1000,
        frame_skip=5,
        ctrl_scale=0.12,
        reset_noise_scale=0.05,
        reward_fn=reward_v4,
        done_fn=done_v2_strict,
        render=False,
        hip_site="base",
        left_foot_site="left_foot_ik",
        right_foot_site="right_foot_ik",
    )
    return MujocoEnv(cfg)


def make_reference_policy(env: MujocoEnv) -> ReferenceWalkerPolicy:
    """Create the reference policy that works."""
    gait_params = GaitParams(
        step_length=0.02,
        step_height=0.01,
        cycle_duration=1.25,
    )

    joint_map = WalkerJointMap(
        left=LegJointIndices(hip=7, knee=8, ankle=9),
        right=LegJointIndices(hip=10, knee=11, ankle=12),
    )

    left_leg_geom = Planar2RLegConfig(
        L1=0.05, L2=0.058,
        knee_sign=1.0,
        hip_offset=np.pi/2,
        knee_offset=0.0,
        ankle_offset=-np.pi/2,
    )
    right_leg_geom = Planar2RLegConfig(
        L1=0.05, L2=0.058,
        knee_sign=1.0,
        hip_offset=np.pi/2,
        knee_offset=0.0,
        ankle_offset=-np.pi/2,
    )

    pd_cfg = PDConfig(kp=50.0, kd=1.0, torque_limit=None)

    return ReferenceWalkerPolicy(
        env=env,
        gait_params=gait_params,
        joint_map=joint_map,
        left_leg_geom=left_leg_geom,
        right_leg_geom=right_leg_geom,
        pd_config=pd_cfg,
        desired_foot_angle=0.0,
    )


def main():
    """Collect high-quality demonstrations."""
    
    print("="*60)
    print("Collecting HIGH-QUALITY Demonstrations")
    print("="*60)
    print()
    print("Strategy: Collect 100k steps (10x more than before)")
    print("This ensures NN sees all phases of weight shift")
    print("="*60)
    
    env = make_env()
    ref_policy = make_reference_policy(env)
    
    # Collect 100k demonstrations
    num_demos = 100_000
    
    print(f"\nCollecting {num_demos:,} demonstration steps...")
    print("This will take ~10-15 minutes...")
    
    obs_list = []
    action_list = []
    
    obs = env.reset()
    ref_policy.reset()
    
    steps = 0
    episodes = 0
    total_reward = 0.0
    
    while steps < num_demos:
        # Get action from reference
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_tensor, _ = ref_policy.act(obs_tensor, deterministic=True)
        action = action_tensor.squeeze(0).numpy()
        
        # Store
        obs_list.append(obs)
        action_list.append(action)
        
        # Step
        step_res = env.step(action)
        total_reward += step_res.reward
        obs = step_res.obs
        
        if step_res.done:
            episodes += 1
            obs = env.reset()
            ref_policy.reset()
        
        steps += 1
        if steps % 10000 == 0:
            avg_reward = total_reward / steps
            print(f"  {steps:,}/{num_demos:,} steps | "
                  f"{episodes} episodes | "
                  f"avg reward: {avg_reward:.2f}")
    
    obs_array = np.array(obs_list, dtype=np.float32)
    action_array = np.array(action_list, dtype=np.float32)
    
    print(f"\n✓ Collected {len(obs_array):,} high-quality demonstrations")
    print(f"  Episodes completed: {episodes}")
    print(f"  Average reward: {total_reward / steps:.2f}")
    
    # Save to disk
    import os
    os.makedirs("data", exist_ok=True)
    
    np.save("data/demo_obs_100k.npy", obs_array)
    np.save("data/demo_actions_100k.npy", action_array)
    
    print(f"\n✓ Saved to:")
    print(f"  data/demo_obs_100k.npy")
    print(f"  data/demo_actions_100k.npy")
    print()
    print("="*60)
    print("Next step: Train with more demonstrations and epochs")
    print("  python scripts/biped/train_with_100k_demos.py")
    print("="*60)


if __name__ == "__main__":
    main()