# scripts/biped/train_from_pretrained_balance.py
"""
RL training starting from pretrained policy.

The pretrained policy can take a few steps but falls.
This training will teach it balance and recovery.

Strategy:
- Very conservative RL (don't forget walking skill)
- Heavily reward episode length (staying upright longer)
- Lower learning rate
- Less exploration (already knows how to step)
"""
from __future__ import annotations
import torch

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from policies.actor_critic import ActorCritic
from algorithms.ppo import PPO, PPOConfig
from training.on_policy import OnPolicyTrainer, TrainConfig

from tasks.biped.reward_v2 import reward
from tasks.biped.done_v2 import done_v2


def make_env():
    """Create biped environment."""
    cfg = MujocoEnvConfig(
        xml_path="assets/biped/biped.xml",
        episode_length=1000,
        frame_skip=5,
        ctrl_scale=0.12,
        reset_noise_scale=0.03,  # Small noise, we already know the skill
        reward_fn=reward,
        done_fn=done_v2,
        render=False,
        hip_site="base",
        left_foot_site="left_foot_ik",
        right_foot_site="right_foot_ik",
    )
    return MujocoEnv(cfg)


def make_policy(env):
    """Create policy and load pretrained weights."""
    policy = ActorCritic(env.spec, hidden_sizes=(128, 128))
    
    # Load the pretrained policy
    checkpoint = torch.load("checkpoints/biped_pretrained.pt", map_location="cpu")
    policy.load_state_dict(checkpoint)
    
    print("Loaded pretrained policy from checkpoints/biped_pretrained.pt")
    return policy


def make_ppo(actor_critic):
    """Create PPO with VERY conservative settings."""
    ppo_cfg = PPOConfig(
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.1,       # VERY SMALL (was 0.2) - don't change policy much
        lr=3e-5,              # VERY LOW (was 3e-4) - small updates
        train_iters=5,        # FEWER (was 10) - less aggressive
        batch_size=128,
        value_coef=0.5,
        entropy_coef=0.002,   # VERY LOW (was 0.01) - minimal exploration
        max_grad_norm=0.3,    # STRICT (was 0.5) - prevent big changes
    )
    return PPO(actor_critic, ppo_cfg, device="cpu")


def main():
    """Run conservative RL from pretrained policy."""
    train_cfg = TrainConfig(
        total_steps=500_000,   # Moderate training
        horizon=2048,
        log_interval=5,
        device="cpu",
        checkpoint_path="checkpoints/biped_balance.pt",
    )

    print("="*60)
    print("Training Biped Walker - BALANCE REFINEMENT")
    print("="*60)
    print()
    print("Starting from pretrained policy that can take a few steps.")
    print()
    print("Goal: Learn to maintain balance for longer episodes")
    print()
    print("Strategy:")
    print("  - VERY conservative RL (clip_ratio=0.1, lr=3e-5)")
    print("  - Small changes to avoid forgetting walking skill")
    print("  - Focus on balance and recovery")
    print("="*60)
    print(f"Total steps: {train_cfg.total_steps:,}")
    print(f"Checkpoint: {train_cfg.checkpoint_path}")
    print("="*60)
    print()
    print("What to watch:")
    print("  - Episode length should increase (robot stays up longer)")
    print("  - Returns should gradually improve from starting point")
    print("  - If returns drop >30%, stop training (forgetting)")
    print("="*60)
    
    trainer = OnPolicyTrainer(
        env_factory=make_env,
        policy_factory=make_policy,
        algo_factory=make_ppo,
        train_cfg=train_cfg,
    )
    
    trainer.run()
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print("\nTest with:")
    print("  python scripts/biped/stream_biped_balance.py")


if __name__ == "__main__":
    main()