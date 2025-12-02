# scripts/biped/train_biped_v2.py
"""
Clean training script for biped walking with PPO.
Single-process version for debugging.
"""
from __future__ import annotations

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from policies.actor_critic import ActorCritic
from algorithms.ppo import PPO, PPOConfig
from training.on_policy import OnPolicyTrainer, TrainConfig

from tasks.biped.reward_v2 import reward
from tasks.biped.done_v2 import done_v2


def make_env():
    """Create biped environment with clean reward/done functions."""
    cfg = MujocoEnvConfig(
        xml_path="assets/biped/biped.xml",
        episode_length=1000,  # 1000 steps * 5 frame_skip * 0.002 dt = 10 seconds
        frame_skip=5,
        ctrl_scale=0.15,  # Scale down actions for stability
        reset_noise_scale=0.02,  # Small initial noise
        reward_fn=reward,
        done_fn=done_v2,
        render=False,
    )
    return MujocoEnv(cfg)


def make_policy(env):
    """Create policy network."""
    return ActorCritic(
        env.spec,
        hidden_sizes=(128, 128),  # Slightly larger than before
    )


def make_ppo(actor_critic):
    """Create PPO algorithm."""
    ppo_cfg = PPOConfig(
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        lr=3e-4,
        train_iters=10,  # 10 epochs per batch
        batch_size=128,
        value_coef=0.5,
        entropy_coef=0.02,  # Encourage exploration initially
        max_grad_norm=0.5,
    )
    return PPO(actor_critic, ppo_cfg, device="cpu")


def main():
    """Run training."""
    train_cfg = TrainConfig(
        total_steps=500_000,  
        horizon=2048,  # Collect 2048 steps before each update
        log_interval=5,  # Log every 5 iterations
        device="cpu",
        checkpoint_path="checkpoints/biped_v2.pt",
    )

    print("="*60)
    print("Training Biped Walker - V2")
    print("="*60)
    print(f"Total steps: {train_cfg.total_steps:,}")
    print(f"Horizon: {train_cfg.horizon}")
    print(f"Checkpoint: {train_cfg.checkpoint_path}")
    print("="*60)
    
    trainer = OnPolicyTrainer(
        env_factory=make_env,
        policy_factory=make_policy,
        algo_factory=make_ppo,
        train_cfg=train_cfg,
    )
    
    trainer.run()
    
    print("\nTraining complete! Test with:")
    print(f"  python scripts/biped/stream_biped_v2.py")


if __name__ == "__main__":
    main()