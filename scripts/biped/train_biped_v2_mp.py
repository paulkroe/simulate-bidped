# scripts/biped/train_biped_v2_mp.py
"""
Multi-process training for biped walking.
Much faster than single-process version.
"""
from __future__ import annotations
import multiprocessing as mp

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from policies.actor_critic import ActorCritic
from algorithms.ppo import PPO, PPOConfig
from training.mp_on_policy import MultiProcessOnPolicyTrainer, MPTrainConfig

from tasks.biped.reward_v2 import reward
from tasks.biped.done_v2 import done_v2


def make_env() -> MujocoEnv:
    """Create biped environment (used by each worker)."""
    cfg = MujocoEnvConfig(
        xml_path="assets/biped/biped.xml",
        episode_length=1000,
        frame_skip=5,
        ctrl_scale=0.05,
        reset_noise_scale=0.02,
        reward_fn=reward,
        done_fn=done_v2,
        render=False,  # Workers don't render
    )
    return MujocoEnv(cfg)


def make_policy(env):
    """Create policy network."""
    return ActorCritic(
        env.spec,
        hidden_sizes=(128, 128),
    )


def make_ppo(actor_critic):
    """Create PPO algorithm."""
    ppo_cfg = PPOConfig(
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        lr=3e-4,
        train_iters=10,
        batch_size=256,  # Larger batch for multi-process
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    )
    return PPO(actor_critic, ppo_cfg, device="cpu")


def main():
    """Run multi-process training."""
    # Required for multiprocessing with PyTorch
    mp.set_start_method("spawn", force=True)

    train_cfg = MPTrainConfig(
        total_steps=200_000,
        horizon=1024,  # Each worker collects 1024 steps
        num_workers=8,  # 8 parallel environments
        log_interval=5,
        device="cpu",
        checkpoint_path="checkpoints/biped_v2_mp.pt",
    )

    print("="*60)
    print("Training Biped Walker - V2 (Multi-Process)")
    print("="*60)
    print(f"Total steps: {train_cfg.total_steps:,}")
    print(f"Workers: {train_cfg.num_workers}")
    print(f"Horizon per worker: {train_cfg.horizon}")
    print(f"Steps per iteration: {train_cfg.num_workers * train_cfg.horizon:,}")
    print(f"Checkpoint: {train_cfg.checkpoint_path}")
    print("="*60)
    
    trainer = MultiProcessOnPolicyTrainer(
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