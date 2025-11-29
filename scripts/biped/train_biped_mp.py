# scripts/biped/train_biped_mp.py
from __future__ import annotations

import multiprocessing as mp

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from core.biped_obs_wrapper import BipedSensorWrapper
from policies.actor_critic import ActorCritic
from algorithms.ppo import PPO, PPOConfig
from training.mp_on_policy import MultiProcessOnPolicyTrainer, MPTrainConfig

from tasks.biped.reward import reward
from tasks.biped.done import done


def make_base_env() -> MujocoEnv:
    cfg = MujocoEnvConfig(
        xml_path="assets/biped/biped.xml",
        episode_length=5_000,
        frame_skip=5,
        ctrl_scale=0.1,
        reward_fn=reward,
        done_fn=done,
        reset_noise_scale=0.01,
        render=False,
        hip_site="base",
        left_foot_site="left_foot_ik",
        right_foot_site="right_foot_ik",
    )
    return MujocoEnv(cfg)


def make_env():
    return BipedSensorWrapper(make_base_env())


def make_policy(env):
    # obs dim now comes from BipedSensorWrapper (21 dims)
    return ActorCritic(env.spec, hidden_sizes=(128, 128))


def make_ppo(actor_critic):
    ppo_cfg = PPOConfig(
        gamma=0.995,
        lam=0.98,
        clip_ratio=0.2,
        lr=3e-4,
        train_iters=80,
        batch_size=512,
        value_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=0.5,
    )
    return PPO(actor_critic, ppo_cfg, device="cpu")


def main():
    mp.set_start_method("spawn", force=True)

    train_cfg = MPTrainConfig(
        total_steps=2_000_000,
        horizon=1024,
        num_workers=7,
        log_interval=10,
        device="cpu",
        checkpoint_path="checkpoints/biped_ppo_mp.pt",
    )

    trainer = MultiProcessOnPolicyTrainer(
        env_factory=make_env,
        policy_factory=make_policy,
        algo_factory=make_ppo,
        train_cfg=train_cfg,
    )
    trainer.run()


if __name__ == "__main__":
    main()