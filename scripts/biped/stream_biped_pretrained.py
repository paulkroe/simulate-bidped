# scripts/biped/stream_biped_pretrained.py
"""
Visualize the PRETRAINED policy (before RL fine-tuning).

This is pure imitation learning - the neural network
mimicking the reference policy.
"""
from __future__ import annotations
import uvicorn

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from policies.actor_critic import ActorCritic
from streaming.mjpeg_server import create_app

from tasks.biped.reward_v2 import reward
from tasks.biped.done_v2 import done_v2


def make_env() -> MujocoEnv:
    """Create environment with rendering."""
    cfg = MujocoEnvConfig(
        xml_path="assets/biped/biped.xml",
        episode_length=5000,
        frame_skip=5,
        ctrl_scale=0.12,
        reset_noise_scale=0.05,
        reward_fn=reward,
        done_fn=done_v2,
        render=True,
        width=640,
        height=480,
        hip_site="base",
        left_foot_site="left_foot_ik",
        right_foot_site="right_foot_ik",
    )
    return MujocoEnv(cfg)


def make_policy(env):
    """Create policy network."""
    return ActorCritic(
        env.spec,
        hidden_sizes=(128, 128),
    )


def main():
    """Start streaming server."""
    # Use the PRETRAINED checkpoint (before RL)
    checkpoint_path = "checkpoints/biped_pretrained.pt"
    device = "cpu"

    print("="*60)
    print("Biped Walker Visualization - PRETRAINED ONLY")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Server: http://localhost:8000")
    print("="*60)
    print("This policy uses PURE IMITATION LEARNING")
    print("(no RL fine-tuning, just mimics reference policy)")
    print("="*60)

    app = create_app(
        env_factory=make_env,
        policy_factory=make_policy,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()