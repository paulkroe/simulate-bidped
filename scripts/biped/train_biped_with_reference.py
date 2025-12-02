# scripts/biped/train_biped_with_reference.py
"""
Hybrid approach: Use reference policy to bootstrap learning.

Strategy:
1. Collect demonstrations from reference policy
2. Pretrain neural network with supervised learning
3. Fine-tune with RL

This is how most robotics papers actually do it.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from policies.actor_critic import ActorCritic
from policies.reference_policy import (
    ReferenceWalkerPolicy,
    GaitParams,
    WalkerJointMap,
    LegJointIndices,
)
from control.ik_2r import Planar2RLegConfig
from control.pd import PDConfig
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
        reset_noise_scale=0.05,
        reward_fn=reward,
        done_fn=done_v2,
        render=False,
        hip_site="base",
        left_foot_site="left_foot_ik",
        right_foot_site="right_foot_ik",
    )
    return MujocoEnv(cfg)


def make_reference_policy(env: MujocoEnv) -> ReferenceWalkerPolicy:
    """Create the hand-crafted reference policy."""
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


def collect_demonstrations(
    env: MujocoEnv,
    ref_policy: ReferenceWalkerPolicy,
    num_steps: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect state-action pairs from reference policy."""
    print(f"Collecting {num_steps} demonstration steps...")
    
    obs_list = []
    action_list = []
    
    obs = env.reset()
    ref_policy.reset()
    
    steps = 0
    while steps < num_steps:
        # Get action from reference policy
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_tensor, _ = ref_policy.act(obs_tensor, deterministic=True)
        action = action_tensor.squeeze(0).numpy()
        
        # Store
        obs_list.append(obs)
        action_list.append(action)
        
        # Step environment
        step_res = env.step(action)
        obs = step_res.obs if not step_res.done else env.reset()
        
        if step_res.done:
            ref_policy.reset()
        
        steps += 1
        if steps % 1000 == 0:
            print(f"  {steps}/{num_steps} steps collected")
    
    obs_array = np.array(obs_list, dtype=np.float32)
    action_array = np.array(action_list, dtype=np.float32)
    
    print(f"Collected {len(obs_array)} demonstrations")
    return obs_array, action_array


def pretrain_policy(
    policy: ActorCritic,
    demo_obs: np.ndarray,
    demo_actions: np.ndarray,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "cpu",
):
    """Pretrain policy using supervised learning on demonstrations."""
    print(f"\nPretraining policy for {epochs} epochs...")
    
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    dataset_size = len(demo_obs)
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(dataset_size)
        demo_obs_shuffled = demo_obs[indices]
        demo_actions_shuffled = demo_actions[indices]
        
        total_loss = 0.0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, dataset_size, batch_size):
            batch_obs = torch.as_tensor(
                demo_obs_shuffled[i:i+batch_size],
                dtype=torch.float32,
                device=device
            )
            batch_actions = torch.as_tensor(
                demo_actions_shuffled[i:i+batch_size],
                dtype=torch.float32,
                device=device
            )
            
            # Forward pass
            policy_output = policy.forward(batch_obs)
            predicted_actions = policy_output.action
            
            # Supervised learning loss (MSE)
            loss = nn.functional.mse_loss(predicted_actions, batch_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    print("Pretraining complete!")
    policy.eval()


def main():
    """Main training loop with reference policy initialization."""
    
    print("="*60)
    print("Training Biped Walker - WITH REFERENCE POLICY")
    print("="*60)
    print()
    print("This approach:")
    print("  1. Collects 10k steps from reference policy")
    print("  2. Pretrains neural network (supervised learning)")
    print("  3. Fine-tunes with PPO (reinforcement learning)")
    print()
    print("This combines the best of both:")
    print("  - Reference policy: known-good behavior")
    print("  - RL: optimization and adaptation")
    print("="*60)
    
    # Create environment
    env = make_env()
    
    # Create reference policy
    ref_policy = make_reference_policy(env)
    
    # Collect demonstrations
    demo_obs, demo_actions = collect_demonstrations(env, ref_policy, num_steps=10000)
    
    # Create neural network policy
    policy = ActorCritic(env.spec, hidden_sizes=(128, 128))
    
    # Pretrain with demonstrations
    pretrain_policy(policy, demo_obs, demo_actions, epochs=500)
    
    # Save pretrained policy
    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(policy.state_dict(), "checkpoints/biped_pretrained.pt")
    print("\nPretrained policy saved to checkpoints/biped_pretrained.pt")
    
    # Now do RL fine-tuning
    print("\n" + "="*60)
    print("Starting RL fine-tuning...")
    print("="*60)
    
    def make_policy_factory(env):
        """Factory that loads pretrained policy."""
        p = ActorCritic(env.spec, hidden_sizes=(128, 128))
        p.load_state_dict(torch.load("checkpoints/biped_pretrained.pt"))
        return p
    
    def make_ppo(actor_critic):
        ppo_cfg = PPOConfig(
            gamma=0.99,
            lam=0.95,
            clip_ratio=0.2,
            lr=1e-4,  # Lower LR for fine-tuning
            train_iters=10,
            batch_size=128,
            value_coef=0.5,
            entropy_coef=0.01,  # Lower entropy, policy already good
            max_grad_norm=0.5,
        )
        return PPO(actor_critic, ppo_cfg, device="cpu")
    
    train_cfg = TrainConfig(
        total_steps=1_000_000,
        horizon=2048,
        log_interval=5,
        device="cpu",
        checkpoint_path="checkpoints/biped_reference_rl.pt",
    )
    
    trainer = OnPolicyTrainer(
        env_factory=make_env,
        policy_factory=make_policy_factory,
        algo_factory=make_ppo,
        train_cfg=train_cfg,
    )
    
    trainer.run()
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print("\nThe policy started from reference behavior and was")
    print("refined with RL. Test with:")
    print("  python scripts/biped/stream_biped_reference_rl.py")


if __name__ == "__main__":
    main()