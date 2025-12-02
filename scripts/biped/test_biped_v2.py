# scripts/biped/test_biped_v2.py
"""
Diagnostic script to verify biped environment setup.
Run this BEFORE training to catch issues early.
"""
from __future__ import annotations
import numpy as np

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from policies.actor_critic import ActorCritic
from tasks.biped.reward_v2 import reward_v2
from tasks.biped.done_v2 import done_v2


def make_env():
    cfg = MujocoEnvConfig(
        xml_path="assets/biped/biped.xml",
        episode_length=1000,
        frame_skip=5,
        ctrl_scale=0.05,
        reset_noise_scale=0.02,
        reward_fn=reward_v2,
        done_fn=done_v2,
        render=False,
    )
    return MujocoEnv(cfg)


def test_basic_setup():
    """Test 1: Environment can be created and reset."""
    print("\n" + "="*60)
    print("TEST 1: Basic Setup")
    print("="*60)
    
    try:
        env = make_env()
        print(f"✓ Environment created")
        print(f"  - Observation dim: {env.spec.obs.shape[0]}")
        print(f"  - Action dim: {env.spec.act.shape[0]}")
        print(f"  - dt: {env.dt:.4f} seconds")
        
        obs = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  - Initial obs shape: {obs.shape}")
        print(f"  - Obs range: [{obs.min():.3f}, {obs.max():.3f}]")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_step_function():
    """Test 2: Can step environment and get valid rewards."""
    print("\n" + "="*60)
    print("TEST 2: Step Function")
    print("="*60)
    
    try:
        env = make_env()
        env.reset()
        
        # Try zero action
        action = np.zeros(env.spec.act.shape[0])
        step_res = env.step(action)
        
        print(f"✓ Step executed successfully")
        print(f"  - Reward: {step_res.reward:.4f}")
        print(f"  - Done: {step_res.done}")
        print(f"  - Info keys: {list(step_res.info.keys())}")
        
        # Check reward components
        if "reward_components" in step_res.info:
            components = step_res.info["reward_components"]
            print(f"\n  Reward components:")
            for k, v in components.items():
                print(f"    {k:15s}: {v:7.4f}")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_episode_rollout():
    """Test 3: Run a short episode with random actions."""
    print("\n" + "="*60)
    print("TEST 3: Episode Rollout (100 steps)")
    print("="*60)
    
    try:
        env = make_env()
        env.reset()
        
        total_reward = 0.0
        done = False
        step_count = 0
        max_steps = 100
        
        while step_count < max_steps and not done:
            # Random action
            action = np.random.randn(env.spec.act.shape[0]) * 0.1
            step_res = env.step(action)
            
            total_reward += step_res.reward
            done = step_res.done
            step_count += 1
        
        print(f"✓ Rollout completed")
        print(f"  - Steps taken: {step_count}")
        print(f"  - Episode terminated: {done}")
        print(f"  - Total reward: {total_reward:.4f}")
        print(f"  - Average reward: {total_reward/step_count:.4f}")
        
        if done and step_count < max_steps:
            print(f"  ⚠ Episode ended early (step {step_count}/{max_steps})")
            print(f"    This might indicate robot fell or violated constraints")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_policy_creation():
    """Test 4: Policy can be created and used."""
    print("\n" + "="*60)
    print("TEST 4: Policy Creation")
    print("="*60)
    
    try:
        env = make_env()
        policy = ActorCritic(env.spec, hidden_sizes=(128, 128))
        
        print(f"✓ Policy created")
        
        # Count parameters
        total_params = sum(p.numel() for p in policy.parameters())
        print(f"  - Total parameters: {total_params:,}")
        
        # Test forward pass
        obs = env.reset()
        import torch
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = policy.forward(obs_tensor)
        
        print(f"✓ Forward pass successful")
        print(f"  - Action shape: {output.action.shape}")
        print(f"  - Value shape: {output.value.shape}")
        print(f"  - Log prob shape: {output.log_prob.shape}")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_components():
    """Test 5: Analyze reward function behavior."""
    print("\n" + "="*60)
    print("TEST 5: Reward Function Analysis")
    print("="*60)
    
    try:
        env = make_env()
        env.reset()
        
        # Collect rewards over 50 steps
        rewards = []
        components_sum = {}
        
        for _ in range(50):
            action = np.random.randn(env.spec.act.shape[0]) * 0.05
            step_res = env.step(action)
            rewards.append(step_res.reward)
            
            # Accumulate components
            if "reward_components" in step_res.info:
                for k, v in step_res.info["reward_components"].items():
                    if k not in components_sum:
                        components_sum[k] = []
                    components_sum[k].append(v)
        
        print(f"✓ Collected 50 steps of reward data")
        print(f"\n  Overall statistics:")
        print(f"    Mean reward: {np.mean(rewards):.4f}")
        print(f"    Std reward:  {np.std(rewards):.4f}")
        print(f"    Min reward:  {np.min(rewards):.4f}")
        print(f"    Max reward:  {np.max(rewards):.4f}")
        
        print(f"\n  Component averages:")
        for k, v_list in components_sum.items():
            if not k.startswith("_"):  # Skip internal metrics
                print(f"    {k:15s}: {np.mean(v_list):7.4f} ± {np.std(v_list):6.4f}")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all diagnostic tests."""
    print("\n" + "="*60)
    print("BIPED V2 DIAGNOSTIC TESTS")
    print("="*60)
    
    tests = [
        test_basic_setup,
        test_step_function,
        test_episode_rollout,
        test_policy_creation,
        test_reward_components,
    ]
    
    results = []
    for test_fn in tests:
        results.append(test_fn())
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Ready to train.")
        print("\nNext steps:")
        print("  1. Start with single-process training:")
        print("     python scripts/biped/train_biped_v2.py")
        print("\n  2. Or use multi-process for faster training:")
        print("     python scripts/biped/train_biped_v2_mp.py")
    else:
        print("\n✗ Some tests failed. Please fix issues before training.")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()