from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional

import numpy as np
import torch
import time

from core.base_env import Env
from algorithms.ppo import PPO, PPOConfig, compute_gae
from policies.actor_critic import ActorCritic

@dataclass
class TrainConfig:
    total_steps: int = 200_000
    horizon: int = 2048
    log_interval: int = 10
    device: str = "cpu"
    checkpoint_path: str = "checkpoints/model.pt"


EnvFactory = Callable[[], Env]
PolicyFactory = Callable[[Any], ActorCritic]
AlgoFactory = Callable[[ActorCritic], PPO]


class OnPolicyTrainer:
    """
    Generic single-env on-policy trainer.
    """
    def __init__(
        self,
        env_factory: EnvFactory,
        policy_factory: PolicyFactory,
        algo_factory: Optional[AlgoFactory] = None,
        train_cfg: Optional[TrainConfig] = None,
        ppo_cfg: Optional[PPOConfig] = None,
    ):
        self.env = env_factory()

        self.train_cfg = train_cfg or TrainConfig()
        self.device = self.train_cfg.device

        # Policy factory receives env.spec
        self.policy = policy_factory(self.env.spec).to(self.device)

        ppo_cfg = ppo_cfg or PPOConfig()
        self.algo = algo_factory(self.policy) if algo_factory else PPO(
            self.policy, ppo_cfg, device=self.device
        )

    # ---------------------------------------------------------
    # Rollout collection
    # ---------------------------------------------------------
    def _collect_trajectory(self) -> Dict[str, np.ndarray]:
        horizon = self.train_cfg.horizon
        env = self.env
        ac = self.policy
        device = self.device

        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

        obs = env.reset()
        for _ in range(horizon):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_tensor, logp_tensor = ac.act(obs_tensor, deterministic=False)
                value_tensor = ac.value(obs_tensor)

            action = action_tensor.squeeze(0).cpu().numpy()
            logp = logp_tensor.squeeze(0).cpu().numpy()
            value = value_tensor.squeeze(0).cpu().numpy()

            step_res = env.step(action)

            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp)
            rew_buf.append(step_res.reward)
            val_buf.append(value)
            done_buf.append(step_res.done)

            obs = step_res.obs if not step_res.done else env.reset()

        # Bootstrap last value
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            last_value = ac.value(obs_tensor).item()

        rewards = np.array(rew_buf, dtype=np.float32)
        values = np.array(val_buf, dtype=np.float32)
        dones = np.array(done_buf, dtype=np.bool_)

        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            last_value,
            gamma=self.algo.cfg.gamma,
            lam=self.algo.cfg.lam,
        )

        batch = {
            "obs": np.array(obs_buf, dtype=np.float32),
            "actions": np.array(act_buf, dtype=np.float32),
            "log_probs": np.array(logp_buf, dtype=np.float32),
            "advantages": advantages,
            "returns": returns,
        }
        return batch

    # ---------------------------------------------------------
    # Training loop WITH TIMING
    # ---------------------------------------------------------
    def run(self) -> None:
        total_steps = 0
        iteration = 0
        start_time = time.time()

        while total_steps < self.train_cfg.total_steps:
            iteration += 1

            iter_start = time.time()

            batch = self._collect_trajectory()
            steps_this_iter = batch["obs"].shape[0]
            total_steps += steps_this_iter

            metrics = self.algo.update(batch)

            iter_time = time.time() - iter_start
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / max(1e-6, elapsed)

            if iteration % self.train_cfg.log_interval == 0:
                avg_return = batch["returns"].mean()
                print(
                    f"[Iter {iteration}]  "
                    f"Steps={total_steps}  "
                    f"IterTime={iter_time:.2f}s  "
                    f"Steps/sec={steps_per_sec:.1f}  "
                    f"Return={avg_return:.2f}  "
                    f"pi_loss={metrics['policy_loss']:.3f}  "
                    f"v_loss={metrics['value_loss']:.3f}  "
                    f"entropy={metrics['entropy']:.3f}"
                )

        # -----------------------------------------------------
        # Final summary
        # -----------------------------------------------------
        total_time = time.time() - start_time
        steps_per_sec = total_steps / max(1e-6, total_time)

        print("\n=================== Training finished ===================")
        print(f"Total steps:        {total_steps}")
        print(f"Total time:         {total_time:.2f} seconds")
        print(f"Average steps/sec:  {steps_per_sec:.1f}")
        print("=========================================================\n")

        # Save final checkpoint
        if self.train_cfg.checkpoint_path:
            import os
            os.makedirs(os.path.dirname(self.train_cfg.checkpoint_path), exist_ok=True)
            torch.save(self.policy.state_dict(), self.train_cfg.checkpoint_path)

        self.env.close()