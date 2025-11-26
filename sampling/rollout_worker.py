# sampling/rollout_worker.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Callable

import numpy as np
import torch

from core.base_env import Env
from policies.actor_critic import ActorCritic

EnvFactory = Callable[[], Env]
PolicyFactory = Callable[[Any], ActorCritic]


@dataclass
class WorkerConfig:
    horizon: int = 1024


@dataclass
class RolloutBatch:
    obs: np.ndarray        # [T, obs_dim]
    actions: np.ndarray    # [T, act_dim]
    rewards: np.ndarray    # [T]
    dones: np.ndarray      # [T]
    log_probs: np.ndarray  # [T]
    last_obs: np.ndarray   # [obs_dim]


def worker_loop(
    worker_id: int,
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    params_queue,
    rollout_queue,
    cfg: WorkerConfig,
    device: str = "cpu",
):
    """
    Blocking loop:
    - Waits for params (state_dict) from params_queue
    - Runs rollout of length cfg.horizon
    - Puts RolloutBatch into rollout_queue
    - Repeats
    """

    # Important: factories must be top-level functions (picklable)
    env = env_factory()
    policy = policy_factory(env.spec).to(device)
    policy.eval()

    while True:
        msg = params_queue.get()  # block until we get a command
        if msg == "STOP":
            break

        state_dict = msg
        policy.load_state_dict(state_dict)

        batch = _collect_rollout(env, policy, cfg.horizon, device)
        rollout_queue.put((worker_id, batch))

    env.close()


def _collect_rollout(
    env: Env,
    policy: ActorCritic,
    horizon: int,
    device: str,
) -> RolloutBatch:
    obs_buf, act_buf, logp_buf, rew_buf, done_buf = [], [], [], [], []

    obs = env.reset()
    for _ in range(horizon):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t, logp_t = policy.act(obs_t, deterministic=False)

        action = action_t.squeeze(0).cpu().numpy()
        logp = logp_t.squeeze(0).cpu().numpy()

        step_res = env.step(action)

        obs_buf.append(obs)
        act_buf.append(action)
        logp_buf.append(logp)
        rew_buf.append(step_res.reward)
        done_buf.append(step_res.done)

        obs = step_res.obs if not step_res.done else env.reset()

    last_obs = obs

    return RolloutBatch(
        obs=np.array(obs_buf, dtype=np.float32),
        actions=np.array(act_buf, dtype=np.float32),
        rewards=np.array(rew_buf, dtype=np.float32),
        dones=np.array(done_buf, dtype=np.bool_),
        log_probs=np.array(logp_buf, dtype=np.float32),
        last_obs=np.array(last_obs, dtype=np.float32),
    )