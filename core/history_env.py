# core/history_env.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from collections import deque

import numpy as np

from .base_env import Env, StepResult
from .specs import EnvSpec, SpaceSpec


@dataclass
class HistoryConfig:
    short_horizon: int = 4    # K_short in the paper
    long_horizon: int = 66    # K_long in the paper


class HistoryEnv(Env):
    """
    Wraps a base Env and augments observations with I/O history and an optional
    command vector.

    Observation layout:
        [ base_obs | short_history | long_history | command ]

    where
        short_history:  K_short * (obs_dim + act_dim)
        long_history:   K_long  * (obs_dim + act_dim)
    """

    def __init__(
        self,
        base_env: Env,
        hist_cfg: Optional[HistoryConfig] = None,
        command_dim: int = 0,
    ):
        self.base_env = base_env
        self.hist_cfg = hist_cfg or HistoryConfig()
        self.command_dim = command_dim

        # History buffers of (obs, action)
        self._short = deque(maxlen=self.hist_cfg.short_horizon)
        self._long = deque(maxlen=self.hist_cfg.long_horizon)

        # Dimensions from base env
        self.base_obs_dim = base_env.spec.obs.shape[0]
        self.act_dim = base_env.spec.act.shape[0]

        self.pair_dim = self.base_obs_dim + self.act_dim
        short_dim = self.hist_cfg.short_horizon * self.pair_dim
        long_dim = self.hist_cfg.long_horizon * self.pair_dim

        total_obs_dim = self.base_obs_dim + short_dim + long_dim + command_dim

        self.spec = EnvSpec(
            obs=SpaceSpec(shape=(total_obs_dim,), dtype=np.float32),
            act=base_env.spec.act,
        )

        # Current command (for now fixed during an episode; can be changed)
        if command_dim > 0:
            self._command = np.zeros(command_dim, dtype=np.float32)
        else:
            self._command = np.zeros(0, dtype=np.float32)

        # Just forward episode counter if needed; otherwise keep local
        self._t = 0

    # Small convenience: dt passthrough if base_env has it (MujocoEnv does)
    @property
    def dt(self) -> float:
        return getattr(self.base_env, "dt", 0.0)

    def set_command(self, command: np.ndarray) -> None:
        assert self.command_dim == command.shape[0]
        self._command = command.astype(np.float32)

    def _build_aug_obs(self, obs: np.ndarray) -> np.ndarray:
        # Short history
        short_list = list(self._short)
        if len(short_list) < self.hist_cfg.short_horizon:
            pad = self.hist_cfg.short_horizon - len(short_list)
            short_list = [(np.zeros(self.base_obs_dim, dtype=np.float32),
                           np.zeros(self.act_dim, dtype=np.float32))] * pad + short_list

        short_flat = []
        for o, a in short_list:
            short_flat.append(o)
            short_flat.append(a)
        short_flat = np.concatenate(short_flat, axis=0)

        # Long history
        long_list = list(self._long)
        if len(long_list) < self.hist_cfg.long_horizon:
            pad = self.hist_cfg.long_horizon - len(long_list)
            long_list = [(np.zeros(self.base_obs_dim, dtype=np.float32),
                          np.zeros(self.act_dim, dtype=np.float32))] * pad + long_list

        long_flat = []
        for o, a in long_list:
            long_flat.append(o)
            long_flat.append(a)
        long_flat = np.concatenate(long_flat, axis=0)

        return np.concatenate(
            [obs.astype(np.float32), short_flat, long_flat, self._command],
            axis=0,
        )

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._short.clear()
        self._long.clear()
        self._t = 0

        base_obs = self.base_env.reset(seed=seed)
        # At the very first step, history is all zeros
        aug_obs = self._build_aug_obs(base_obs)
        return aug_obs

    def step(self, action: np.ndarray) -> StepResult:
        # Call base env
        res = self.base_env.step(action)
        self._t += 1

        # Append new (obs, action) to history
        # Note: we store base obs, not augmented
        self._short.append((res.obs.copy(), action.copy()))
        self._long.append((res.obs.copy(), action.copy()))

        aug_obs = self._build_aug_obs(res.obs)

        # Forward info but keep obs replaced by augmented one
        info = dict(res.info)
        info["base_obs"] = res.obs  # optional, handy for debugging

        return StepResult(
            obs=aug_obs,
            reward=res.reward,
            done=res.done,
            info=info,
            # If your StepResult has 'frame', forward that too; if not, just omit
            frame=getattr(res, "frame", None),
        )

    def close(self) -> None:
        self.base_env.close()