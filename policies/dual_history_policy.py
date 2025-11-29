# policies/dual_history_policy.py
from __future__ import annotations
from typing import Tuple
import torch
from torch import nn
import torch.distributions as D
from typing import Tuple

from core.specs import EnvSpec


class DualHistoryActorCritic(nn.Module):
    """
    https://arxiv.org/abs/2401.16889
    Dual-history policy:
    - Long history encoded with 1D CNN
    - Short history + current obs + command go directly to base MLP
    - Output: Gaussian over normalized actions (tanh on mean)
    """

    def __init__(
        self,
        spec: EnvSpec,
        base_obs_dim: int,
        short_hist_len: int,
        long_hist_len: int,
        command_dim: int,
        hidden_size: int = 512,
        act_std: float = 0.1,
    ):
        super().__init__()
        self.spec = spec
        self.base_obs_dim = base_obs_dim
        self.short_hist_len = short_hist_len
        self.long_hist_len = long_hist_len
        self.command_dim = command_dim

        act_dim = spec.act.shape[0]
        pair_dim = base_obs_dim + act_dim

        short_dim = short_hist_len * pair_dim
        long_dim = long_hist_len * pair_dim

        # CNN expects (B, C, T). We'll set C=pair_dim, T=long_hist_len.
        self.long_channels = pair_dim
        self.long_steps = long_hist_len
        assert long_dim == self.long_channels * self.long_steps

        # CNN encoder as in the paper: [6,32,3] and [4,16,2]
        self.long_encoder = nn.Sequential(
            nn.Conv1d(self.long_channels, 32, kernel_size=6, stride=3),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size with a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, self.long_channels, self.long_steps)
            cnn_out_dim = self.long_encoder(dummy).shape[-1]

        base_input_dim = base_obs_dim + short_dim + command_dim + cnn_out_dim

        self.base_net = nn.Sequential(
            nn.Linear(base_input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.mu_head = nn.Linear(hidden_size, act_dim)
        self.v_head = nn.Linear(hidden_size, 1)

        # log_std is state-independent (like in your simple ActorCritic)
        self.log_std = nn.Parameter(
            torch.full(
                (act_dim,),
                float(torch.log(torch.tensor(act_std)))
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _split_obs(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs: (B, total_obs_dim) = [base_obs | short | long | command]
        """
        B, D = obs.shape
        base = self.base_obs_dim
        act_dim = self.spec.act.shape[0]
        pair_dim = self.base_obs_dim + act_dim

        short_dim = self.short_hist_len * pair_dim
        long_dim = self.long_hist_len * pair_dim

        base_obs = obs[:, :base]
        short_hist = obs[:, base:base + short_dim]
        long_hist = obs[:, base + short_dim:base + short_dim + long_dim]
        cmd = obs[:, base + short_dim + long_dim:]

        return base_obs, short_hist, long_hist, cmd

    def _forward_body(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Shared trunk: encode long history with CNN, then concatenate
        base_obs, short_hist, command, long_emb and run through base_net.
        Returns features h used by both policy and value heads.
        """
        base_obs, short_hist, long_hist, cmd = self._split_obs(obs)

        B = obs.shape[0]
        pair_dim = self.long_channels

        # reshape long_hist to (B, C, T)
        long_hist_reshaped = long_hist.view(B, pair_dim, self.long_steps)
        long_emb = self.long_encoder(long_hist_reshaped)

        x = torch.cat([base_obs, short_hist, cmd, long_emb], dim=-1)
        h = self.base_net(x)
        return h

    # ------------------------------------------------------------------
    # The _dist method PPO needs
    # ------------------------------------------------------------------
    def _dist(self, obs: torch.Tensor):
        """
        Construct action distribution and return (dist, features).
        Matches the pattern used by ActorCritic so PPO can call it.
        """
        h = self._forward_body(obs)
        mu = torch.tanh(self.mu_head(h))
        std = self.log_std.exp()
        dist = D.Normal(mu, std)
        return dist, h

    # ------------------------------------------------------------------
    # High-level API: forward / act / value
    # ------------------------------------------------------------------
    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns mu, value, log_std (shared across batch).
        Mainly useful if you want to use it directly.
        """
        h = self._forward_body(obs)
        mu = torch.tanh(self.mu_head(h))
        value = self.v_head(h).squeeze(-1)
        return mu, value, self.log_std

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        For rollout: returns (action, log_prob)
        """
        dist, _ = self._dist(obs)
        if deterministic:
            action = dist.loc
        else:
            action = dist.rsample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp

    def value(self, obs: torch.Tensor):
        """
        Value function V(s)
        """
        h = self._forward_body(obs)
        return self.v_head(h).squeeze(-1)