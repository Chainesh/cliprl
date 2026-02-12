"""
Policy network for CLIP-RL.

Architecture:
    Input  (148,) → Linear → ReLU → Linear → ReLU → Actor head + Critic head

This is a simple MLP shared across ALL tasks.
The same architecture is used for every base task and every target task.
This is important because:
  - weighted average initialization (the transfer mechanism) only makes sense
    if all policies share identical architecture
  - the trajectory encoder needs consistent (obs, action) inputs regardless of task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import OBS_DIM, ACTION_DIM, POLICY_NET


class PolicyNetwork(nn.Module):
    """
    Actor-Critic policy network.

    Shared trunk:
        Linear(148 → 256) → ReLU → Linear(256 → 128) → ReLU

    Actor head (policy):
        Linear(128 → 7) → Softmax  → action probabilities

    Critic head (value function):
        Linear(128 → 1) → scalar state value V(s)

    The trunk is shared to encourage learning useful features for both
    policy and value estimation (standard PPO practice).
    """

    def __init__(
        self,
        obs_dim:     int = OBS_DIM,
        action_dim:  int = ACTION_DIM,
        hidden_dims: list = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = POLICY_NET["hidden_dims"]  # [256, 128]

        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        # ── Shared trunk ──────────────────────────────────────────────────────
        trunk_layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            trunk_layers.append(nn.Linear(in_dim, h_dim))
            trunk_layers.append(nn.ReLU())
            in_dim = h_dim
        self.trunk = nn.Sequential(*trunk_layers)
        self.trunk_out_dim = hidden_dims[-1]  # 128

        # ── Actor head ────────────────────────────────────────────────────────
        self.actor = nn.Linear(self.trunk_out_dim, action_dim)

        # ── Critic head ───────────────────────────────────────────────────────
        self.critic = nn.Linear(self.trunk_out_dim, 1)

        # ── Weight initialization ─────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal initialization — standard for PPO.
        Actor head uses smaller scale to start with near-uniform policy.
        Critic head uses larger scale.
        """
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)

        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: (batch, 148) float32 tensor

        Returns:
            action_logits: (batch, 7)  — raw logits for actor
            value:         (batch, 1)  — state value estimate
        """
        features      = self.trunk(obs)
        action_logits = self.actor(features)
        value         = self.critic(features)
        return action_logits, value

    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns trunk features (128,) — used for analysis/visualization.
        """
        return self.trunk(obs)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        Select an action given a single observation (numpy).

        Args:
            obs:          (148,) float32 numpy array
            deterministic: if True, take argmax; else sample from distribution

        Returns:
            action: int in [0, 6]
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0)  # (1, 148)
        with torch.no_grad():
            logits, _ = self.forward(obs_t)
            if deterministic:
                action = logits.argmax(dim=-1).item()
            else:
                dist   = Categorical(logits=logits)
                action = dist.sample().item()
        return action

    def evaluate_actions(
        self,
        obs:     torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probs, entropy, and value for given (obs, action) pairs.
        Used in PPO update step.

        Args:
            obs:     (batch, 148)
            actions: (batch,) int64

        Returns:
            log_probs: (batch,)
            entropy:   scalar
            values:    (batch, 1)
        """
        logits, values = self.forward(obs)
        dist      = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy().mean()
        return log_probs, entropy, values

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"PolicyNetwork(\n"
            f"  obs_dim={self.obs_dim}, action_dim={self.action_dim}\n"
            f"  trunk={self.trunk}\n"
            f"  actor={self.actor}\n"
            f"  critic={self.critic}\n"
            f"  total_params={self.param_count():,}\n"
            f")"
        )


# ─── Weighted Average Initialization ─────────────────────────────────────────

def weighted_average_init(
    policies: list,              # list of PolicyNetwork
    weights:  list,              # list of float (must sum to 1.0)
) -> PolicyNetwork:
    """
    Creates a new PolicyNetwork whose weights are the weighted average
    of the provided policies. This is the core transfer mechanism.

    Args:
        policies: list of N trained PolicyNetwork instances
        weights:  list of N floats (normalized similarity scores)

    Returns:
        new_policy: PolicyNetwork initialized with weighted average weights
    """
    assert len(policies) == len(weights), "policies and weights must match"
    assert abs(sum(weights) - 1.0) < 1e-5, f"weights must sum to 1, got {sum(weights)}"

    new_policy = PolicyNetwork()
    new_state  = {}

    # Get parameter names from first policy
    ref_state = policies[0].state_dict()

    for param_name in ref_state.keys():
        # weighted sum of tensors for each parameter
        weighted_param = sum(
            w * p.state_dict()[param_name].float()
            for p, w in zip(policies, weights)
        )
        new_state[param_name] = weighted_param

    new_policy.load_state_dict(new_state)
    return new_policy


# ─── Main (smoke test) ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing PolicyNetwork...")

    net = PolicyNetwork()
    print(net)

    # test forward pass
    batch_obs = torch.randn(8, OBS_DIM)
    logits, values = net(batch_obs)
    print(f"\nForward pass:")
    print(f"  input:   {batch_obs.shape}")
    print(f"  logits:  {logits.shape}   (should be [8, 7])")
    print(f"  values:  {values.shape}   (should be [8, 1])")

    # test single act
    single_obs = np.random.randn(OBS_DIM).astype(np.float32)
    action = net.act(single_obs)
    print(f"\nSingle action: {action}  (should be int in [0, 6])")

    # test weighted average
    p1 = PolicyNetwork()
    p2 = PolicyNetwork()
    p_avg = weighted_average_init([p1, p2], weights=[0.6, 0.4])
    print(f"\nWeighted average policy created: {p_avg.param_count():,} params")

    print("\nAll tests passed!")