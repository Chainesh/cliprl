"""
Trajectory Encoder for CLIP-RL.

Takes a set of (obs, action) sequences from one policy and produces
a single fixed-size embedding vector representing that policy's behavior.

Architecture:
    Per timestep: concat(obs_vec, action_onehot) → (155,)
    Step projection: Linear(155 → 128) + ReLU
    GRU(128, hidden=256, layers=2)
    Take final hidden state → (256,)
    Output projection: Linear(256 → 128) + L2 normalize

For multiple episodes per policy:
    Encode each episode independently → get N embedding vectors
    Mean pool across episodes → single (128,) policy embedding

This is the "policy tower" of our CLIP model.
Its output lives in the same 128-dim shared space as the language tower.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import TRAJ_ENCODER, OBS_DIM, ACTION_DIM, TRAJECTORY


class TrajectoryEncoder(nn.Module):
    """
    GRU-based encoder that maps a variable-length (obs, action) sequence
    to a fixed-size L2-normalized embedding vector.

    Input per episode:
        sequence of shape (T, 155) where T = episode length
        155 = OBS_DIM(148) + ACTION_DIM(7)

    Output:
        embedding of shape (128,) — L2 normalized unit vector

    For a full policy (multiple episodes):
        call encode_policy() which mean-pools across episodes
    """

    def __init__(
        self,
        input_dim:   int = TRAJ_ENCODER["input_dim"],    # 155
        proj_dim:    int = TRAJ_ENCODER["proj_dim"],     # 128
        gru_hidden:  int = TRAJ_ENCODER["gru_hidden"],   # 256
        gru_layers:  int = TRAJ_ENCODER["gru_layers"],   # 2
        embed_dim:   int = TRAJ_ENCODER["embed_dim"],    # 128
        dropout:     float = TRAJ_ENCODER["dropout"],    # 0.1
    ):
        super().__init__()

        self.input_dim  = input_dim
        self.proj_dim   = proj_dim
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.embed_dim  = embed_dim

        # ── Step projection: map each timestep to smaller dim ─────────────────
        # This compresses the 155-dim input before feeding to GRU,
        # reducing GRU parameter count and overfitting risk.
        self.step_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── GRU: processes the sequence ───────────────────────────────────────
        # batch_first=True means input shape is (batch, seq_len, features)
        self.gru = nn.GRU(
            input_size  = proj_dim,
            hidden_size = gru_hidden,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
        )

        # ── Output projection: GRU hidden → shared CLIP embedding space ───────
        self.output_proj = nn.Sequential(
            nn.Linear(gru_hidden, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier init for linear layers, orthogonal for GRU."""
        for name, param in self.named_parameters():
            if "gru" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)
            elif isinstance(param, nn.Linear):
                nn.init.xavier_uniform_(param.weight)
                nn.init.constant_(param.bias, 0)

    def forward_episode(
        self,
        sequence:  torch.Tensor,   # (T, 155) or (batch, T, 155)
        length:    Optional[torch.Tensor] = None,  # actual seq length for packing
    ) -> torch.Tensor:
        """
        Encode a single episode (or batch of episodes).

        Args:
            sequence: (batch, T, 155) padded sequence tensor
            length:   (batch,) actual lengths for pack_padded_sequence

        Returns:
            embedding: (batch, 128) L2-normalized
        """
        if sequence.dim() == 2:
            sequence = sequence.unsqueeze(0)   # add batch dim → (1, T, 155)

        batch_size, seq_len, _ = sequence.shape

        # Step projection: (batch, T, 155) → (batch, T, 128)
        projected = self.step_proj(sequence)

        # Pack padded sequence for efficient GRU (skip padding tokens)
        if length is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                projected,
                length.cpu(),
                batch_first = True,
                enforce_sorted = False,
            )
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(projected)

        # hidden: (gru_layers, batch, gru_hidden)
        # Take the last layer's hidden state
        last_hidden = hidden[-1]   # (batch, gru_hidden=256)

        # Project to embedding space
        embedding = self.output_proj(last_hidden)   # (batch, 128)

        # L2 normalize — important for cosine similarity in CLIP
        embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding   # (batch, 128)

    def encode_policy(
        self,
        trajectories_tensor: torch.Tensor,   # (N_episodes, T, 155)
        lengths:             torch.Tensor,   # (N_episodes,)
    ) -> torch.Tensor:
        """
        Encode a full policy from multiple episodes.
        Mean-pools episode embeddings into a single policy embedding.

        Args:
            trajectories_tensor: (N, T, 155)
            lengths:             (N,) actual episode lengths

        Returns:
            policy_embedding: (128,) single L2-normalized vector
        """
        # Encode all episodes: (N, 128)
        episode_embeddings = self.forward_episode(trajectories_tensor, lengths)

        # Mean pool across episodes
        policy_embedding = episode_embeddings.mean(dim=0)   # (128,)

        # Re-normalize after mean pooling
        policy_embedding = F.normalize(policy_embedding, p=2, dim=-1)

        return policy_embedding   # (128,)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"TrajectoryEncoder(\n"
            f"  step_proj: {self.input_dim} → {self.proj_dim}\n"
            f"  GRU:       {self.proj_dim} → hidden={self.gru_hidden} × {self.gru_layers} layers\n"
            f"  output:    {self.gru_hidden} → {self.embed_dim}\n"
            f"  params:    {self.param_count():,}\n"
            f")"
        )


# ─── Convenience: encode trajectories from collector output ──────────────────

def encode_trajectories(
    encoder:      TrajectoryEncoder,
    trajectories: list,                # List[Episode] from collector.py
    device:       str = "cpu",
) -> torch.Tensor:
    """
    Convenience wrapper: takes raw trajectories from collector.py
    and returns a single policy embedding.

    Args:
        encoder:      TrajectoryEncoder (trained or untrained)
        trajectories: list of episodes from collect_trajectories()
        device:       torch device

    Returns:
        policy_embedding: (128,) tensor
    """
    from trajectory.collector import trajectories_to_tensor

    inputs, lengths = trajectories_to_tensor(
        trajectories,
        max_ep_len = TRAJECTORY["max_steps"],
        device     = device,
    )

    encoder = encoder.to(device)
    encoder.eval()

    with torch.no_grad():
        embedding = encoder.encode_policy(inputs, lengths)

    return embedding   # (128,)


# ─── Main (smoke test) ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing TrajectoryEncoder...\n")

    encoder = TrajectoryEncoder()
    print(encoder)

    # ── Test 1: single episode ─────────────────────────────────────────────
    T          = 47   # arbitrary episode length
    single_ep  = torch.randn(1, T, TRAJ_ENCODER["input_dim"])
    lengths    = torch.tensor([T])
    emb        = encoder.forward_episode(single_ep, lengths)
    print(f"\nSingle episode:")
    print(f"  input:  {single_ep.shape}")
    print(f"  output: {emb.shape}   (should be [1, 128])")
    print(f"  L2 norm: {emb.norm().item():.4f}   (should be ~1.0)")

    # ── Test 2: batch of episodes (simulating one policy's trajectories) ───
    N          = 50   # episodes
    T_max      = 200  # max steps
    batch      = torch.randn(N, T_max, TRAJ_ENCODER["input_dim"])
    lengths    = torch.randint(10, T_max, (N,))
    policy_emb = encoder.encode_policy(batch, lengths)
    print(f"\nFull policy (50 episodes):")
    print(f"  input:  {batch.shape}")
    print(f"  output: {policy_emb.shape}   (should be [128])")
    print(f"  L2 norm: {policy_emb.norm().item():.4f}   (should be ~1.0)")

    # ── Test 3: two policies should get different embeddings ───────────────
    policy_emb_2 = encoder.encode_policy(
        torch.randn(N, T_max, TRAJ_ENCODER["input_dim"]),
        torch.randint(10, T_max, (N,)),
    )
    sim = F.cosine_similarity(policy_emb.unsqueeze(0), policy_emb_2.unsqueeze(0))
    print(f"\nCosine sim between two random policies: {sim.item():.4f}")
    print(f"  (random encoder → arbitrary similarity, expected to change after CLIP training)")

    print("\nAll checks passed!")