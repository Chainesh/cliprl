"""
CLIP-RL Model

Combines the language tower and trajectory tower into a single model.
Computes the InfoNCE contrastive loss over (instruction, policy) pairs.

The CLIP loss:
    Given N (instruction, policy) pairs, build N×N similarity matrix S where
        S[i,j] = dot(proj_lang[i], proj_traj[j])

    Diagonal entries S[i,i] are positive pairs (same task).
    Off-diagonal S[i,j] (i≠j) are negative pairs (different tasks).

    Loss pulls diagonal together and pushes off-diagonal apart:
        L = -1/2N * Σᵢ log[ exp(S[i,i]/δ) / Σⱼ exp(S[i,j]/δ) ]
          - 1/2N * Σⱼ log[ exp(S[j,j]/δ) / Σᵢ exp(S[i,j]/δ) ]

    Temperature δ is a learnable scalar (initialized from config).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoders.language_encoder    import LanguageEncoder
from encoders.trajectory_encoder  import TrajectoryEncoder
from utils.config import CLIP_TRAIN, LANG_ENCODER, TRAJ_ENCODER


class CLIPModel(nn.Module):
    """
    CLIP-RL: aligns language instructions with policy trajectories.

    Contains:
        lang_encoder  : LanguageEncoder  (SBERT frozen + trainable projection)
        traj_encoder  : TrajectoryEncoder (GRU + trainable projection)
        log_temp      : learnable log temperature (log δ)

    Forward:
        Given batch of instructions + trajectory tensors,
        returns the N×N similarity matrix and the InfoNCE loss.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()

        self.device = torch.device(device)

        # ── Two towers ────────────────────────────────────────────────────────
        self.lang_encoder = LanguageEncoder()
        self.traj_encoder = TrajectoryEncoder()

        # ── Learnable temperature ─────────────────────────────────────────────
        # We store log(temperature) for numerical stability.
        # Initialized so that temperature = CLIP_TRAIN["temperature"] = 0.07
        init_log_temp = np.log(CLIP_TRAIN["temperature"])
        if CLIP_TRAIN["learn_temp"]:
            self.log_temp = nn.Parameter(torch.tensor(init_log_temp, dtype=torch.float32))
        else:
            self.register_buffer("log_temp", torch.tensor(init_log_temp, dtype=torch.float32))

        self.to(self.device)

    @property
    def temperature(self) -> float:
        return self.log_temp.exp().item()

    def encode_instructions(self, instructions: list) -> torch.Tensor:
        """
        Encode a list of instruction strings → (N, 128) normalized embeddings.
        """
        return self.lang_encoder.encode_batch(instructions)   # (N, 128)

    def encode_trajectories(
        self,
        traj_tensors: list,   # List of (n_ep, T, 155) tensors, one per task
        traj_lengths: list,   # List of (n_ep,) tensors
    ) -> torch.Tensor:
        """
        Encode trajectories for N tasks → (N, 128) normalized embeddings.
        Each task's episodes are mean-pooled into one embedding.
        """
        embeddings = []
        for traj_t, lengths in zip(traj_tensors, traj_lengths):
            emb = self.traj_encoder.encode_policy(
                traj_t.to(self.device),
                lengths.to(self.device),
            )  # (128,)
            embeddings.append(emb)
        return torch.stack(embeddings, dim=0)   # (N, 128)

    def similarity_matrix(
        self,
        lang_embs: torch.Tensor,   # (N, 128)
        traj_embs: torch.Tensor,   # (N, 128)
    ) -> torch.Tensor:
        """
        Compute scaled cosine similarity matrix.
        Since embeddings are L2-normalized, dot product = cosine similarity.

        Returns:
            S: (N, N) matrix where S[i,j] = sim(lang[i], traj[j]) / temperature
        """
        temp = self.log_temp.exp()
        S = torch.matmul(lang_embs, traj_embs.T) / temp   # (N, N)
        return S

    def infonce_loss(self, S: torch.Tensor) -> torch.Tensor:
        """
        Symmetric InfoNCE loss over similarity matrix S.

        Args:
            S: (N, N) similarity matrix (already scaled by temperature)

        Returns:
            loss: scalar
        """
        N      = S.shape[0]
        labels = torch.arange(N, device=S.device)   # [0, 1, 2, ..., N-1]

        # Language → Trajectory direction: each row, correct col is diagonal
        loss_lang = F.cross_entropy(S, labels)

        # Trajectory → Language direction: each col, correct row is diagonal
        loss_traj = F.cross_entropy(S.T, labels)

        return (loss_lang + loss_traj) / 2.0

    def forward(
        self,
        instructions: list,
        traj_tensors: list,
        traj_lengths: list,
    ) -> dict:
        """
        Full forward pass: encode both modalities, compute loss.

        Args:
            instructions: List[str] of length N
            traj_tensors: List of (n_ep, T, 155) tensors
            traj_lengths: List of (n_ep,) tensors

        Returns:
            dict with keys: loss, similarity_matrix, lang_embs, traj_embs
        """
        # Encode both modalities → (N, 128) each
        lang_embs = self.encode_instructions(instructions)
        traj_embs = self.encode_trajectories(traj_tensors, traj_lengths)

        # Move lang_embs to same device as traj (SBERT may return different device)
        lang_embs = lang_embs.to(self.device)

        # Similarity matrix
        S = self.similarity_matrix(lang_embs, traj_embs)

        # Loss
        loss = self.infonce_loss(S)

        return {
            "loss"             : loss,
            "similarity_matrix": S.detach(),
            "lang_embs"        : lang_embs.detach(),
            "traj_embs"        : traj_embs.detach(),
            "temperature"      : self.temperature,
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "lang_encoder_projection": self.lang_encoder.projection.state_dict(),
            "traj_encoder"           : self.traj_encoder.state_dict(),
            "log_temp"               : self.log_temp.data,
        }, path)
        print(f"  Saved CLIP model → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.lang_encoder.projection.load_state_dict(ckpt["lang_encoder_projection"])
        self.traj_encoder.load_state_dict(ckpt["traj_encoder"])
        self.log_temp.data = ckpt["log_temp"]
        print(f"  Loaded CLIP model ← {path}")
        return self

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)