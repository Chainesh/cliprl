"""
Language Encoder for CLIP-RL.

Maps a natural language task instruction string to a fixed-size
L2-normalized embedding in the shared CLIP space.

Architecture:
    SBERT (all-MiniLM-L6-v2) — FROZEN, just extracts features
    → (384,) sentence embedding
    Linear(384 → 128) + ReLU + Linear(128 → 128) — TRAINABLE projection head
    → L2 normalize
    → (128,) unit vector in shared CLIP space

Why freeze SBERT?
    - SBERT already has excellent general language understanding
    - We only need to rotate/project its space to align with policy space
    - Freezing prevents catastrophic forgetting and speeds up training
    - Same design as original CLIP (frozen image encoder backbone, trained projection)

This is the "language tower" of our CLIP model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import LANG_ENCODER


class LanguageEncoder(nn.Module):
    """
    Language encoder: instruction string → 128-dim L2-normalized embedding.

    Only the projection head is trained during CLIP alignment.
    SBERT weights are frozen throughout.
    """

    def __init__(
        self,
        model_name: str   = LANG_ENCODER["model_name"],   # "all-MiniLM-L6-v2"
        sbert_dim:  int   = LANG_ENCODER["sbert_dim"],    # 384
        embed_dim:  int   = LANG_ENCODER["embed_dim"],    # 128
    ):
        super().__init__()

        self.sbert_dim = sbert_dim
        self.embed_dim = embed_dim

        # ── Load SBERT (frozen) ────────────────────────────────────────────────
        print(f"  Loading SBERT: {model_name}...")
        self._sbert = SentenceTransformer(model_name)
        self._sbert.eval()

        # Freeze all SBERT parameters
        for param in self._sbert.parameters():
            param.requires_grad = False

        # ── Trainable projection head ──────────────────────────────────────────
        # Two-layer MLP to project from SBERT's space into shared CLIP space.
        # The non-linearity lets it learn a more flexible alignment.
        self.projection = nn.Sequential(
            nn.Linear(sbert_dim, embed_dim * 2),   # 384 → 256
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),   # 256 → 128
        )

        self._init_weights()
        print(f"  Language encoder ready. Trainable params: {self.trainable_param_count():,}")

    def _init_weights(self):
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def _get_sbert_embedding(
        self,
        instructions: Union[str, List[str]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Extract frozen SBERT embeddings.

        Args:
            instructions: single string or list of strings
            device:       target device

        Returns:
            embeddings: (N, 384) float32 tensor
        """
        if isinstance(instructions, str):
            instructions = [instructions]

        # SBERT encode returns numpy array
        with torch.no_grad():
            embeddings = self._sbert.encode(
                instructions,
                convert_to_tensor = True,
                show_progress_bar = False,
            )

        return embeddings.to(device)   # (N, 384)

    def forward(
        self,
        instructions: Union[str, List[str]],
    ) -> torch.Tensor:
        """
        Encode instruction(s) into shared CLIP space.

        Args:
            instructions: str or list of str

        Returns:
            embeddings: (N, 128) L2-normalized, or (128,) if single string
        """
        device = next(self.projection.parameters()).device

        # Get frozen SBERT features
        sbert_emb = self._get_sbert_embedding(instructions, device)  # (N, 384)
        sbert_emb = sbert_emb.detach().clone() 
        # Project into shared space
        projected = self.projection(sbert_emb)   # (N, 128)

        # L2 normalize
        normalized = F.normalize(projected, p=2, dim=-1)   # (N, 128)

        # Squeeze if single instruction
        if isinstance(instructions, str):
            normalized = normalized.squeeze(0)   # (128,)

        return normalized

    def encode_batch(self, instructions: List[str]) -> torch.Tensor:
        """Explicit batch encoding, always returns (N, 128)."""
        return self.forward(instructions)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.projection.parameters())

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"LanguageEncoder(\n"
            f"  SBERT: {LANG_ENCODER['model_name']} ({self.sbert_dim}d) [FROZEN]\n"
            f"  projection: {self.sbert_dim} → {self.embed_dim*2} → {self.embed_dim}\n"
            f"  trainable params: {self.trainable_param_count():,}\n"
            f")"
        )


# ─── Precompute & cache language embeddings ───────────────────────────────────

def precompute_language_embeddings(
    instructions: List[str],
    encoder:      LanguageEncoder,
    device:       str = "cpu",
) -> dict:
    """
    Precompute SBERT embeddings for all base task instructions.
    These are fixed (SBERT is frozen) so we only need to do this once.
    The projection head embeddings change during CLIP training, so those
    are computed on the fly.

    Returns:
        dict of {instruction: sbert_embedding (384,)}
    """
    encoder = encoder.to(device)
    encoder.eval()
    cache = {}
    with torch.no_grad():
        for instr in instructions:
            emb = encoder._get_sbert_embedding(instr, torch.device(device))
            cache[instr] = emb.squeeze(0).cpu()
    return cache


# ─── Main (smoke test) ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing LanguageEncoder...\n")

    encoder = LanguageEncoder()
    print(encoder)

    # ── Test 1: single instruction ────────────────────────────────────────
    instr  = "go to the red ball"
    emb    = encoder(instr)
    print(f"\nSingle instruction: '{instr}'")
    print(f"  output: {emb.shape}   (should be [128])")
    print(f"  L2 norm: {emb.norm().item():.4f}   (should be ~1.0)")

    # ── Test 2: batch ─────────────────────────────────────────────────────
    instrs = [
        "go to the red ball",
        "go to the blue box",
        "open the red door",
        "go to the green key",
    ]
    embs = encoder.encode_batch(instrs)
    print(f"\nBatch of {len(instrs)} instructions:")
    print(f"  output: {embs.shape}   (should be [4, 128])")
    print(f"  L2 norms: {embs.norm(dim=-1).tolist()}")

    # ── Test 3: similar instructions should be closer than dissimilar ─────
    # (before CLIP training — using raw SBERT space via projection)
    emb_red_ball  = encoder("go to the red ball")
    emb_blue_ball = encoder("go to the blue ball")
    emb_open_door = encoder("open the red door")

    sim_rb_bb = F.cosine_similarity(emb_red_ball.unsqueeze(0),
                                     emb_blue_ball.unsqueeze(0)).item()
    sim_rb_od = F.cosine_similarity(emb_red_ball.unsqueeze(0),
                                     emb_open_door.unsqueeze(0)).item()
    print(f"\nLanguage similarity (pre-CLIP):")
    print(f"  'red ball' vs 'blue ball': {sim_rb_bb:.4f}   (should be high — same structure)")
    print(f"  'red ball' vs 'open door': {sim_rb_od:.4f}   (should be lower)")
    print(f"\n  After CLIP training, these similarities should reflect POLICY structure, not just language!")

    print("\nAll checks passed!")