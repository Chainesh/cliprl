"""
Transfer Initializer for CLIP-RL.

Given a new (target) task instruction, finds the most similar base tasks
using the trained CLIP model, and initializes the target policy as a
weighted average of base policies.

Two modes compared:
    1. Language-only  : cosine similarity on raw SBERT embeddings (Algorithm 1)
    2. CLIP-RL        : cosine similarity in CLIP-aligned space   (Algorithm 2)
"""

import torch
import torch.nn.functional as F
import numpy as np
import os, sys
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.policy_network import PolicyNetwork, weighted_average_init
from utils.config import CHECKPOINTS


def softmax_weights(similarities: torch.Tensor, temperature: float = 1.0) -> List[float]:
    """
    Convert similarity scores to normalized weights via softmax.

    Args:
        similarities: (N,) cosine similarity scores
        temperature:  sharpness — lower = more weight on the top match

    Returns:
        weights: List[float] summing to 1.0
    """
    scaled  = similarities / temperature
    weights = F.softmax(scaled, dim=0)
    return weights.tolist()


def load_base_policies(device: str = "cpu") -> Tuple[List[PolicyNetwork], List[str], List[str]]:
    """
    Load all trained base policies from checkpoints.

    Returns:
        policies:     List[PolicyNetwork]
        task_ids:     List[str]
        instructions: List[str]
    """
    from envs.task_suite import TaskRegistry
    from agents.ppo_agent import PPOAgent

    registry     = TaskRegistry()
    policies     = []
    task_ids     = []
    instructions = []

    for env_id, instruction, task_id in registry.base_tasks():
        ckpt_path = os.path.join(CHECKPOINTS, f"policy_{task_id}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Policy checkpoint not found: {ckpt_path}\n"
                f"Run phase 1 (train) first."
            )
        agent = PPOAgent.load(task_id=task_id, env_id=env_id, device=device)
        policies.append(agent.policy)
        task_ids.append(task_id)
        instructions.append(instruction)

    print(f"  Loaded {len(policies)} base policies")
    return policies, task_ids, instructions


# ─── Algorithm 1: Language-only transfer ─────────────────────────────────────

def initialize_language_only(
    target_instruction: str,
    base_policies:      List[PolicyNetwork],
    base_instructions:  List[str],
    verbose:            bool = True,
) -> PolicyNetwork:
    """
    Algorithm 1: initialize target policy using raw SBERT similarity.
    No CLIP alignment — just cosine similarity on sentence embeddings.

    Args:
        target_instruction: instruction for the new task
        base_policies:      list of trained base PolicyNetworks
        base_instructions:  list of base task instructions

    Returns:
        new PolicyNetwork initialized by weighted average
    """
    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    with torch.no_grad():
        all_instructions = base_instructions + [target_instruction]
        embeddings = sbert.encode(all_instructions, convert_to_tensor=True)
        base_embs  = embeddings[:-1]    # (N, 384)
        target_emb = embeddings[-1]     # (384,)

        # Cosine similarities
        sims = F.cosine_similarity(
            target_emb.unsqueeze(0).expand(len(base_instructions), -1),
            base_embs
        )   # (N,)

    weights = softmax_weights(sims)

    if verbose:
        print(f"\n  Language-only transfer weights for: '{target_instruction}'")
        for instr, w, s in zip(base_instructions, weights, sims.tolist()):
            print(f"    {w:.3f}  (sim={s:.3f})  {instr}")

    return weighted_average_init(base_policies, weights)


# ─── Algorithm 2: CLIP-RL transfer ───────────────────────────────────────────

def initialize_clip_rl(
    target_instruction: str,
    base_policies:      List[PolicyNetwork],
    base_instructions:  List[str],
    clip_model,                             # CLIPModel
    verbose:            bool = True,
) -> PolicyNetwork:
    """
    Algorithm 2: initialize target policy using CLIP-aligned similarity.
    Uses the trained CLIP projection to find structurally similar tasks.

    Args:
        target_instruction: instruction for the new task
        base_policies:      list of trained base PolicyNetworks
        base_instructions:  list of base task instructions
        clip_model:         trained CLIPModel

    Returns:
        new PolicyNetwork initialized by weighted average
    """
    clip_model.eval()
    device = clip_model.device

    with torch.no_grad():
        # Encode all instructions through CLIP-aligned language encoder
        all_instructions = base_instructions + [target_instruction]
        all_embs         = clip_model.lang_encoder.encode_batch(all_instructions)
        all_embs         = all_embs.to(device)

        base_embs   = all_embs[:-1]   # (N, 128)
        target_emb  = all_embs[-1]    # (128,)

        # Cosine similarities in CLIP-aligned space
        sims = F.cosine_similarity(
            target_emb.unsqueeze(0).expand(len(base_instructions), -1),
            base_embs
        )   # (N,)

    weights = softmax_weights(sims)

    if verbose:
        print(f"\n  CLIP-RL transfer weights for: '{target_instruction}'")
        for instr, w, s in zip(base_instructions, weights, sims.tolist()):
            print(f"    {w:.3f}  (sim={s:.3f})  {instr}")

    return weighted_average_init(base_policies, weights)


# ─── Random baseline ──────────────────────────────────────────────────────────

def initialize_random() -> PolicyNetwork:
    """Baseline: random initialization (no transfer)."""
    return PolicyNetwork()


# ─── Top-1 oracle (upper bound) ───────────────────────────────────────────────

def initialize_top1_language(
    target_instruction: str,
    base_policies:      List[PolicyNetwork],
    base_instructions:  List[str],
) -> PolicyNetwork:
    """
    Hard assignment: just use the single most similar policy (no averaging).
    Useful as a comparison to understand if averaging helps.
    """

    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    with torch.no_grad():
        all_embs  = sbert.encode(base_instructions + [target_instruction],
                                  convert_to_tensor=True)
        base_embs = all_embs[:-1]
        tgt_emb   = all_embs[-1]
        sims = F.cosine_similarity(tgt_emb.unsqueeze(0).expand(len(base_instructions), -1),
                                    base_embs)
        best_idx  = sims.argmax().item()

    print(f"\n  Top-1 language match: '{base_instructions[best_idx]}'")
    # One-hot weights
    weights = [0.0] * len(base_policies)
    weights[best_idx] = 1.0
    return weighted_average_init(base_policies, weights)