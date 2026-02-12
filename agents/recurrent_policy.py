"""
Recurrent Policy Network for CLIP-RL (Hard BabyAI levels).

Architecture (following the original BabyAI paper):
    Instruction:  word tokens → Embedding(32) → biGRU(128) → (256,)
    Observation:  flat(148,) → Linear(128) → FiLM(conditioned on instr) → (128,)
    Memory:       FiLM output → GRUCell(hidden=256) → step hidden state
    Actor head:   hidden → Linear → 7 logits
    Critic head:  hidden → Linear → 1 scalar

FiLM (Feature-wise Linear Modulation):
    γ, β = Linear(instr_dim → film_dim) split
    output = γ ⊙ obs_features + β
    This conditions the visual processing on the instruction at every step.

Why this works better than flat MLP for hard levels:
    1. GRU memory: agent remembers where it's been (crucial for multi-room)
    2. FiLM: instruction actively shapes what features the agent attends to
    3. Step-by-step hidden state: works with TBPTT during training

The same architecture is used for both IL pretraining and RL fine-tuning.
It produces per-step (obs, action) pairs so it's compatible with the
existing trajectory/collector.py and the CLIP encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Optional, List
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import OBS_DIM, ACTION_DIM
from utils.il_config import RECURRENT_POLICY


# ─── Vocabulary ───────────────────────────────────────────────────────────────

def build_vocab(instructions: List[str]) -> dict:
    """
    Build a word-to-index vocabulary from a list of instruction strings.
    Includes special tokens: <pad>=0, <unk>=1.
    """
    vocab = {"<pad>": 0, "<unk>": 1}
    for instr in instructions:
        for word in instr.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def tokenize(instruction: str, vocab: dict, max_len: int = 20) -> torch.Tensor:
    """
    Tokenize a single instruction string to a padded LongTensor.
    Returns shape (max_len,).
    """
    tokens = [vocab.get(w, vocab["<unk>"]) for w in instruction.lower().split()]
    tokens = tokens[:max_len]
    tokens += [vocab["<pad>"]] * (max_len - len(tokens))
    return torch.LongTensor(tokens)


def tokenize_batch(instructions: List[str], vocab: dict, max_len: int = 20) -> torch.Tensor:
    """Returns (N, max_len) LongTensor."""
    return torch.stack([tokenize(i, vocab, max_len) for i in instructions])


# ─── Instruction Encoder ─────────────────────────────────────────────────────

class InstructionEncoder(nn.Module):
    """
    Encodes a variable-length word token sequence into a fixed-dim vector.

    word tokens → Embedding(word_emb_dim) → biGRU(instr_gru_dim) → last hidden
    Output: (batch, instr_dim) = (batch, 2 * instr_gru_dim)
    """

    def __init__(
        self,
        vocab_size:    int,
        word_emb_dim:  int = RECURRENT_POLICY["word_emb_dim"],   # 32
        gru_dim:       int = RECURRENT_POLICY["instr_gru_dim"],  # 128
    ):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, word_emb_dim, padding_idx=0)
        self.gru      = nn.GRU(
            input_size  = word_emb_dim,
            hidden_size = gru_dim,
            bidirectional = True,
            batch_first = True,
        )
        self.out_dim = gru_dim * 2   # 256

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (batch, seq_len) LongTensor
        Returns:
            (batch, 256) instruction embedding
        """
        emb = self.word_emb(tokens)            # (B, seq_len, 32)
        _, hidden = self.gru(emb)              # hidden: (2, B, 128) — fwd+bwd
        # Concatenate forward and backward final hidden states
        instr_emb = torch.cat([hidden[0], hidden[1]], dim=-1)  # (B, 256)
        return instr_emb


# ─── FiLM Layer ──────────────────────────────────────────────────────────────

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.
    Conditions a feature vector on a conditioning vector (the instruction).

    γ, β = Linear(cond_dim → 2 * feat_dim) split
    output = γ ⊙ features + β

    This is equivalent to a learned per-feature scale+shift conditioned on
    the instruction — much more expressive than just concatenating.
    """

    def __init__(self, feat_dim: int, cond_dim: int):
        super().__init__()
        self.gamma_beta = nn.Linear(cond_dim, feat_dim * 2)
        self.feat_dim   = feat_dim

    def forward(self, features: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, feat_dim)
            cond:     (batch, cond_dim)
        Returns:
            (batch, feat_dim)  — modulated features
        """
        gb     = self.gamma_beta(cond)                            # (B, 2*feat_dim)
        gamma  = gb[:, :self.feat_dim]                           # (B, feat_dim)
        beta   = gb[:, self.feat_dim:]                           # (B, feat_dim)
        return gamma * features + beta


# ─── Recurrent Policy ────────────────────────────────────────────────────────

class RecurrentPolicy(nn.Module):
    """
    FiLM-conditioned GRU policy for hard BabyAI levels.

    Can be used as a drop-in replacement for PolicyNetwork on tasks
    that require memory (multi-room navigation, sequential tasks, BossLevel).

    Forward pass (during TBPTT training):
        obs_seq:  (batch, seq_len, 148) — sequence of observations
        tokens:   (batch, 20)           — tokenized instruction (same for all steps)
        hidden:   (batch, memory_dim)   — initial GRU hidden state

    Returns:
        logits:      (batch, seq_len, 7)
        values:      (batch, seq_len, 1)
        new_hidden:  (batch, memory_dim)

    Single-step inference:
        Use act() which manages hidden state externally.
    """

    def __init__(
        self,
        vocab_size:  int,
        obs_dim:     int   = OBS_DIM,
        action_dim:  int   = ACTION_DIM,
        obs_proj_dim: int  = RECURRENT_POLICY["obs_proj_dim"],   # 128
        instr_dim:   int   = RECURRENT_POLICY["instr_dim"],      # 256
        film_dim:    int   = RECURRENT_POLICY["film_dim"],        # 128
        memory_dim:  int   = RECURRENT_POLICY["memory_dim"],     # 256 (or 2048)
        word_emb_dim: int  = RECURRENT_POLICY["word_emb_dim"],   # 32
        gru_dim:     int   = RECURRENT_POLICY["instr_gru_dim"],  # 128
    ):
        super().__init__()

        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.memory_dim = memory_dim

        # ── Instruction encoder ────────────────────────────────────────────────
        self.instr_encoder = InstructionEncoder(
            vocab_size   = vocab_size,
            word_emb_dim = word_emb_dim,
            gru_dim      = gru_dim,
        )
        # obs_proj_dim must equal film_dim
        assert obs_proj_dim == film_dim, \
            f"obs_proj_dim ({obs_proj_dim}) must equal film_dim ({film_dim})"

        # ── Obs projection ─────────────────────────────────────────────────────
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim, obs_proj_dim),
            nn.ReLU(),
        )

        # ── FiLM conditioning ──────────────────────────────────────────────────
        self.film = FiLM(feat_dim=film_dim, cond_dim=instr_dim)

        # ── Memory GRU ─────────────────────────────────────────────────────────
        # GRUCell processes one step at a time (needed for TBPTT and rollout)
        self.memory = nn.GRUCell(
            input_size  = film_dim,
            hidden_size = memory_dim,
        )

        # ── Actor + Critic heads ───────────────────────────────────────────────
        self.actor  = nn.Linear(memory_dim, action_dim)
        self.critic = nn.Linear(memory_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.actor.bias,  0.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Returns zero initial hidden state (batch, memory_dim)."""
        return torch.zeros(batch_size, self.memory_dim, device=device)

    def forward(
        self,
        obs_seq:    torch.Tensor,          # (batch, seq_len, 148)
        tokens:     torch.Tensor,          # (batch, 20) — same instr for whole seq
        hidden:     Optional[torch.Tensor] = None,  # (batch, memory_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a sequence of observations with shared instruction.
        Used during TBPTT training.

        Returns:
            logits:     (batch, seq_len, 7)
            values:     (batch, seq_len, 1)
            new_hidden: (batch, memory_dim)  — final hidden state
        """
        batch_size, seq_len, _ = obs_seq.shape
        device = obs_seq.device

        if hidden is None:
            hidden = self.init_hidden(batch_size, device)

        # Encode instruction once per sequence (same for all timesteps)
        instr_emb = self.instr_encoder(tokens)   # (batch, 256)

        all_logits = []
        all_values = []

        for t in range(seq_len):
            obs_t = obs_seq[:, t, :]              # (batch, 148)

            # Project obs
            obs_feat = self.obs_proj(obs_t)       # (batch, 128)

            # FiLM conditioning: modulate obs features with instruction
            modulated = self.film(obs_feat, instr_emb)  # (batch, 128)
            modulated = F.relu(modulated)

            # Update memory
            hidden = self.memory(modulated, hidden)  # (batch, memory_dim)

            # Heads
            all_logits.append(self.actor(hidden))    # (batch, 7)
            all_values.append(self.critic(hidden))   # (batch, 1)

        logits = torch.stack(all_logits, dim=1)      # (batch, seq_len, 7)
        values = torch.stack(all_values, dim=1)      # (batch, seq_len, 1)

        return logits, values, hidden

    def forward_step(
        self,
        obs:    torch.Tensor,   # (batch, 148)  — single timestep
        tokens: torch.Tensor,   # (batch, 20)
        hidden: torch.Tensor,   # (batch, memory_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step forward pass — used during rollout collection."""
        instr_emb = self.instr_encoder(tokens)         # (batch, 256)
        obs_feat  = self.obs_proj(obs)                 # (batch, 128)
        modulated = F.relu(self.film(obs_feat, instr_emb))  # (batch, 128)
        hidden    = self.memory(modulated, hidden)     # (batch, memory_dim)
        logits    = self.actor(hidden)                 # (batch, 7)
        value     = self.critic(hidden)                # (batch, 1)
        return logits, value, hidden

    def act(
        self,
        obs:           np.ndarray,    # (148,)
        tokens:        torch.Tensor,  # (1, 20)
        hidden:        torch.Tensor,  # (1, memory_dim)
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor]:
        """
        Select action for a single environment step.

        Returns:
            action: int in [0, 6]
            new_hidden: (1, memory_dim) — pass back in on next step
        """
        device = next(self.parameters()).device
        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)  # (1, 148)
        tokens = tokens.to(device)
        hidden = hidden.to(device)

        with torch.no_grad():
            logits, _, new_hidden = self.forward_step(obs_t, tokens, hidden)
            if deterministic:
                action = logits.argmax(dim=-1).item()
            else:
                dist   = Categorical(logits=logits)
                action = dist.sample().item()

        return int(action), new_hidden

    def evaluate_actions(
        self,
        obs_seq:    torch.Tensor,   # (batch, seq_len, 148)
        tokens:     torch.Tensor,   # (batch, 20)
        actions:    torch.Tensor,   # (batch, seq_len)
        hidden:     torch.Tensor,   # (batch, memory_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For PPO update step. Returns log_probs, entropy, values."""
        logits, values, _ = self.forward(obs_seq, tokens, hidden)  # (B, T, 7), (B, T, 1)

        # Flatten over batch and time for loss computation
        B, T, _ = logits.shape
        logits_flat  = logits.view(B * T, -1)         # (B*T, 7)
        actions_flat = actions.view(B * T)             # (B*T,)
        values_flat  = values.view(B * T, 1)           # (B*T, 1)

        dist      = Categorical(logits=logits_flat)
        log_probs = dist.log_prob(actions_flat)        # (B*T,)
        entropy   = dist.entropy().mean()              # scalar

        return log_probs, entropy, values_flat

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_recurrent_policy(
    instructions: List[str],
    large: bool = False,
) -> Tuple["RecurrentPolicy", dict]:
    """
    Convenience constructor: builds vocab from instructions and creates policy.

    Args:
        instructions: all instruction strings the policy will ever see
        large:        if True, use memory_dim=2048 (for BossLevel)

    Returns:
        policy: RecurrentPolicy
        vocab:  dict mapping word → index (needed for tokenization)
    """
    vocab      = build_vocab(instructions)
    memory_dim = RECURRENT_POLICY["memory_dim_large"] if large else RECURRENT_POLICY["memory_dim"]
    policy     = RecurrentPolicy(vocab_size=len(vocab), memory_dim=memory_dim)
    return policy, vocab


# ─── Main (smoke test) ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing RecurrentPolicy...\n")

    instrs = [
        "go to the red ball",
        "open the locked door",
        "put the ball next to the key",
        "boss level task",
    ]

    policy, vocab = make_recurrent_policy(instrs)
    print(f"Vocab size: {len(vocab)}")
    print(f"Policy params: {policy.param_count():,}")

    # Test sequence forward pass
    B, T = 4, 40
    obs_seq = torch.randn(B, T, OBS_DIM)
    tokens  = tokenize_batch(instrs, vocab)         # (4, 20)
    hidden  = policy.init_hidden(B, torch.device("cpu"))

    logits, values, new_hidden = policy(obs_seq, tokens, hidden)
    print(f"\nSequence forward:")
    print(f"  logits:     {logits.shape}       (should be [4, 40, 7])")
    print(f"  values:     {values.shape}       (should be [4, 40, 1])")
    print(f"  new_hidden: {new_hidden.shape}   (should be [4, 256])")

    # Test single step (rollout)
    obs_single = np.random.randn(OBS_DIM).astype(np.float32)
    tok_single = tokenize(instrs[0], vocab).unsqueeze(0)    # (1, 20)
    hid_single = policy.init_hidden(1, torch.device("cpu"))
    action, new_hid = policy.act(obs_single, tok_single, hid_single)
    print(f"\nSingle step act:")
    print(f"  action:    {action}             (should be int in [0, 6])")
    print(f"  new_hid:   {new_hid.shape}   (should be [1, 256])")

    print("\nAll checks passed!")