"""
Imitation Learning (Behavioral Cloning) Agent for CLIP-RL.

Trains the RecurrentPolicy to mimic bot demonstrations via cross-entropy
loss on action predictions. No reward signal needed.

Training loop:
    for each epoch:
        for each batch of (obs_seq, action_seq) from demos:
            hidden = zeros
            logits, _, _ = policy(obs_seq, tokens, hidden)
            loss = cross_entropy(logits.view(-1, 7), actions.view(-1))
            loss.backward()
            optimizer.step()

After IL pretraining, the policy is saved in the same checkpoint format
as PPOAgent so the rest of the CLIP-RL pipeline can load it transparently.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, sys, time, pickle
from typing import Optional, List, Dict, Tuple
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.recurrent_policy import (
    RecurrentPolicy, build_vocab, tokenize_batch, make_recurrent_policy
)
from trajectory.bot_collector import (
    Demos, load_demos, demos_to_sequences, demos_exist
)
from utils.il_config import IL, RECURRENT_POLICY, CHECKPOINTS, get_stage_demo_count


# ─── IL Agent ────────────────────────────────────────────────────────────────

class ILAgent:
    """
    Behavioral Cloning agent.

    Wraps RecurrentPolicy with an IL training loop on bot demonstrations.
    Saves checkpoints compatible with the main CLIP-RL pipeline.

    Usage:
        agent = ILAgent(
            env_id      = "BabyAI-BossLevel-v0",
            task_id     = "BossLevel__boss_level_task",
            instruction = "boss level task",
            all_instructions = [...],   # for vocab building
        )
        agent.train_on_demos(demos)
        agent.save()
    """

    def __init__(
        self,
        env_id:           str,
        task_id:          str,
        instruction:      str,
        all_instructions: List[str],
        large_model:      bool = False,
        device:           str  = "auto",
        verbose:          bool = True,
    ):
        self.env_id      = env_id
        self.task_id     = task_id
        self.instruction = instruction
        self.verbose     = verbose

        # ── Device ────────────────────────────────────────────────────────────
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── Build vocab from ALL instructions the system will ever see ────────
        # (important: vocab must be consistent across tasks for weight transfer)
        self.vocab = build_vocab(all_instructions)

        # ── Policy ────────────────────────────────────────────────────────────
        memory_dim = (
            RECURRENT_POLICY["memory_dim_large"]
            if large_model
            else RECURRENT_POLICY["memory_dim"]
        )
        self.policy = RecurrentPolicy(
            vocab_size  = len(self.vocab),
            memory_dim  = memory_dim,
        ).to(self.device)

        # Pre-tokenize this task's instruction (reused every batch)
        self.tokens = tokenize_batch(
            [self.instruction], self.vocab
        ).to(self.device)   # (1, 20) — will be expanded to batch size

        # ── Metrics ───────────────────────────────────────────────────────────
        self.train_losses : List[float] = []
        self.val_losses   : List[float] = []
        self.val_accs     : List[float] = []
        self.best_val_loss = float("inf")
        self.best_state    = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train_on_demos(
        self,
        demos:      Demos,
        epochs:     int   = IL["epochs"],
        lr:         float = IL["lr"],
        batch_size: int   = IL["batch_size"],
        seq_len:    int   = IL["seq_len"],
        val_split:  float = IL["val_split"],
        patience:   int   = IL["patience"],
    ) -> Dict:
        """
        Train the policy via behavioral cloning on bot demonstrations.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  IL Training: {self.task_id}")
            print(f"  Demos: {len(demos)}  |  Device: {self.device}")
            print(f"  epochs={epochs}  lr={lr}  batch_size={batch_size}  seq_len={seq_len}")
            print(f"{'='*60}\n")

        # ── Build sequences ────────────────────────────────────────────────────
        # CHANGED: Now returns 6 values instead of 4
        train_obs, train_act, train_len, val_obs, val_act, val_len = demos_to_sequences(
            demos, seq_len=seq_len, val_split=val_split
        )

        if self.verbose:
            print(f"  Train sequences: {train_obs.shape[0]:,}")
            print(f"  Val sequences:   {val_obs.shape[0]:,}\n")

        # ── DataLoaders ────────────────────────────────────────────────────────
        # CHANGED: Include lengths in datasets
        train_loader = DataLoader(
            TensorDataset(train_obs, train_act, train_len),  # ← Added train_len
            batch_size = batch_size,
            shuffle    = True,
            pin_memory = (self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            TensorDataset(val_obs, val_act, val_len),  # ← Added val_len
            batch_size = batch_size * 2,
            shuffle    = False,
        )

        # ── Optimizer ─────────────────────────────────────────────────────────
        optimizer = optim.Adam(
            self.policy.parameters(),
            lr           = lr,
            weight_decay = IL["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )

        # ── Expand tokens to batch size ────────────────────────────────────────
        def expand_tokens(b: int) -> torch.Tensor:
            return self.tokens.expand(b, -1)

        # ── Training loop ─────────────────────────────────────────────────────
        no_improve = 0
        start_time = time.time()

        if self.verbose:
            print(f"  {'Epoch':>5}  {'TrainLoss':>10}  {'ValLoss':>10}  {'ValAcc':>8}  {'LR':>8}")
            print(f"  {'-'*50}")

        for epoch in range(1, epochs + 1):
            # ── Train ──────────────────────────────────────────────────────────
            self.policy.train()
            epoch_loss = 0.0
            n_batches  = 0

            # CHANGED: Unpack 3 values instead of 2
            for obs_batch, act_batch, len_batch in train_loader:
                obs_batch = obs_batch.to(self.device)
                act_batch = act_batch.to(self.device)
                len_batch = len_batch.to(self.device)  # ← New (for future use)
                B = obs_batch.shape[0]

                # Zero hidden state at the start of each sequence chunk
                hidden = self.policy.init_hidden(B, self.device)

                # Forward
                logits, _, _ = self.policy(
                    obs_batch, expand_tokens(B), hidden
                )

                # Cross-entropy over all (B*T) action predictions
                loss = nn.functional.cross_entropy(
                    logits.view(-1, self.policy.action_dim),
                    act_batch.view(-1),
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), IL["max_grad_norm"]
                )
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            train_loss = epoch_loss / n_batches

            # ── Validate ───────────────────────────────────────────────────────
            val_loss, val_acc = self._validate(val_loader, expand_tokens)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            if self.verbose:
                print(
                    f"  {epoch:>5}  {train_loss:>10.4f}  "
                    f"{val_loss:>10.4f}  {val_acc*100:>7.1f}%  {current_lr:>8.2e}"
                )

            # ── Early stopping ─────────────────────────────────────────────────
            if val_loss < self.best_val_loss - 1e-4:
                self.best_val_loss = val_loss
                self.best_state    = {
                    k: v.cpu().clone() for k, v in self.policy.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    if self.verbose:
                        print(f"\n  Early stopping at epoch {epoch}")
                    break

        # Restore best weights
        if self.best_state is not None:
            self.policy.load_state_dict(self.best_state)

        elapsed = time.time() - start_time

        if self.verbose:
            print(f"\n  IL training complete in {elapsed:.1f}s")
            print(f"  Best val loss: {self.best_val_loss:.4f}")
            print(f"  Final val acc: {self.val_accs[-1]*100:.1f}%\n")

        return {
            "train_losses": self.train_losses,
            "val_losses"  : self.val_losses,
            "val_accs"    : self.val_accs,
            "best_val_loss": self.best_val_loss,
        }


    # ─── CHANGE 2: _validate method ──────────────────────────────────────────────

    def _validate(
        self,
        val_loader: DataLoader,
        expand_tokens,
    ) -> Tuple[float, float]:
        """Returns (val_loss, val_accuracy)."""
        self.policy.eval()
        total_loss  = 0.0
        total_correct = 0
        total_tokens  = 0

        with torch.no_grad():
            # CHANGED: Unpack 3 values instead of 2
            for obs_batch, act_batch, len_batch in val_loader:
                obs_batch = obs_batch.to(self.device)
                act_batch = act_batch.to(self.device)
                len_batch = len_batch.to(self.device)  # ← New (for future use)
                B = obs_batch.shape[0]

                hidden = self.policy.init_hidden(B, self.device)
                logits, _, _ = self.policy(obs_batch, expand_tokens(B), hidden)

                loss = nn.functional.cross_entropy(
                    logits.view(-1, self.policy.action_dim),
                    act_batch.view(-1),
                )
                total_loss += loss.item()

                preds = logits.view(-1, self.policy.action_dim).argmax(dim=-1)
                total_correct += (preds == act_batch.view(-1)).sum().item()
                total_tokens  += act_batch.numel()

        n = len(val_loader)
        return total_loss / n, total_correct / total_tokens

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, n_episodes: int = 20) -> Dict:
        """Run the policy in the env deterministically, return success rate."""
        import gymnasium as gym
        from envs.task_suite import FlatObsWrapper, flatten_obs

        env = gym.make(self.env_id, render_mode=None)
        env = FlatObsWrapper(env)
        self.policy.eval()

        rewards   = []
        lengths   = []
        successes = []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            done   = False
            ep_r   = 0.0
            ep_l   = 0
            hidden = self.policy.init_hidden(1, self.device)
            tokens = self.tokens   # (1, 20)

            while not done and ep_l < 256:
                action, hidden = self.policy.act(
                    obs, tokens, hidden, deterministic=True
                )
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_r += reward
                ep_l += 1

            rewards.append(ep_r)
            lengths.append(ep_l)
            successes.append(float(ep_r > 0))

        env.close()
        return {
            "mean_reward" : float(np.mean(rewards)),
            "mean_length" : float(np.mean(lengths)),
            "success_rate": float(np.mean(successes)),
        }

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        if path is None:
            path = os.path.join(CHECKPOINTS, f"il_policy_{self.task_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "vocab"            : self.vocab,
            "task_id"          : self.task_id,
            "env_id"           : self.env_id,
            "instruction"      : self.instruction,
            "memory_dim"       : self.policy.memory_dim,
            "train_losses"     : self.train_losses,
            "val_losses"       : self.val_losses,
            "val_accs"         : self.val_accs,
        }, path)
        if self.verbose:
            print(f"  Saved IL checkpoint → {path}")
        return path

    @classmethod
    def load(
        cls,
        task_id:          str,
        all_instructions: List[str],
        path:             Optional[str] = None,
        device:           str = "auto",
        verbose:          bool = False,
    ) -> "ILAgent":
        if path is None:
            path = os.path.join(CHECKPOINTS, f"il_policy_{task_id}.pt")

        ckpt = torch.load(path, map_location="cpu")
        agent = cls(
            env_id           = ckpt["env_id"],
            task_id          = ckpt["task_id"],
            instruction      = ckpt["instruction"],
            all_instructions = all_instructions,
            device           = device,
            verbose          = verbose,
        )
        agent.policy.load_state_dict(ckpt["policy_state_dict"])
        agent.train_losses = ckpt.get("train_losses", [])
        agent.val_losses   = ckpt.get("val_losses", [])
        agent.val_accs     = ckpt.get("val_accs", [])
        if verbose:
            print(f"  Loaded IL checkpoint ← {path}")
        return agent


# ─── Batch IL training for all tasks in a curriculum stage ───────────────────

def train_il_stage(
    stage_idx:   int,
    n_demos:     Optional[int] = None,
    device:      str   = "auto",
    large_model: bool  = False,
    skip_existing: bool = True,
) -> Dict[str, ILAgent]:
    """
    Train IL policies for all tasks in a given curriculum stage.

    Args:
        stage_idx:    which stage from CURRICULUM_STAGES to train
        n_demos:      demos per task override (None => per-stage from config)
        device:       cpu or cuda
        large_model:  use 2048-dim GRU (needed for BossLevel)
        skip_existing: skip if IL checkpoint already exists

    Returns:
        dict of {task_id: ILAgent}
    """
    from utils.il_config import CURRICULUM_STAGES, ALL_IL_TASKS
    from trajectory.bot_collector import collect_bot_demos, save_demos

    def make_task_id(env_id, instruction):
        env_short  = env_id.replace("BabyAI-", "").replace("-v0", "")
        instr_slug = instruction.replace(" ", "_")
        return f"{env_short}__{instr_slug}"

    stage_tasks      = CURRICULUM_STAGES[stage_idx]
    all_instructions = [instr for _, instr in ALL_IL_TASKS]
    agents           = {}
    demos_per_task   = n_demos if n_demos is not None else get_stage_demo_count(stage_idx)

    print(f"\n{'='*60}")
    print(f"  IL Stage {stage_idx}  |  {len(stage_tasks)} tasks  |  device={device}")
    print(f"{'='*60}")

    for env_id, instruction in stage_tasks:
        task_id  = make_task_id(env_id, instruction)
        ckpt_path = os.path.join(CHECKPOINTS, f"il_policy_{task_id}.pt")

        if skip_existing and os.path.exists(ckpt_path):
            print(f"\n  [{task_id}]  checkpoint exists, loading...")
            agent = ILAgent.load(task_id, all_instructions, device=device)
        else:
            # Collect demos if not cached
            if demos_exist(task_id):
                demos, _missions = load_demos(task_id)
                if len(demos) < demos_per_task:
                    demos, missions = collect_bot_demos(
                        env_id=env_id,
                        n_episodes=demos_per_task,
                        extract_missions=True,
                    )
                    save_demos(demos, task_id, missions)
            else:
                demos, missions = collect_bot_demos(
                    env_id=env_id,
                    n_episodes=demos_per_task,
                    extract_missions=True,
                )
                save_demos(demos, task_id, missions)

            agent = ILAgent(
                env_id           = env_id,
                task_id          = task_id,
                instruction      = instruction,
                all_instructions = all_instructions,
                large_model      = large_model,
                device           = device,
                verbose          = True,
            )
            agent.train_on_demos(demos)
            agent.save()

        # Evaluate
        metrics = agent.evaluate(n_episodes=20)
        print(f"  Eval → success: {metrics['success_rate']*100:.1f}%  "
              f"reward: {metrics['mean_reward']:.3f}")

        agents[task_id] = agent

    return agents
