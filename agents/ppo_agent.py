"""
PPO Agent for CLIP-RL.

We implement PPO from scratch (not using stable-baselines3) so that:
  1. We have full control over the policy network architecture
  2. We can easily extract and load policy weights for transfer
  3. No external dependency on SB3's internal network format

Algorithm: Proximal Policy Optimization (Schulman et al., 2017)

Training loop:
  repeat:
    1. Collect n_steps using current policy (rollout phase)
    2. Compute advantages using GAE
    3. Update policy for n_epochs using PPO-clip objective
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Optional, Tuple, List, Dict
import os, sys, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.policy_network import PolicyNetwork
from utils.config import PPO, OBS_DIM, ACTION_DIM, CHECKPOINTS


# ─── Rollout Buffer ───────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores transitions collected during a single rollout phase.
    Computes GAE advantages after rollout is complete.

    Stores:
        observations: (n_steps, obs_dim)
        actions:      (n_steps,)
        rewards:      (n_steps,)
        values:       (n_steps,)
        log_probs:    (n_steps,)
        dones:        (n_steps,)
        advantages:   (n_steps,)  — computed after rollout
        returns:      (n_steps,)  — advantages + values
    """

    def __init__(self, n_steps: int, obs_dim: int, gamma: float, gae_lambda: float):
        self.n_steps    = n_steps
        self.obs_dim    = obs_dim
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        self.observations = np.zeros((self.n_steps, self.obs_dim), dtype=np.float32)
        self.actions      = np.zeros(self.n_steps, dtype=np.int64)
        self.rewards      = np.zeros(self.n_steps, dtype=np.float32)
        self.values       = np.zeros(self.n_steps, dtype=np.float32)
        self.log_probs    = np.zeros(self.n_steps, dtype=np.float32)
        self.dones        = np.zeros(self.n_steps, dtype=np.float32)
        self.advantages   = np.zeros(self.n_steps, dtype=np.float32)
        self.returns      = np.zeros(self.n_steps, dtype=np.float32)
        self.pos          = 0
        self.full         = False

    def add(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        value:    float,
        log_prob: float,
        done:     bool,
    ):
        self.observations[self.pos] = obs
        self.actions[self.pos]      = action
        self.rewards[self.pos]      = reward
        self.values[self.pos]       = value
        self.log_probs[self.pos]    = log_prob
        self.dones[self.pos]        = float(done)
        self.pos += 1
        if self.pos == self.n_steps:
            self.full = True

    def compute_advantages(self, last_value: float, last_done: bool):
        """
        Generalized Advantage Estimation (GAE).

        δₜ = rₜ + γ V(sₜ₊₁)(1-done) - V(sₜ)
        Aₜ = δₜ + (γλ)δₜ₊₁ + (γλ)² δₜ₊₂ + ...

        Computed backwards for efficiency.
        """
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value        = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value        = self.values[t + 1]

            delta    = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """
        Yields random mini-batches for PPO update.
        Each mini-batch is a dict of tensors.
        """
        indices = np.random.permutation(self.n_steps)
        for start in range(0, self.n_steps, batch_size):
            batch_idx = indices[start: start + batch_size]
            yield {
                "obs":        torch.FloatTensor(self.observations[batch_idx]),
                "actions":    torch.LongTensor(self.actions[batch_idx]),
                "old_log_probs": torch.FloatTensor(self.log_probs[batch_idx]),
                "advantages": torch.FloatTensor(self.advantages[batch_idx]),
                "returns":    torch.FloatTensor(self.returns[batch_idx]),
            }


# ─── PPO Agent ────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    PPO Agent. Wraps PolicyNetwork with a full training loop.

    Usage:
        agent = PPOAgent(env_id="BabyAI-GoToRedBall-v0", task_id="GoToRedBall")
        agent.train(total_timesteps=500_000)
        agent.save()
        # later:
        agent = PPOAgent.load(task_id="GoToRedBall")
    """

    def __init__(
        self,
        env_id:    str,
        task_id:   str,
        seed:      int = 42,
        device:    str = "auto",
        verbose:   bool = True,
    ):
        self.env_id  = env_id
        self.task_id = task_id
        self.seed    = seed
        self.verbose = verbose

        # ── Device ────────────────────────────────────────────────────────────
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── Policy network ────────────────────────────────────────────────────
        self.policy = PolicyNetwork().to(self.device)

        # ── Optimizer ─────────────────────────────────────────────────────────
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr  = PPO["learning_rate"],
            eps = 1e-5,
        )

        # ── Hyperparameters ───────────────────────────────────────────────────
        self.n_steps      = PPO["n_steps"]
        self.batch_size   = PPO["batch_size"]
        self.n_epochs     = PPO["n_epochs"]
        self.gamma        = PPO["gamma"]
        self.gae_lambda   = PPO["gae_lambda"]
        self.clip_range   = PPO["clip_range"]
        self.ent_coef     = PPO["ent_coef"]
        self.vf_coef      = PPO["vf_coef"]
        self.max_grad_norm = PPO["max_grad_norm"]

        # ── Rollout buffer ────────────────────────────────────────────────────
        self.buffer = RolloutBuffer(
            n_steps    = self.n_steps,
            obs_dim    = OBS_DIM,
            gamma      = self.gamma,
            gae_lambda = self.gae_lambda,
        )

        # ── Metrics ───────────────────────────────────────────────────────────
        self.ep_rewards:  List[float] = []
        self.ep_lengths:  List[int]   = []
        self.train_losses: List[float] = []
        self.timesteps_done: int = 0

    # ── Environment setup ─────────────────────────────────────────────────────

    def _make_env(self):
        """Import here to avoid circular issues."""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from envs.task_suite import make_env
        return make_env(self.env_id, seed=self.seed)

    # ── Core PPO methods ──────────────────────────────────────────────────────

    def _collect_rollout(self, env) -> Tuple[float, float]:
        """
        Collect n_steps of experience using current policy.
        Returns (mean_episode_reward, mean_episode_length) for logging.
        """
        self.buffer.reset()
        self.policy.eval()

        obs, _          = env.reset()
        ep_reward       = 0.0
        ep_length       = 0
        completed_eps   = []
        ep_lengths_list = []

        for step in range(self.n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits, value = self.policy(obs_t)
                dist      = Categorical(logits=logits)
                action    = dist.sample()
                log_prob  = dist.log_prob(action)

            action_int = action.item()
            next_obs, reward, terminated, truncated, _ = env.step(action_int)
            done = terminated or truncated

            self.buffer.add(
                obs      = obs,
                action   = action_int,
                reward   = float(reward),
                value    = value.item(),
                log_prob = log_prob.item(),
                done     = done,
            )

            ep_reward += float(reward)
            ep_length += 1
            self.timesteps_done += 1

            if done:
                completed_eps.append(ep_reward)
                ep_lengths_list.append(ep_length)
                obs, _    = env.reset()
                ep_reward = 0.0
                ep_length = 0
            else:
                obs = next_obs

        # Bootstrap: get value of last state for GAE
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, last_value = self.policy(obs_t)
        self.buffer.compute_advantages(
            last_value = last_value.item(),
            last_done  = done,
        )

        mean_reward = float(np.mean(completed_eps)) if completed_eps else 0.0
        mean_length = float(np.mean(ep_lengths_list)) if ep_lengths_list else 0.0
        return mean_reward, mean_length

    def _ppo_update(self) -> Dict[str, float]:
        """
        Run n_epochs of PPO updates using the current rollout buffer.

        Returns dict of mean losses for logging.
        """
        self.policy.train()

        # Normalize advantages (standard PPO practice)
        adv = self.buffer.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.buffer.advantages = adv

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        n_updates         = 0

        for _ in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                obs         = batch["obs"].to(self.device)
                actions     = batch["actions"].to(self.device)
                old_log_probs = batch["old_log_probs"].to(self.device)
                advantages  = batch["advantages"].to(self.device)
                returns     = batch["returns"].to(self.device)

                # Current policy evaluation
                log_probs, entropy, values = self.policy.evaluate_actions(obs, actions)

                # ── Policy loss (clipped surrogate) ───────────────────────────
                ratio        = torch.exp(log_probs - old_log_probs)
                policy_loss1 = ratio * advantages
                policy_loss2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                policy_loss  = -torch.min(policy_loss1, policy_loss2).mean()

                # ── Value loss (clipped) ──────────────────────────────────────
                value_loss = F.mse_loss(values.squeeze(-1), returns)

                # ── Total loss ────────────────────────────────────────────────
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.item()
                n_updates         += 1

        import torch.nn.functional as F
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss":  total_value_loss  / n_updates,
            "entropy":     total_entropy     / n_updates,
        }

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, total_timesteps: Optional[int] = None) -> None:
        """
        Full PPO training loop.

        Args:
            total_timesteps: override default from config if provided
        """
        if total_timesteps is None:
            total_timesteps = PPO["total_timesteps"]

        env        = self._make_env()
        start_time = time.time()
        n_updates  = total_timesteps // self.n_steps

        if self.verbose:
            print(f"\n{'='*55}")
            print(f"  Training: {self.task_id}")
            print(f"  Env:      {self.env_id}")
            print(f"  Steps:    {total_timesteps:,}  |  Updates: {n_updates}")
            print(f"  Device:   {self.device}")
            print(f"{'='*55}\n")

        for update in range(1, n_updates + 1):
            # 1. Collect rollout
            mean_reward, mean_length = self._collect_rollout(env)

            # 2. PPO update
            losses = self._ppo_update()

            # 3. Log
            self.ep_rewards.append(mean_reward)
            self.ep_lengths.append(mean_length)
            self.train_losses.append(losses["policy_loss"])

            if self.verbose and update % 10 == 0:
                elapsed = time.time() - start_time
                fps = self.timesteps_done / elapsed
                print(
                    f"  Update {update:4d}/{n_updates} | "
                    f"steps: {self.timesteps_done:7,} | "
                    f"reward: {mean_reward:6.3f} | "
                    f"ep_len: {mean_length:5.1f} | "
                    f"π_loss: {losses['policy_loss']:7.4f} | "
                    f"fps: {fps:5.0f}"
                )

        env.close()
        elapsed = time.time() - start_time
        if self.verbose:
            print(f"\n  Training complete in {elapsed:.1f}s")
            print(f"  Final mean reward (last 10): {np.mean(self.ep_rewards[-10:]):.3f}\n")

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """Save policy weights and training metadata."""
        if path is None:
            path = os.path.join(CHECKPOINTS, f"policy_{self.task_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state":   self.optimizer.state_dict(),
            "ep_rewards":        self.ep_rewards,
            "ep_lengths":        self.ep_lengths,
            "timesteps_done":    self.timesteps_done,
            "task_id":           self.task_id,
            "env_id":            self.env_id,
        }
        torch.save(checkpoint, path)
        if self.verbose:
            print(f"  Saved checkpoint: {path}")
        return path

    @classmethod
    def load(
        cls,
        task_id:   str,
        env_id:    str,
        path:      Optional[str] = None,
        device:    str = "auto",
        verbose:   bool = False,
    ) -> "PPOAgent":
        """Load a trained agent from checkpoint."""
        if path is None:
            path = os.path.join(CHECKPOINTS, f"policy_{task_id}.pt")

        agent = cls(env_id=env_id, task_id=task_id, device=device, verbose=verbose)
        checkpoint = torch.load(path, map_location=agent.device)
        agent.policy.load_state_dict(checkpoint["policy_state_dict"])
        agent.ep_rewards      = checkpoint.get("ep_rewards", [])
        agent.ep_lengths      = checkpoint.get("ep_lengths", [])
        agent.timesteps_done  = checkpoint.get("timesteps_done", 0)

        if verbose:
            print(f"  Loaded checkpoint: {path}")
        return agent

    def evaluate(self, n_episodes: int = 20) -> Dict[str, float]:
        """
        Evaluate current policy deterministically.

        Returns:
            dict with mean_reward, std_reward, mean_length, success_rate
        """
        env = self._make_env()
        self.policy.eval()

        rewards = []
        lengths = []
        successes = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_length = 0
            done = False

            while not done and ep_length < 200:
                action = self.policy.act(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_length += 1

            rewards.append(ep_reward)
            lengths.append(ep_length)
            successes.append(float(ep_reward > 0))  # BabyAI: reward>0 = success

        env.close()
        return {
            "mean_reward":  float(np.mean(rewards)),
            "std_reward":   float(np.std(rewards)),
            "mean_length":  float(np.mean(lengths)),
            "success_rate": float(np.mean(successes)),
        }


# ─── Batch Training (all base tasks) ─────────────────────────────────────────

def train_all_base_tasks(
    total_timesteps: int = PPO["total_timesteps"],
    seed: int = 42,
) -> Dict[str, PPOAgent]:
    """
    Train a PPO agent for every base task and save checkpoints.
    Returns dict of {task_id: agent}.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from envs.task_suite import TaskRegistry

    registry = TaskRegistry()
    agents   = {}

    print(f"\nTraining {len(registry.base_tasks())} base tasks...\n")

    for i, (env_id, instruction, task_id) in enumerate(registry.base_tasks()):
        print(f"[{i+1}/{len(registry.base_tasks())}] Task: {task_id}")

        # Skip if already trained
        ckpt_path = os.path.join(CHECKPOINTS, f"policy_{task_id}.pt")
        if os.path.exists(ckpt_path):
            print(f"  → Checkpoint exists, loading...")
            agent = PPOAgent.load(task_id=task_id, env_id=env_id, verbose=True)
        else:
            agent = PPOAgent(
                env_id  = env_id,
                task_id = task_id,
                seed    = seed + i,
                verbose = True,
            )
            agent.train(total_timesteps=total_timesteps)
            agent.save()

        # Evaluate and report
        metrics = agent.evaluate(n_episodes=20)
        print(f"  Eval → reward: {metrics['mean_reward']:.3f} | "
              f"success: {metrics['success_rate']*100:.1f}%\n")

        agents[task_id] = agent

    return agents


# ─── Main (smoke test — single short training run) ────────────────────────────

if __name__ == "__main__":
    import torch.nn.functional as F

    print("Quick smoke test — 5000 steps on GoToRedBall...")

    agent = PPOAgent(
        env_id  = "BabyAI-GoToRedBall-v0",
        task_id = "test_run",
        verbose = True,
    )
    agent.train(total_timesteps=5_000)

    metrics = agent.evaluate(n_episodes=10)
    print(f"\nEval metrics: {metrics}")