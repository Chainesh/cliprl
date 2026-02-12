"""
Trajectory Collector for CLIP-RL.

After a policy is trained, we run it in the environment for N episodes
and record (obs, action) at every timestep. This sequence is the
"behavioral fingerprint" of the policy — what it actually does, not
what its weights look like.

Output per policy:
    List[List[Tuple[np.ndarray, int]]]
    = list of episodes, each episode = list of (obs_vec, action) pairs

Saved to: checkpoints/trajectories_{task_id}.pkl
"""

import numpy as np
import torch
import pickle
import os
import sys
from typing import List, Tuple, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.task_suite import make_env, action_to_onehot
from agents.policy_network import PolicyNetwork
from utils.config import TRAJECTORY, ACTION_DIM, OBS_DIM, CHECKPOINTS


# Type alias for clarity
Episode    = List[Tuple[np.ndarray, int]]        # [(obs_vec, action), ...]
Trajectory = List[Episode]                        # list of episodes


def collect_trajectories(
    policy:      PolicyNetwork,
    env_id:      str,
    n_episodes:  int = TRAJECTORY["n_episodes"],
    max_steps:   int = TRAJECTORY["max_steps"],
    seed:        int = 0,
    deterministic: bool = True,
    device:      str = "cpu",
) -> Trajectory:
    """
    Run the policy in the environment for n_episodes episodes.
    Records (obs_vec, action) at every step.

    Args:
        policy:       trained PolicyNetwork
        env_id:       BabyAI gymnasium environment id
        n_episodes:   number of episodes to collect
        max_steps:    max steps per episode (hard cutoff)
        seed:         base random seed
        deterministic: if True, use argmax policy (cleaner behavioral signal)
        device:       torch device for policy inference

    Returns:
        List of episodes. Each episode is a list of (obs_vec, action) tuples.
        obs_vec shape: (OBS_DIM,) = (148,)
        action: int in [0, 6]
    """
    env = make_env(env_id, seed=seed)
    policy = policy.to(device)
    policy.eval()

    trajectories: Trajectory = []

    for ep_idx in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep_idx)
        episode: Episode = []
        done = False
        steps = 0

        while not done and steps < max_steps:
            # Select action
            action = policy.act(obs, deterministic=deterministic)

            # Record BEFORE stepping (we want the obs that caused this action)
            episode.append((obs.copy(), action))

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        trajectories.append(episode)

    env.close()
    return trajectories


def trajectories_to_tensor(
    trajectories: Trajectory,
    max_ep_len:   int = TRAJECTORY["max_steps"],
    device:       str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts collected trajectories into padded tensors for GRU input.

    Each timestep input = concat(obs_vec, action_onehot) = (155,)

    Args:
        trajectories: list of N episodes
        max_ep_len:   pad/truncate all episodes to this length
        device:       torch device

    Returns:
        inputs:  (N, max_ep_len, 155)  float32 — padded sequences
        lengths: (N,)                  int     — actual length of each episode
    """
    N          = len(trajectories)
    input_dim  = OBS_DIM + ACTION_DIM  # 148 + 7 = 155

    inputs  = np.zeros((N, max_ep_len, input_dim), dtype=np.float32)
    lengths = np.zeros(N, dtype=np.int64)

    for i, episode in enumerate(trajectories):
        ep_len = min(len(episode), max_ep_len)
        lengths[i] = ep_len
        for t in range(ep_len):
            obs_vec, action = episode[t]
            action_onehot   = action_to_onehot(action)
            inputs[i, t, :OBS_DIM]  = obs_vec
            inputs[i, t, OBS_DIM:]  = action_onehot

    return (
        torch.FloatTensor(inputs).to(device),
        torch.LongTensor(lengths).to(device),
    )


def save_trajectories(trajectories: Trajectory, task_id: str, path: Optional[str] = None):
    """Save collected trajectories to disk."""
    if path is None:
        path = os.path.join(CHECKPOINTS, f"trajectories_{task_id}.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"task_id": task_id, "trajectories": trajectories}, f)
    print(f"  Saved {len(trajectories)} trajectories → {path}")


def load_trajectories(task_id: str, path: Optional[str] = None) -> Trajectory:
    """Load previously collected trajectories from disk."""
    if path is None:
        path = os.path.join(CHECKPOINTS, f"trajectories_{task_id}.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["trajectories"]


def collect_all_base_trajectories(
    n_episodes: int = TRAJECTORY["n_episodes"],
    device:     str = "cpu",
) -> Dict[str, Trajectory]:
    """
    Collect trajectories for all base tasks.
    Loads trained policy checkpoints, runs them, saves trajectories.

    Returns:
        dict of {task_id: trajectories}
    """
    from envs.task_suite import TaskRegistry
    from agents.ppo_agent import PPOAgent

    registry = TaskRegistry()
    all_trajectories: Dict[str, Trajectory] = {}

    print(f"\nCollecting trajectories for {len(registry.base_tasks())} base tasks...")
    print(f"  Episodes per task: {n_episodes}\n")

    for env_id, instruction, task_id in registry.base_tasks():
        traj_path = os.path.join(CHECKPOINTS, f"trajectories_{task_id}.pkl")

        # Skip if already collected
        if os.path.exists(traj_path):
            print(f"  [{task_id}] Loading existing trajectories...")
            trajectories = load_trajectories(task_id)
        else:
            # Load trained policy
            ckpt_path = os.path.join(CHECKPOINTS, f"policy_{task_id}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  [{task_id}] ✗ No checkpoint found — run train phase first!")
                continue

            print(f"  [{task_id}] Collecting {n_episodes} episodes...")
            agent = PPOAgent.load(task_id=task_id, env_id=env_id, device=device)

            trajectories = collect_trajectories(
                policy      = agent.policy,
                env_id      = env_id,
                n_episodes  = n_episodes,
                device      = device,
            )
            save_trajectories(trajectories, task_id)

        all_trajectories[task_id] = trajectories

        # Print some stats
        ep_lengths = [len(ep) for ep in trajectories]
        print(f"    → {len(trajectories)} episodes | "
              f"avg length: {np.mean(ep_lengths):.1f} | "
              f"min: {min(ep_lengths)} | max: {max(ep_lengths)}")

    print(f"\nTrajectory collection complete.\n")
    return all_trajectories


# ─── Main (smoke test) ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from agents.policy_network import PolicyNetwork
    from envs.task_suite import make_env

    print("Smoke test — collecting 5 episodes with a random policy...\n")

    policy = PolicyNetwork()   # untrained / random — just testing the pipeline

    trajs = collect_trajectories(
        policy     = policy,
        env_id     = "BabyAI-GoToRedBall-v0",
        n_episodes = 5,
        deterministic = False,   # random policy needs stochastic sampling
    )

    print(f"Collected {len(trajs)} episodes")
    print(f"Episode lengths: {[len(ep) for ep in trajs]}")
    print(f"Single step — obs shape: {trajs[0][0][0].shape}, action: {trajs[0][0][1]}")

    # Test tensor conversion
    inputs, lengths = trajectories_to_tensor(trajs)
    print(f"\nTensor shapes:")
    print(f"  inputs:  {inputs.shape}   (should be [5, 200, 155])")
    print(f"  lengths: {lengths}")

    print("\nAll checks passed!")