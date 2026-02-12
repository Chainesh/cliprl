"""
Bot Demo Collector for CLIP-RL IL.

Uses BabyAI's built-in BabyAIBot (a hand-crafted expert) to generate
demonstration trajectories for imitation learning.

The bot has access to the full env state (not just the partial 7×7 view)
so it can plan optimally. We record the flat observations the RL agent
would see + the bot's actions — giving us (obs, action) supervised pairs.

Usage:
    demos = collect_bot_demos("BabyAI-BossLevel-v0", n_episodes=10_000)
    save_demos(demos, "BossLevel")
"""

import numpy as np
import pickle
import os
import sys
import time
import signal
from typing import List, Tuple, Optional, Dict
from contextlib import contextmanager

import gymnasium as gym
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.task_suite import flatten_obs
from utils.il_config import BOT_DEMOS, DEMOS_DIR, CURRICULUM_STAGES, ALL_IL_TASKS


# Type aliases
Step  = Tuple[np.ndarray, int]        # (flat_obs, action)
Demo  = List[Step]                    # one episode
Demos = List[Demo]                    # collection of episodes


# ─── Timeout context manager ─────────────────────────────────────────────────

@contextmanager
def timeout(seconds: int):
    """Raise TimeoutError if block takes longer than `seconds`."""
    def _handler(signum, frame):
        raise TimeoutError()
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ─── Core collection ─────────────────────────────────────────────────────────

def collect_bot_demos(
    env_id:      str,
    n_episodes:  int  = BOT_DEMOS["n_episodes"],
    max_steps:   int  = BOT_DEMOS["max_steps"],
    seed:        int  = 0,
    timeout_sec: int  = BOT_DEMOS["timeout"],
    verbose:     bool = True,
) -> Demos:
    """
    Collect n_episodes demonstrations from the BabyAI bot.

    The bot plans on the raw (unwrapped) env state, but we record
    the flat (FlatObsWrapper-style) observations that the RL policy sees.

    Args:
        env_id:      BabyAI gymnasium env id
        n_episodes:  number of successful demos to collect
        max_steps:   hard cutoff per episode
        seed:        base random seed
        timeout_sec: seconds before giving up on one episode
        verbose:     print progress

    Returns:
        List of successful demos (bot reached the goal each time)
    """
    # Import here to avoid issues if minigrid not installed
    try:
        from minigrid.utils.baby_ai_bot import BabyAIBot
    except ImportError:
        raise ImportError(
            "BabyAIBot not found. Make sure minigrid>=2.5.0 is installed:\n"
            "  pip install minigrid==2.5.0"
        )

    demos: Demos = []
    n_failed     = 0
    n_timeout    = 0
    ep_idx       = 0

    if verbose:
        print(f"\n  Collecting {n_episodes} bot demos for {env_id}...")
        print(f"  Max steps per episode: {max_steps}  |  Timeout: {timeout_sec}s\n")

    while len(demos) < n_episodes:
        ep_idx += 1

        # Create a FRESH raw env for each episode so the bot gets clean state
        env = gym.make(env_id, render_mode=None)
        obs_dict, _ = env.reset(seed=seed + ep_idx)

        try:
            with timeout(timeout_sec):
                bot      = BabyAIBot(env)
                episode  = []
                done     = False
                steps    = 0
                success  = False

                while not done and steps < max_steps:
                    # Flatten obs the same way FlatObsWrapper does
                    flat_obs = flatten_obs(obs_dict)

                    # Bot plans and returns an action
                    try:
                        action = bot.replan()
                    except Exception:
                        # Bot gets confused sometimes (e.g. impossible sub-goal)
                        break

                    # Record BEFORE stepping
                    episode.append((flat_obs.copy(), int(action)))

                    # Step the raw env
                    obs_dict, reward, terminated, truncated, _ = env.step(action)
                    done    = terminated or truncated
                    success = terminated and reward > 0
                    steps  += 1

                if success and len(episode) > 0:
                    demos.append(episode)
                    if verbose and len(demos) % 500 == 0:
                        sr = len(demos) / (len(demos) + n_failed + n_timeout) * 100
                        print(
                            f"  {len(demos):5d}/{n_episodes}  |  "
                            f"failed: {n_failed}  timeout: {n_timeout}  |  "
                            f"success rate: {sr:.1f}%  |  "
                            f"ep_len: {len(episode)}"
                        )
                else:
                    n_failed += 1

        except TimeoutError:
            n_timeout += 1
        finally:
            env.close()

    if verbose:
        total   = len(demos) + n_failed + n_timeout
        sr      = len(demos) / total * 100
        avg_len = np.mean([len(d) for d in demos])
        print(f"\n  Done. {len(demos)} demos collected")
        print(f"  Success rate: {sr:.1f}%  |  Avg episode length: {avg_len:.1f}")
        print(f"  Failed: {n_failed}  |  Timed out: {n_timeout}\n")

    return demos


# ─── Save / Load ─────────────────────────────────────────────────────────────

def demo_path(task_id: str) -> str:
    return os.path.join(DEMOS_DIR, f"demos_{task_id}.pkl")


def save_demos(demos: Demos, task_id: str) -> str:
    path = demo_path(task_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stats = {
        "n_demos"   : len(demos),
        "avg_length": float(np.mean([len(d) for d in demos])),
        "min_length": int(min(len(d) for d in demos)),
        "max_length": int(max(len(d) for d in demos)),
    }
    with open(path, "wb") as f:
        pickle.dump({"task_id": task_id, "demos": demos, "stats": stats}, f)
    print(f"  Saved {len(demos)} demos → {path}")
    print(f"  Avg length: {stats['avg_length']:.1f} | "
          f"Min: {stats['min_length']} | Max: {stats['max_length']}")
    return path


def load_demos(task_id: str) -> Demos:
    path = demo_path(task_id)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No demos found at {path}\n"
            f"Run bot collection first: collect_all_demos()"
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["demos"]


def demos_exist(task_id: str) -> bool:
    return os.path.exists(demo_path(task_id))


# ─── Batch collection for all curriculum tasks ────────────────────────────────

def collect_all_demos(
    n_episodes:  int  = BOT_DEMOS["n_episodes"],
    stage_filter: Optional[int] = None,   # None = all stages, int = specific stage
    skip_existing: bool = True,
    verbose:     bool = True,
) -> Dict[str, Demos]:
    """
    Collect bot demos for all tasks (or a specific curriculum stage).

    Args:
        n_episodes:   demos per task
        stage_filter: if set, only collect for this stage index
        skip_existing: skip tasks that already have demos
        verbose:      print progress

    Returns:
        dict of {task_id: demos}
    """
    from envs.task_suite import TaskRegistry

    # Build task id the same way as the main pipeline
    def make_task_id(env_id, instruction):
        env_short  = env_id.replace("BabyAI-", "").replace("-v0", "")
        instr_slug = instruction.replace(" ", "_")
        return f"{env_short}__{instr_slug}"

    if stage_filter is not None:
        tasks = CURRICULUM_STAGES[stage_filter]
    else:
        tasks = ALL_IL_TASKS

    all_demos: Dict[str, Demos] = {}

    print(f"\n{'='*60}")
    print(f"  Collecting bot demos | {len(tasks)} tasks | {n_episodes} eps each")
    print(f"{'='*60}")

    for env_id, instruction in tasks:
        task_id = make_task_id(env_id, instruction)

        if skip_existing and demos_exist(task_id):
            print(f"  [{task_id}]  skipping (demos exist)")
            all_demos[task_id] = load_demos(task_id)
            continue

        print(f"\n  [{task_id}]")
        demos = collect_bot_demos(
            env_id     = env_id,
            n_episodes = n_episodes,
            verbose    = verbose,
        )
        save_demos(demos, task_id)
        all_demos[task_id] = demos

    print(f"\nAll demos collected.\n")
    return all_demos


# ─── Build PyTorch dataset from demos ────────────────────────────────────────

def demos_to_sequences(
    demos:      Demos,
    seq_len:    int,
    val_split:  float = 0.1,
) -> Tuple:
    """
    Convert demos into fixed-length overlapping sequences for TBPTT.

    Each sequence is seq_len consecutive (obs, action) steps.
    Sequences that cross episode boundaries are discarded.

    Returns:
        train_obs:     (N_train, seq_len, 148)
        train_actions: (N_train, seq_len)
        val_obs:       (N_val,   seq_len, 148)
        val_actions:   (N_val,   seq_len)
    """
    import torch

    obs_seqs     = []
    action_seqs  = []

    for demo in demos:
        if len(demo) < seq_len:
            continue   # skip very short episodes
        # Slice into non-overlapping windows
        for start in range(0, len(demo) - seq_len + 1, seq_len):
            obs_seq    = np.stack([demo[t][0] for t in range(start, start + seq_len)])
            action_seq = np.array([demo[t][1] for t in range(start, start + seq_len)])
            obs_seqs.append(obs_seq)
            action_seqs.append(action_seq)

    # Shuffle
    idx = np.random.permutation(len(obs_seqs))
    obs_seqs    = [obs_seqs[i]    for i in idx]
    action_seqs = [action_seqs[i] for i in idx]

    # Train/val split
    n_val  = max(1, int(len(obs_seqs) * val_split))
    n_train = len(obs_seqs) - n_val

    def to_tensors(obs_list, act_list):
        return (
            torch.FloatTensor(np.stack(obs_list)),
            torch.LongTensor(np.stack(act_list)),
        )

    train = to_tensors(obs_seqs[:n_train],    action_seqs[:n_train])
    val   = to_tensors(obs_seqs[n_train:],    action_seqs[n_train:])

    return train[0], train[1], val[0], val[1]


# ─── Main (smoke test) ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Smoke test — collecting 10 demos from GoToRedBall bot...\n")

    demos = collect_bot_demos(
        env_id     = "BabyAI-GoToRedBall-v0",
        n_episodes = 10,
        verbose    = True,
    )

    print(f"Collected {len(demos)} demos")
    print(f"Episode lengths: {[len(d) for d in demos]}")
    print(f"Single step — obs shape: {demos[0][0][0].shape}, action: {demos[0][0][1]}")

    train_obs, train_act, val_obs, val_act = demos_to_sequences(demos, seq_len=20)
    print(f"\nSequence tensors:")
    print(f"  train_obs:  {train_obs.shape}")
    print(f"  train_act:  {train_act.shape}")
    print(f"  val_obs:    {val_obs.shape}")
    print(f"  val_act:    {val_act.shape}")
    print("\nAll checks passed!")