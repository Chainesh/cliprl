"""
Bot Demo Collector for CLIP-RL IL - CORRECTED VERSION

Key fixes:
1. Extracts actual mission strings from environments
2. Proper sequence extraction without artificial padding
3. Handles variable-length sequences correctly
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
from utils.il_config import (
    BOT_DEMOS,
    DEMOS_DIR,
    CURRICULUM_STAGES,
    get_stage_demo_count,
)


# Type aliases
Step  = Tuple[np.ndarray, int]        # (flat_obs, action)
Demo  = List[Step]                    # one episode
Demos = List[Demo]                    # collection of episodes


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


def collect_bot_demos(
    env_id:      str,
    n_episodes:  int  = BOT_DEMOS["n_episodes"],
    max_steps:   int  = BOT_DEMOS["max_steps"],
    seed:        int  = 0,
    timeout_sec: int  = BOT_DEMOS["timeout"],
    verbose:     bool = True,
    extract_missions: bool = True,  # NEW: extract actual missions
) -> Tuple[Demos, Optional[List[str]]]:  # NEW: also return missions
    """
    Collect n_episodes demonstrations from the BabyAI bot.
    
    CRITICAL: Now extracts and returns the ACTUAL mission strings from the env,
    not the hardcoded ones from the curriculum config.

    Args:
        env_id:      BabyAI gymnasium env id
        n_episodes:  number of successful demos to collect
        max_steps:   hard cutoff per episode
        seed:        base random seed
        timeout_sec: seconds before giving up on one episode
        verbose:     print progress
        extract_missions: if True, return actual missions from env

    Returns:
        demos: List of successful demos
        missions: List of actual mission strings (one per demo), or None
    """
    try:
        from minigrid.utils.baby_ai_bot import BabyAIBot
    except ImportError:
        raise ImportError(
            "BabyAIBot not found. Make sure minigrid>=2.5.0 is installed:\n"
            "  pip install minigrid==2.5.0"
        )

    demos: Demos = []
    missions: List[str] = [] if extract_missions else None
    n_failed     = 0
    n_timeout    = 0
    ep_idx       = 0

    if verbose:
        print(f"\n  Collecting {n_episodes} bot demos for {env_id}...")
        print(f"  Max steps per episode: {max_steps}  |  Timeout: {timeout_sec}s")
        if extract_missions:
            print(f"  Extracting actual mission strings from environment")
        print()

    while len(demos) < n_episodes:
        ep_idx += 1

        env = gym.make(env_id, render_mode=None)
        obs_dict, _ = env.reset(seed=seed + ep_idx)
        
        # CRITICAL FIX: Extract actual mission from environment
        actual_mission = obs_dict['mission'] if extract_missions else None

        try:
            with timeout(timeout_sec):
                bot      = BabyAIBot(env)
                episode  = []
                done     = False
                steps    = 0
                success  = False

                while not done and steps < max_steps:
                    flat_obs = flatten_obs(obs_dict)

                    try:
                        action = bot.replan()
                    except Exception:
                        break

                    episode.append((flat_obs.copy(), int(action)))
                    obs_dict, reward, terminated, truncated, _ = env.step(action)
                    done    = terminated or truncated
                    success = terminated and reward > 0
                    steps  += 1

                if success and len(episode) > 0:
                    demos.append(episode)
                    if extract_missions:
                        missions.append(actual_mission)
                    
                    if verbose and len(demos) % 500 == 0:
                        sr = len(demos) / (len(demos) + n_failed + n_timeout) * 100
                        print(
                            f"  {len(demos):5d}/{n_episodes}  |  "
                            f"failed: {n_failed}  timeout: {n_timeout}  |  "
                            f"success rate: {sr:.1f}%  |  "
                            f"ep_len: {len(episode)}"
                        )
                        if extract_missions and len(demos) == 500:
                            # Show sample missions
                            unique_missions = set(missions[-500:])
                            if len(unique_missions) == 1:
                                print(f"  Mission (fixed): '{missions[-1]}'")
                            else:
                                print(f"  Missions (variable, {len(unique_missions)} variants)")
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
        print(f"  Failed: {n_failed}  |  Timed out: {n_timeout}")
        
        if extract_missions:
            unique_missions = set(missions)
            if len(unique_missions) == 1:
                print(f"  Mission string (FIXED): '{missions[0]}'")
            else:
                print(f"  Mission strings (VARIABLE): {len(unique_missions)} different")
                for m in list(unique_missions)[:3]:
                    print(f"    - '{m}'")
                if len(unique_missions) > 3:
                    print(f"    ... and {len(unique_missions) - 3} more")
        print()

    return demos, missions


def demo_path(task_id: str) -> str:
    return os.path.join(DEMOS_DIR, f"demos_{task_id}.pkl")


def save_demos(
    demos: Demos, 
    task_id: str, 
    missions: Optional[List[str]] = None  # NEW: save missions
) -> str:
    path = demo_path(task_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    stats = {
        "n_demos"   : len(demos),
        "avg_length": float(np.mean([len(d) for d in demos])),
        "min_length": int(min(len(d) for d in demos)),
        "max_length": int(max(len(d) for d in demos)),
    }
    
    # Store missions if provided
    if missions:
        unique_missions = list(set(missions))
        stats["mission_type"] = "fixed" if len(unique_missions) == 1 else "variable"
        stats["unique_missions"] = unique_missions
        stats["mission_counts"] = {m: missions.count(m) for m in unique_missions}
    
    data = {
        "task_id": task_id,
        "demos": demos,
        "missions": missions,  # NEW: include missions
        "stats": stats
    }
    
    with open(path, "wb") as f:
        pickle.dump(data, f)
    
    print(f"  Saved {len(demos)} demos → {path}")
    print(f"  Avg length: {stats['avg_length']:.1f} | "
          f"Min: {stats['min_length']} | Max: {stats['max_length']}")
    
    if missions:
        if stats["mission_type"] == "fixed":
            print(f"  Mission (fixed): '{unique_missions[0]}'")
        else:
            print(f"  ⚠ WARNING: Variable missions detected ({len(unique_missions)} variants)")
            print(f"     You should use a fixed-mission environment instead!")
    
    return path


def load_demos(task_id: str) -> Tuple[Demos, Optional[List[str]]]:
    """Load demos and missions (if available)"""
    path = demo_path(task_id)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No demos found at {path}\n"
            f"Run bot collection first: collect_all_demos()"
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    # Handle old format (no missions stored)
    missions = data.get("missions", None)
    return data["demos"], missions


def demos_exist(task_id: str) -> bool:
    return os.path.exists(demo_path(task_id))


def demos_to_sequences(
    demos:      Demos,
    seq_len:    int,
    val_split:  float = 0.1,
    min_len:    int = 2,      # Minimum sequence length (lowered to 2 for very short demos)
    stride:     int = None,   # Stride for overlapping sequences
) -> Tuple:
    """
    Convert demos into sequences for TBPTT training.
    
    CORRECTED APPROACH:
    - Use overlapping sequences with stride for more data
    - Track actual lengths for pack_padded_sequence
    - Pad with ZEROS (not repeated states) - they'll be masked anyway
    - Auto-adjust seq_len if demos are too short
    
    Returns:
        train_obs:     (N_train, seq_len, 148)
        train_actions: (N_train, seq_len)
        train_lengths: (N_train,)  -- NEW: actual lengths
        val_obs:       (N_val, seq_len, 148)
        val_actions:   (N_val, seq_len)
        val_lengths:   (N_val,)    -- NEW: actual lengths
    """
    import torch

    # Auto-adjust seq_len if demos are too short
    max_demo_len = max(len(demo) for demo in demos) if demos else 0
    original_seq_len = seq_len
    
    if max_demo_len < seq_len:
        seq_len = max(min(max_demo_len, 30), min_len)
        print(f"  ℹ Adjusted seq_len from {original_seq_len} to {seq_len} "
              f"(max demo length: {max_demo_len})")
    
    # Set stride for overlapping sequences
    if stride is None:
        stride = max(seq_len // 2, 1)  # 50% overlap
    
    obs_seqs     = []
    action_seqs  = []
    length_seqs  = []  # NEW: track actual lengths
    
    for demo in demos:
        # Extract overlapping sequences
        for start in range(0, len(demo), stride):
            end = min(start + seq_len, len(demo))
            actual_len = end - start
            
            if actual_len < min_len:
                continue  # Skip very short sequences
            
            # Extract sequence
            obs_seq = np.stack([demo[t][0] for t in range(start, end)])
            action_seq = np.array([demo[t][1] for t in range(start, end)])
            
            # Pad to seq_len with ZEROS (will be masked by pack_padded_sequence)
            if actual_len < seq_len:
                pad_len = seq_len - actual_len
                obs_pad = np.zeros((pad_len, obs_seq.shape[1]), dtype=obs_seq.dtype)
                act_pad = np.zeros(pad_len, dtype=action_seq.dtype)
                obs_seq = np.vstack([obs_seq, obs_pad])
                action_seq = np.concatenate([action_seq, act_pad])
            
            obs_seqs.append(obs_seq)
            action_seqs.append(action_seq)
            length_seqs.append(actual_len)  # Store actual length
    
    if len(obs_seqs) == 0:
        raise ValueError(
            f"No sequences could be extracted from {len(demos)} demos. "
            f"Demo lengths: min={min(len(d) for d in demos) if demos else 0}, "
            f"max={max(len(d) for d in demos) if demos else 0}, "
            f"seq_len={seq_len}, min_len={min_len}"
        )
    
    print(f"  Extracted {len(obs_seqs):,} sequences "
          f"(seq_len={seq_len}, stride={stride}, min_len={min_len})")
    print(f"  Sequence length stats: "
          f"mean={np.mean(length_seqs):.1f}, "
          f"min={min(length_seqs)}, max={max(length_seqs)}")
    
    # Shuffle
    idx = np.random.permutation(len(obs_seqs))
    obs_seqs    = [obs_seqs[i]    for i in idx]
    action_seqs = [action_seqs[i] for i in idx]
    length_seqs = [length_seqs[i] for i in idx]
    
    # Train/val split
    n_val  = max(1, int(len(obs_seqs) * val_split))
    n_train = len(obs_seqs) - n_val
    
    def to_tensors(obs_list, act_list, len_list):
        return (
            torch.FloatTensor(np.stack(obs_list)),
            torch.LongTensor(np.stack(act_list)),
            torch.LongTensor(len_list),  # NEW: lengths tensor
        )
    
    train = to_tensors(obs_seqs[:n_train], action_seqs[:n_train], length_seqs[:n_train])
    val   = to_tensors(obs_seqs[n_train:], action_seqs[n_train:], length_seqs[n_train:])
    
    return train[0], train[1], train[2], val[0], val[1], val[2]  # Now includes lengths!


def collect_all_demos(
    n_episodes:  Optional[int] = None,
    stage_filter: Optional[int] = None,
    skip_existing: bool = True,
    verbose:     bool = True,
) -> Dict[str, Tuple[Demos, List[str]]]:
    """Collect bot demos for all tasks, extracting actual missions."""
    def make_task_id(env_id, instruction):
        env_short  = env_id.replace("BabyAI-", "").replace("-v0", "")
        instr_slug = instruction.replace(" ", "_")
        return f"{env_short}__{instr_slug}"

    if stage_filter is not None:
        task_rows = [(stage_filter, env_id, instruction) for env_id, instruction in CURRICULUM_STAGES[stage_filter]]
    else:
        task_rows = []
        for stage_idx, stage_tasks in enumerate(CURRICULUM_STAGES):
            for env_id, instruction in stage_tasks:
                task_rows.append((stage_idx, env_id, instruction))

    all_demos: Dict[str, Tuple[Demos, List[str]]] = {}

    demos_label = f"{n_episodes} eps each" if n_episodes is not None else "stage-specific eps/task"
    print(f"\n{'='*60}")
    print(f"  Collecting bot demos | {len(task_rows)} tasks | {demos_label}")
    print(f"{'='*60}")

    for stage_idx, env_id, _instruction in task_rows:
        task_id = make_task_id(env_id, _instruction)
        target_episodes = n_episodes if n_episodes is not None else get_stage_demo_count(stage_idx)

        if skip_existing and demos_exist(task_id):
            demos, missions = load_demos(task_id)
            if len(demos) >= target_episodes:
                print(f"  [{task_id}]  loading existing ({len(demos)} demos)")
                all_demos[task_id] = (demos, missions)
                continue
            print(
                f"  [{task_id}]  cached demos ({len(demos)}) below target "
                f"({target_episodes}), recollecting..."
            )

        print(f"\n  [{task_id}]  stage={stage_idx}  target={target_episodes}")
        demos, missions = collect_bot_demos(
            env_id     = env_id,
            n_episodes = target_episodes,
            verbose    = verbose,
            extract_missions = True,  # Always extract actual missions
        )
        save_demos(demos, task_id, missions)
        all_demos[task_id] = (demos, missions)

    print(f"\nAll demos collected.\n")
    return all_demos


if __name__ == "__main__":
    print("Testing mission extraction...\n")
    
    demos, missions = collect_bot_demos(
        env_id     = "BabyAI-GoToRedBall-v0",
        n_episodes = 10,
        verbose    = True,
        extract_missions = True,
    )
    
    print(f"\nCollected {len(demos)} demos")
    print(f"Unique missions: {set(missions)}")
    
    train_obs, train_act, train_len, val_obs, val_act, val_len = demos_to_sequences(
        demos, seq_len=20
    )
    print(f"\nSequence tensors:")
    print(f"  train_obs:  {train_obs.shape}")
    print(f"  train_act:  {train_act.shape}")
    print(f"  train_len:  {train_len.shape}  <- NEW!")
    print("\nAll checks passed!")
