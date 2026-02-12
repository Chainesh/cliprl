"""
CLIP-RL Dataset

Builds the (language, trajectory) paired dataset for CLIP training.
Each sample is one base task: its instruction + its collected trajectories.

The dataset is small (N = number of base tasks, e.g. 8).
We load everything into memory — no DataLoader needed.
"""

import torch
import os, sys, pickle
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trajectory.collector import load_trajectories, trajectories_to_tensor
from utils.config import CHECKPOINTS, BASE_TASKS, TRAJECTORY


class CLIPDataset:
    """
    Holds all (instruction, trajectory_tensor) pairs for CLIP training.

    After loading:
        self.instructions    : List[str]           — N task instructions
        self.task_ids        : List[str]           — N task ids (for logging)
        self.traj_tensors    : (N, T, 155) tensor  — padded trajectory sequences
        self.traj_lengths    : (N,)        tensor  — actual episode lengths
                                                     per task (mean across episodes)

    Note: traj_tensors has shape (N, n_episodes, T, 155).
    We store all episodes; the trajectory encoder handles mean pooling.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.instructions : List[str]        = []
        self.task_ids     : List[str]        = []
        self.all_traj_tensors : List[torch.Tensor] = []  # one per task: (n_ep, T, 155)
        self.all_traj_lengths : List[torch.Tensor] = []  # one per task: (n_ep,)

    def load(self) -> "CLIPDataset":
        """
        Load all base task trajectories from checkpoints.
        Must have run phase 1 (train) and phase 2 (trajectories) first.
        """
        from envs.task_suite import TaskRegistry
        registry = TaskRegistry()

        print("\nLoading CLIP dataset...")
        missing = []

        for env_id, instruction, task_id in registry.base_tasks():
            traj_path = os.path.join(CHECKPOINTS, f"trajectories_{task_id}.pkl")

            if not os.path.exists(traj_path):
                print(f"  ✗ Missing trajectories for {task_id} — run phase 2 first")
                missing.append(task_id)
                continue

            trajectories = load_trajectories(task_id)

            # Convert to tensors: (n_episodes, T, 155) and (n_episodes,)
            traj_tensor, lengths = trajectories_to_tensor(
                trajectories,
                max_ep_len = TRAJECTORY["max_steps"],
                device     = self.device,
            )

            self.instructions.append(instruction)
            self.task_ids.append(task_id)
            self.all_traj_tensors.append(traj_tensor)
            self.all_traj_lengths.append(lengths)

            print(f"  ✓ {task_id:50s} | {len(trajectories)} episodes")

        if missing:
            raise RuntimeError(f"Missing trajectories for: {missing}")

        print(f"\nDataset ready: {len(self.instructions)} tasks\n")
        return self

    def __len__(self):
        return len(self.instructions)

    def get_task(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """Returns (instruction, traj_tensor, traj_lengths) for one task."""
        return (
            self.instructions[idx],
            self.all_traj_tensors[idx],
            self.all_traj_lengths[idx],
        )