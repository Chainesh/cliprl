"""
Environment wrappers and task registry for CLIP-RL.

BabyAI obs structure:
    obs["image"]     → (7, 7, 3) int array  — partial view grid
                        dim 0: row, dim 1: col
                        dim 2: [object_type, color, state]
                        object types: 0=unseen,1=empty,2=wall,3=floor,
                                      4=door,5=key,6=ball,7=box,8=goal,9=lava
                        colors:       0=red,1=green,2=blue,3=purple,
                                      4=yellow,5=grey
                        state:        0=open,1=closed,2=locked
    obs["direction"] → int  0=right, 1=down, 2=left, 3=up
    obs["mission"]   → str  e.g. "go to the red ball"

Actions (Discrete 7):
    0=turn_left, 1=turn_right, 2=move_forward,
    3=pick_up, 4=drop, 5=toggle, 6=done
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import OBS_DIM, OBS_IMAGE_DIM, ACTION_DIM, BASE_TASKS, TARGET_TASKS


class FlatObsWrapper(gym.ObservationWrapper):
    """
    Flattens BabyAI's dict observation into a single 1D vector.

    Input:
        obs["image"]     (7,7,3) → float32, normalized to [0,1]
        obs["direction"] int     → one-hot (4,)

    Output:
        flat vector of shape (148,)
        = image_flat (147,) + direction_onehot (1 scalar, cast to float)

    Note: we keep direction as a scalar (not one-hot) to keep dim=148
    matching OBS_DIM in config. Direction 0-3 normalized to [0,1].
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Override the observation space to flat Box
        self.observation_space = spaces.Box(
            low   = 0.0,
            high  = 1.0,
            shape = (OBS_DIM,),
            dtype = np.float32,
        )

    def observation(self, obs: Dict[str, Any]) -> np.ndarray:
        return flatten_obs(obs)


def flatten_obs(obs: Dict[str, Any]) -> np.ndarray:
    """
    Standalone flatten function — used in both the wrapper and
    during trajectory collection where we have raw obs dicts.

    Returns float32 array of shape (148,).
    """
    # image: (7,7,3) int → flatten → normalize by max object_type value (10)
    image = obs["image"].astype(np.float32).flatten() / 10.0   # (147,)

    # direction: 0-3 scalar → normalize to [0, 1]
    direction = np.array([obs["direction"] / 3.0], dtype=np.float32)  # (1,)

    return np.concatenate([image, direction])  # (148,)


def action_to_onehot(action: int) -> np.ndarray:
    """
    Converts integer action to one-hot vector of shape (7,).
    Used when building trajectory tensors for the GRU encoder.
    """
    onehot = np.zeros(ACTION_DIM, dtype=np.float32)
    onehot[action] = 1.0
    return onehot


def make_env(env_id: str, seed: Optional[int] = None) -> gym.Env:
    """
    Creates a BabyAI environment with the FlatObsWrapper applied.

    Args:
        env_id: gymnasium environment id, e.g. 'BabyAI-GoToRedBall-v0'
        seed:   optional random seed for reproducibility

    Returns:
        Wrapped gym.Env with flat observation space
    """
    env = gym.make(env_id, render_mode=None)
    env = FlatObsWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_vec_env(env_id: str, n_envs: int = 4, seed: int = 0):
    """
    Creates a vectorized environment for PPO training.
    Uses stable_baselines3's SubprocVecEnv for parallel collection.

    Args:
        env_id: gymnasium environment id
        n_envs: number of parallel environments
        seed:   base random seed

    Returns:
        VecEnv wrapping n_envs instances of the environment
    """
    from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def make_single(rank):
        def _init():
            env = gym.make(env_id, render_mode=None)
            env = FlatObsWrapper(env)
            env.reset(seed=seed + rank)
            return env
        return _init

    vec_env = SubprocVecEnv([make_single(i) for i in range(n_envs)])
    return vec_env


# ─── Task Registry ────────────────────────────────────────────────────────────

class TaskRegistry:
    """
    Central registry of all tasks (base + target).
    Provides consistent task_id strings used for checkpointing.

    task_id: sanitized string from env_id + instruction
             e.g. "GoToRedBall_go_to_the_red_ball"
    """

    def __init__(self):
        self._base   = BASE_TASKS
        self._target = TARGET_TASKS

    @staticmethod
    def make_task_id(env_id: str, instruction: str) -> str:
        """
        Creates a filesystem-safe task identifier.

        Example:
            make_task_id("BabyAI-GoToRedBall-v0", "go to the red ball")
            → "GoToRedBall_go_to_the_red_ball"
        """
        # strip "BabyAI-" prefix and "-v0" suffix
        env_short = env_id.replace("BabyAI-", "").replace("-v0", "")
        instr_slug = instruction.replace(" ", "_")
        return f"{env_short}__{instr_slug}"

    def base_tasks(self) -> list:
        """Returns list of (env_id, instruction, task_id) for base tasks."""
        return [
            (env_id, instr, self.make_task_id(env_id, instr))
            for env_id, instr in self._base
        ]

    def target_tasks(self) -> list:
        """Returns list of (env_id, instruction, task_id) for target tasks."""
        return [
            (env_id, instr, self.make_task_id(env_id, instr))
            for env_id, instr in self._target
        ]

    def all_tasks(self) -> list:
        return self.base_tasks() + self.target_tasks()

    def print_summary(self):
        print("\n" + "="*60)
        print(f"  CLIP-RL Task Registry")
        print("="*60)
        print(f"\n  BASE TASKS ({len(self._base)}):")
        for env_id, instr, task_id in self.base_tasks():
            print(f"    [{task_id}]")
            print(f"      env: {env_id}")
            print(f"      instruction: \"{instr}\"")
        print(f"\n  TARGET TASKS ({len(self._target)}):")
        for env_id, instr, task_id in self.target_tasks():
            print(f"    [{task_id}]")
            print(f"      env: {env_id}")
            print(f"      instruction: \"{instr}\"")
        print("="*60 + "\n")


# ─── Sanity Check ─────────────────────────────────────────────────────────────

def verify_env(env_id: str, verbose: bool = True) -> bool:
    """
    Spins up one environment, runs a few random steps, verifies
    obs shape and action space match our config.

    Returns True if everything looks good.
    """
    try:
        env = make_env(env_id, seed=42)
        obs, info = env.reset()

        assert obs.shape == (OBS_DIM,), \
            f"Expected obs shape ({OBS_DIM},), got {obs.shape}"
        assert env.action_space.n == ACTION_DIM, \
            f"Expected {ACTION_DIM} actions, got {env.action_space.n}"

        # run 5 random steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (OBS_DIM,)

        env.close()

        if verbose:
            print(f"  ✓ {env_id} — obs shape: ({OBS_DIM},), actions: {ACTION_DIM}")
        return True

    except Exception as e:
        if verbose:
            print(f"  ✗ {env_id} — {e}")
        return False


def verify_all_envs() -> None:
    """Run verify_env on every registered task."""
    registry = TaskRegistry()
    print("\nVerifying all environments...")
    all_ok = True
    for env_id, instr, task_id in registry.all_tasks():
        ok = verify_env(env_id, verbose=True)
        if not ok:
            all_ok = False
    if all_ok:
        print("\nAll environments verified successfully!\n")
    else:
        print("\nSome environments failed — check BabyAI installation.\n")


# ─── Main (quick smoke test) ──────────────────────────────────────────────────

if __name__ == "__main__":
    registry = TaskRegistry()
    registry.print_summary()
    verify_all_envs()