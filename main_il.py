"""
CLIP-RL: Main entry point (with IL support).

Original phases (simple tasks, flat MLP policy):
    python main.py --phase verify
    python main.py --phase train          # PPO on 8 base tasks
    python main.py --phase trajectories   # collect trajectories
    python main.py --phase clip           # train CLIP alignment
    python main.py --phase transfer       # evaluate transfer

New IL phases (hard tasks, recurrent FiLM+GRU policy):
    python main.py --phase collect_demos  # collect bot demos for all stages
    python main.py --phase il_train       # behavioral cloning on demos
    python main.py --phase il_eval        # evaluate IL policies
    python main.py --phase curriculum     # run full curriculum to BossLevel
    python main.py --phase il_all         # collect_demos → curriculum → clip → transfer

Optional flags:
    --stage     0        curriculum stage to train (0-5)
    --n_demos   10000    bot demos per task
    --epochs    20       IL training epochs
    --large              use large model (2048 GRU) for hard stages
    --rl_stages 3,4,5    comma-separated stages to also do RL fine-tune
    --device    cuda
"""

import torch
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.il_config import ALL_IL_TASKS, CHECKPOINTS
from agents.il_agent import ILAgent
from envs.task_suite import FlatObsWrapper, action_to_onehot
from utils.config import TRAJECTORY
from trajectory.collector import save_trajectories
import gymnasium as gym
import numpy as np
from typing import Optional   

# ─── Original phases (unchanged) ──────────────────────────────────────────────

def phase_verify():
    from envs.task_suite import TaskRegistry, verify_all_envs
    registry = TaskRegistry()
    registry.print_summary()
    verify_all_envs()

def phase_train(timesteps, device):
    from agents.ppo_agent import train_all_base_tasks
    agents = train_all_base_tasks(total_timesteps=timesteps)
    print(f"\nPhase 1 complete. {len(agents)} policies saved.\n")

def phase_trajectories(n_episodes, device):
    from trajectory.collector import collect_all_base_trajectories
    collect_all_base_trajectories(n_episodes=n_episodes, device=device)
    print(f"\nPhase 2 complete.\n")

def phase_clip(epochs, device):
    from clip_rl_model.train_clip import train_clip
    train_clip(device=device, epochs=epochs)
    print(f"\nPhase 3 complete.\n")

def phase_transfer(n_seeds, device):
    from experiments.evaluate_transfer import run_full_evaluation
    run_full_evaluation(device=device, n_seeds=n_seeds)
    print(f"\nPhase 4 complete.\n")


# ─── IL phases ────────────────────────────────────────────────────────────────

def phase_collect_demos(n_demos: int, stage: Optional[int] = None):
    """Collect bot demonstrations for all tasks (or one stage)."""
    from trajectory.bot_collector import collect_all_demos
    collect_all_demos(
        n_episodes   = n_demos,
        stage_filter = stage,
        skip_existing = True,
    )
    print(f"\nDemo collection complete.\n")


def phase_il_train(
    stage:      int,
    device:     str,
    large:      bool = False,
    n_demos:    int  = 10_000,
):
    """Train IL (behavioral cloning) on one curriculum stage."""
    from agents.il_agent import train_il_stage
    agents = train_il_stage(
        stage_idx    = stage,
        n_demos      = n_demos,
        device       = device,
        large_model  = large,
        skip_existing = True,
    )
    print(f"\nIL stage {stage} complete. {len(agents)} policies trained.\n")


def phase_il_eval(stage: int, device: str):
    """Evaluate IL policies on a given stage."""
    from utils.il_config import CURRICULUM_STAGES, ALL_IL_TASKS, CHECKPOINTS
    from agents.il_agent import ILAgent
    import os

    def make_task_id(env_id, instruction):
        return f"{env_id.replace('BabyAI-','').replace('-v0','')}_{instruction.replace(' ','_')}"

    all_instructions = [instr for _, instr in ALL_IL_TASKS]
    stage_tasks      = CURRICULUM_STAGES[stage]

    print(f"\nEvaluating IL Stage {stage}...\n")
    for env_id, instruction in stage_tasks:
        task_id   = make_task_id(env_id, instruction)
        ckpt_path = os.path.join(CHECKPOINTS, f"il_policy_{task_id}.pt")

        if not os.path.exists(ckpt_path):
            print(f"  [{task_id}]  No checkpoint found — skipping")
            continue

        agent   = ILAgent.load(task_id, all_instructions, device=device, verbose=False)
        metrics = agent.evaluate(n_episodes=50)
        print(
            f"  {task_id:55s}  "
            f"success: {metrics['success_rate']*100:5.1f}%  "
            f"reward: {metrics['mean_reward']:.3f}"
        )


def phase_curriculum(
    device:         str,
    large:          bool,
    rl_stages:      list,
    start_stage:    int = 0,
):
    """Run the full curriculum from start_stage to BossLevel."""
    from agents.curriculum import CurriculumManager

    manager = CurriculumManager(
        device      = device,
        large_model = large,
        verbose     = True,
        start_stage = start_stage,
    )
    manager.run(rl_finetune_stages=rl_stages)
    print(f"\nCurriculum complete.\n")


def phase_il_trajectories(device: str):
    """
    Collect trajectories from trained IL policies for CLIP training.
    Replaces phase_trajectories() when using the recurrent IL policies.
    """
    all_instructions = [instr for _, instr in ALL_IL_TASKS]

    def make_task_id(env_id, instruction):
        env_short  = env_id.replace("BabyAI-", "").replace("-v0", "")
        instr_slug = instruction.replace(" ", "_")
        return f"{env_short}__{instr_slug}"

    print(f"\nCollecting trajectories from IL policies...")

    for env_id, instruction in ALL_IL_TASKS:
        task_id   = make_task_id(env_id, instruction)
        ckpt_path = os.path.join(CHECKPOINTS, f"il_policy_{task_id}.pt")

        if not os.path.exists(ckpt_path):
            print(f"  [{task_id}]  No IL checkpoint — skipping")
            continue

        agent = ILAgent.load(task_id, all_instructions, device=device, verbose=False)
        policy = agent.policy
        policy.eval()
        tokens = agent.tokens.to(torch.device(device))

        env = gym.make(env_id, render_mode=None)
        env = FlatObsWrapper(env)

        trajectories = []
        for ep in range(TRAJECTORY["n_episodes"]):
            obs, _ = env.reset(seed=ep)
            hidden = policy.init_hidden(1, torch.device(device))
            episode = []
            done = False
            steps = 0

            while not done and steps < TRAJECTORY["max_steps"]:
                action, hidden = policy.act(obs, tokens, hidden, deterministic=True)
                episode.append((obs.copy(), action))
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1

            trajectories.append(episode)

        env.close()
        save_trajectories(trajectories, task_id)

        ep_lengths = [len(ep) for ep in trajectories]
        print(f"  [{task_id}]  {len(trajectories)} eps  "
              f"avg_len: {np.mean(ep_lengths):.1f}")

    print(f"\nIL trajectory collection complete.\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

from typing import Optional

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP-RL Pipeline (with IL)")

    parser.add_argument("--phase", default="verify", choices=[
        # Original phases
        "verify", "train", "trajectories", "clip", "transfer", "all",
        # IL phases
        "collect_demos", "il_train", "il_eval", "il_trajectories",
        "curriculum", "il_all",
    ])

    # Original flags
    parser.add_argument("--timesteps", type=int,   default=500_000)
    parser.add_argument("--episodes",  type=int,   default=50)
    parser.add_argument("--epochs",    type=int,   default=100)
    parser.add_argument("--seeds",     type=int,   default=3)
    parser.add_argument("--device",    type=str,   default="auto")

    # IL flags
    parser.add_argument("--stage",     type=int,   default=0,
                        help="Curriculum stage index (0-5)")
    parser.add_argument("--n_demos",   type=int,   default=10_000,
                        help="Bot demos per task")
    parser.add_argument("--large",     action="store_true",
                        help="Use large model (2048-dim GRU) for hard stages")
    parser.add_argument("--rl_stages", type=str,   default="3,4,5",
                        help="Comma-separated stage indices to also PPO fine-tune")
    parser.add_argument("--start_stage", type=int, default=0,
                        help="Resume curriculum from this stage")

    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    rl_stages = [int(s) for s in args.rl_stages.split(",") if s.strip()]

    print(f"\n{'='*55}\n  CLIP-RL  |  phase={args.phase}  device={args.device}\n{'='*55}\n")

    # ── Original phases ────────────────────────────────────────────────────────
    if args.phase in ("verify", "all"):
        phase_verify()
    if args.phase in ("train", "all"):
        phase_train(args.timesteps, args.device)
    if args.phase in ("trajectories", "all"):
        phase_trajectories(args.episodes, args.device)
    if args.phase in ("clip", "all"):
        phase_clip(args.epochs, args.device)
    if args.phase in ("transfer", "all"):
        phase_transfer(args.seeds, args.device)

    # ── IL phases ──────────────────────────────────────────────────────────────
    if args.phase == "collect_demos":
        phase_collect_demos(args.n_demos, stage=args.stage)

    if args.phase == "il_train":
        phase_il_train(
            stage   = args.stage,
            device  = args.device,
            large   = args.large,
            n_demos = args.n_demos,
        )

    if args.phase == "il_eval":
        phase_il_eval(args.stage, args.device)

    if args.phase == "il_trajectories":
        phase_il_trajectories(args.device)

    if args.phase == "curriculum":
        phase_curriculum(
            device      = args.device,
            large       = args.large,
            rl_stages   = rl_stages,
            start_stage = args.start_stage,
        )

    # ── Full IL pipeline ───────────────────────────────────────────────────────
    if args.phase == "il_all":
        # 1. Collect all demos
        phase_collect_demos(args.n_demos)
        # 2. Run curriculum (IL + optional RL fine-tune per stage)
        phase_curriculum(
            device      = args.device,
            large       = args.large,
            rl_stages   = rl_stages,
            start_stage = args.start_stage,
        )
        # 3. Collect trajectories from trained IL policies
        phase_il_trajectories(args.device)
        # 4. Train CLIP alignment on IL policy trajectories
        phase_clip(args.epochs, args.device)
        # 5. Evaluate transfer
        phase_transfer(args.seeds, args.device)