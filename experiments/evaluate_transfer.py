"""
Transfer Evaluation for CLIP-RL.

Compares 4 initialization strategies on each target task:
    1. Random init       — no transfer, train from scratch
    2. Language-only     — Algorithm 1 from the paper
    3. CLIP-RL           — Algorithm 2, our contribution
    4. Top-1 language    — hard assignment to best language match

For each strategy × target task:
    - Initialize policy with the given strategy
    - Train with PPO for a fixed budget (fewer steps than base tasks)
    - Record reward curve over training
    - Measure: steps to reach threshold reward, AUC of reward curve

Results saved to results/transfer_results.pkl and plots to results/
"""

import torch
import numpy as np
import os, sys, time, pickle
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ppo_agent       import PPOAgent
from agents.policy_network  import PolicyNetwork
from transfer.initializer   import (
    load_base_policies,
    initialize_random,
    initialize_language_only,
    initialize_clip_rl,
    initialize_top1_language,
)
from clip_rl_model.train_clip import load_trained_clip
from utils.config import CHECKPOINTS, RESULTS, PPO
from envs.task_suite import TaskRegistry


# ── Evaluation config ─────────────────────────────────────────────────────────

TRANSFER_TIMESTEPS = 100_000   # training budget for target task (shorter than base)
EVAL_EPISODES      = 20        # episodes for each evaluation call
EVAL_EVERY         = 5         # evaluate every N PPO updates
SUCCESS_THRESHOLD  = 0.7       # reward threshold to declare "converged"

METHODS = ["random", "language_only", "clip_rl", "top1_language"]

METHOD_LABELS = {
    "random"       : "Random Init",
    "language_only": "Language-Only (Algo 1)",
    "clip_rl"      : "CLIP-RL (Algo 2)",
    "top1_language": "Top-1 Language",
}

METHOD_COLORS = {
    "random"       : "#888888",
    "language_only": "#f4a261",
    "clip_rl"      : "#2a9d8f",
    "top1_language": "#e76f51",
}


# ─── Single method evaluation ─────────────────────────────────────────────────

def evaluate_single_method(
    method:             str,
    target_env_id:      str,
    target_task_id:     str,
    target_instruction: str,
    base_policies:      List[PolicyNetwork],
    base_instructions:  List[str],
    clip_model,
    device:             str = "cpu",
    seed:               int = 42,
) -> Dict:
    """
    Initialize policy with given method, train on target task,
    record reward curve.

    Returns:
        dict with keys: method, rewards, timesteps, steps_to_threshold,
                        auc, final_reward
    """
    print(f"\n    [{method}] initializing...")

    # ── Initialize policy ─────────────────────────────────────────────────────
    if method == "random":
        init_policy = initialize_random()

    elif method == "language_only":
        init_policy = initialize_language_only(
            target_instruction, base_policies, base_instructions, verbose=True
        )

    elif method == "clip_rl":
        init_policy = initialize_clip_rl(
            target_instruction, base_policies, base_instructions,
            clip_model, verbose=True
        )

    elif method == "top1_language":
        init_policy = initialize_top1_language(
            target_instruction, base_policies, base_instructions
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # ── Create agent with initialized policy ──────────────────────────────────
    agent = PPOAgent(
        env_id  = target_env_id,
        task_id = f"{target_task_id}__{method}",
        seed    = seed,
        device  = device,
        verbose = False,
    )
    agent.policy.load_state_dict(init_policy.state_dict())
    agent.policy = agent.policy.to(agent.device)

    # ── Train with periodic evaluation ────────────────────────────────────────
    from envs.task_suite import make_env

    env          = make_env(target_env_id, seed=seed)
    rewards      = []
    timesteps    = []
    total_steps  = 0
    n_updates    = TRANSFER_TIMESTEPS // agent.n_steps

    agent.policy.train()

    for update in range(1, n_updates + 1):
        # Collect rollout
        mean_reward, _ = agent._collect_rollout(env)
        # PPO update
        agent._ppo_update()
        total_steps += agent.n_steps

        if update % EVAL_EVERY == 0:
            metrics = agent.evaluate(n_episodes=EVAL_EPISODES)
            rewards.append(metrics["mean_reward"])
            timesteps.append(total_steps)
            print(f"      step {total_steps:6,} | reward: {metrics['mean_reward']:.3f} | "
                  f"success: {metrics['success_rate']*100:.0f}%")

    env.close()

    # ── Compute summary metrics ───────────────────────────────────────────────
    rewards_arr = np.array(rewards)

    # Steps to reach threshold (interpolated)
    steps_to_threshold = TRANSFER_TIMESTEPS  # default: never reached
    for i, r in enumerate(rewards_arr):
        if r >= SUCCESS_THRESHOLD:
            steps_to_threshold = timesteps[i]
            break

    # AUC — area under reward curve (higher = better sample efficiency)
    auc = float(np.trapz(rewards_arr, timesteps)) if len(rewards) > 1 else 0.0

    return {
        "method"             : method,
        "label"              : METHOD_LABELS[method],
        "rewards"            : rewards_arr.tolist(),
        "timesteps"          : timesteps,
        "steps_to_threshold" : steps_to_threshold,
        "auc"                : auc,
        "final_reward"       : float(rewards_arr[-1]) if len(rewards) > 0 else 0.0,
    }


# ─── Full evaluation across all target tasks ─────────────────────────────────

def run_full_evaluation(
    device:  str = "cpu",
    n_seeds: int = 3,
) -> Dict:
    """
    Run all methods on all target tasks with multiple seeds.
    Saves results and plots.

    Returns:
        all_results: dict of {target_task_id: {method: [seed_results]}}
    """
    os.makedirs(RESULTS, exist_ok=True)

    registry = TaskRegistry()

    # ── Load base policies and CLIP model ─────────────────────────────────────
    print("\nLoading base policies...")
    base_policies, base_task_ids, base_instructions = load_base_policies(device)

    print("\nLoading CLIP model...")
    clip_model = load_trained_clip(device)

    all_results = {}

    # ── Iterate over target tasks ─────────────────────────────────────────────
    for env_id, instruction, task_id in registry.target_tasks():
        print(f"\n{'='*60}")
        print(f"  Target task: {task_id}")
        print(f"  Instruction: '{instruction}'")
        print(f"{'='*60}")

        task_results = {m: [] for m in METHODS}

        for seed in range(n_seeds):
            print(f"\n  Seed {seed + 1}/{n_seeds}")

            for method in METHODS:
                result = evaluate_single_method(
                    method             = method,
                    target_env_id      = env_id,
                    target_task_id     = task_id,
                    target_instruction = instruction,
                    base_policies      = base_policies,
                    base_instructions  = base_instructions,
                    clip_model         = clip_model,
                    device             = device,
                    seed               = seed * 100 + 42,
                )
                task_results[method].append(result)

        all_results[task_id] = task_results

        # Plot this target task
        _plot_task_results(task_id, instruction, task_results)

    # ── Save all results ──────────────────────────────────────────────────────
    results_path = os.path.join(RESULTS, "transfer_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved all results → {results_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    _print_summary_table(all_results)

    return all_results


def _plot_task_results(task_id: str, instruction: str, task_results: dict):
    """Plot reward curves for all methods on one target task."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Transfer: '{instruction}'", fontsize=13)

    # ── Left: reward curves (mean ± std across seeds) ─────────────────────────
    ax = axes[0]
    for method in METHODS:
        results_list = task_results[method]
        if not results_list:
            continue

        # Align timesteps (use first seed's as reference)
        timesteps = results_list[0]["timesteps"]
        rewards   = np.array([r["rewards"] for r in results_list])

        mean_r = rewards.mean(axis=0)
        std_r  = rewards.std(axis=0)

        ax.plot(timesteps, mean_r, label=METHOD_LABELS[method],
                color=METHOD_COLORS[method], linewidth=2)
        ax.fill_between(timesteps, mean_r - std_r, mean_r + std_r,
                        alpha=0.2, color=METHOD_COLORS[method])

    ax.axhline(y=SUCCESS_THRESHOLD, color="black", linestyle="--",
               alpha=0.5, label=f"Threshold ({SUCCESS_THRESHOLD})")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Learning Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Right: bar chart of steps to threshold ────────────────────────────────
    ax = axes[1]
    method_names   = []
    mean_steps     = []
    std_steps      = []

    for method in METHODS:
        results_list = task_results[method]
        if not results_list:
            continue
        steps = [r["steps_to_threshold"] for r in results_list]
        method_names.append(METHOD_LABELS[method])
        mean_steps.append(np.mean(steps))
        std_steps.append(np.std(steps))

    colors = [METHOD_COLORS[m] for m in METHODS if task_results[m]]
    bars = ax.bar(method_names, mean_steps, yerr=std_steps,
                  color=colors, alpha=0.8, capsize=5)
    ax.set_ylabel("Steps to Convergence")
    ax.set_title("Sample Efficiency")
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate bars with speedup vs random
    if mean_steps:
        random_steps = mean_steps[0]  # assuming random is first
        for bar, steps in zip(bars, mean_steps):
            if random_steps > 0 and steps < random_steps:
                speedup = random_steps / steps
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(std_steps) * 0.1,
                        f"{speedup:.1f}×", ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color='green')

    plt.tight_layout()
    out_path = os.path.join(RESULTS, f"transfer_{task_id}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  Saved plot → {out_path}")


def _print_summary_table(all_results: dict):
    """Print a clean comparison table."""
    print(f"\n{'='*70}")
    print(f"  TRANSFER RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Target Task':<35} {'Method':<25} {'Steps↓':>10}  {'AUC↑':>10}")
    print(f"  {'-'*70}")

    for task_id, task_results in all_results.items():
        for method in METHODS:
            results_list = task_results[method]
            if not results_list:
                continue
            mean_steps = np.mean([r["steps_to_threshold"] for r in results_list])
            mean_auc   = np.mean([r["auc"] for r in results_list])
            label      = METHOD_LABELS[method]
            print(f"  {task_id:<35} {label:<25} {mean_steps:>10,.0f}  {mean_auc:>10.1f}")
        print()

    print(f"  Steps↓ = fewer steps to converge is better")
    print(f"  AUC↑   = higher area under reward curve is better")
    print(f"{'='*70}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_full_evaluation(device=device, n_seeds=3)