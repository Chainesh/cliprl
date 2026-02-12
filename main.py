"""
CLIP-RL: Main entry point.

    python main.py --phase verify          # verify environments work
    python main.py --phase train           # Phase 1: train base policies
    python main.py --phase trajectories    # Phase 2: collect trajectories
    python main.py --phase clip            # Phase 3: train CLIP alignment
    python main.py --phase transfer        # Phase 4: evaluate transfer
    python main.py --phase all             # run everything end to end

    Optional flags:
    --timesteps 500000   PPO training steps per task  (phase: train)
    --episodes  50       trajectories per policy      (phase: trajectories)
    --epochs    100      CLIP training epochs          (phase: clip)
    --seeds     3        evaluation seeds              (phase: transfer)
    --device    cuda     cpu / cuda
"""
import torch
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def phase_verify():
    from envs.task_suite import TaskRegistry, verify_all_envs
    registry = TaskRegistry()
    registry.print_summary()
    verify_all_envs()

def phase_train(timesteps, device):
    from agents.ppo_agent import train_all_base_tasks
    agents = train_all_base_tasks(total_timesteps=timesteps)
    print(f"\nPhase 1 complete. {len(agents)} policies saved to checkpoints/\n")

def phase_trajectories(n_episodes, device):
    from trajectory.collector import collect_all_base_trajectories
    trajs = collect_all_base_trajectories(n_episodes=n_episodes, device=device)
    print(f"\nPhase 2 complete. Trajectories saved to checkpoints/\n")

def phase_clip(epochs, device):
    from clip_rl_model.train_clip import train_clip
    train_clip(device=device, epochs=epochs)
    print(f"\nPhase 3 complete. CLIP model saved to checkpoints/\n")

def phase_transfer(n_seeds, device):
    from experiments.evaluate_transfer import run_full_evaluation
    run_full_evaluation(device=device, n_seeds=n_seeds)
    print(f"\nPhase 4 complete. Results saved to results/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP-RL Pipeline")
    parser.add_argument("--phase", default="verify",
                        choices=["verify","train","trajectories","clip","transfer","all"])
    parser.add_argument("--timesteps", type=int,   default=500_000)
    parser.add_argument("--episodes",  type=int,   default=50)
    parser.add_argument("--epochs",    type=int,   default=100)
    parser.add_argument("--seeds",     type=int,   default=3)
    parser.add_argument("--device",    type=str,   default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*55}\n  CLIP-RL  |  phase={args.phase}  device={args.device}\n{'='*55}\n")

    if args.phase in ("verify",       "all"): phase_verify()
    if args.phase in ("train",        "all"): phase_train(args.timesteps, args.device)
    if args.phase in ("trajectories", "all"): phase_trajectories(args.episodes, args.device)
    if args.phase in ("clip",         "all"): phase_clip(args.epochs, args.device)
    if args.phase in ("transfer",     "all"): phase_transfer(args.seeds, args.device)