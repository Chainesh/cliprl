#!/usr/bin/env python3
"""
Multi-Environment Demo Collection with Complexity-Based Scaling

Collect demos for one or more environments with automatic complexity detection.

Usage Examples:
    # Single environment
    python test_simple_env.py BabyAI-GoToObj-v0
    
    # Multiple environments (auto-scales based on complexity)
    python test_simple_env.py BabyAI-GoToObj-v0 BabyAI-Open-v0 BabyAI-PickupLoc-v0
    
    # Override demo counts for all
    python test_simple_env.py BabyAI-GoToObj-v0 BabyAI-Open-v0 --demos 15000
    
Complexity Auto-Scaling:
    - Simple (10k):    GoToObj, GoToLocal, Open
    - Medium (20k):    PickupLoc, PutNextLocal, PutNextS6N3
    - Hard (30k):      GoToSeqS5R2, SynthS5R2
    - Extreme (50k):   BossLevel
"""

import sys
import gymnasium as gym
import minigrid.envs.babyai
from collections import defaultdict

sys.path.insert(0, '.')
from trajectory.bot_collector import collect_bot_demos, save_demos, demos_exist
from envs.task_suite import TaskRegistry


# ═══════════════════════════════════════════════════════════════════════════
# COMPLEXITY CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

ENV_COMPLEXITY = {
    # Simple environments (10k demos)
    'BabyAI-GoToObj-v0': ('simple', 10000),
    'BabyAI-GoToLocal-v0': ('simple', 10000),
    'BabyAI-GoToRedBall-v0': ('simple', 10000),
    'BabyAI-GoToRedBallGrey-v0': ('simple', 10000),
    'BabyAI-GoToRedBallNoDists-v0': ('simple', 10000),
    'BabyAI-Open-v0': ('simple', 10000),
    
    # Medium environments (20k demos)
    'BabyAI-PickupLoc-v0': ('medium', 20000),
    'BabyAI-PutNextLocal-v0': ('medium', 20000),
    'BabyAI-PutNextS6N3-v0': ('medium', 20000),
    'BabyAI-Unlock-v0': ('medium', 20000),
    'BabyAI-UnblockPickup-v0': ('medium', 20000),
    
    # Hard environments (30k demos)
    'BabyAI-GoToSeqS5R2-v0': ('hard', 30000),
    'BabyAI-SynthS5R2-v0': ('hard', 30000),
    'BabyAI-GoToImpUnlock-v0': ('hard', 30000),
    'BabyAI-GoToObjMaze-v0': ('hard', 30000),
    
    # Extreme environments (50k demos)
    'BabyAI-BossLevel-v0': ('extreme', 50000),
}


def get_env_info(env_id):
    """Get complexity tier and default demo count."""
    return ENV_COMPLEXITY.get(env_id, ('simple', 10000))


# ═══════════════════════════════════════════════════════════════════════════
# COLLECTION LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def collect_multi_mission_demos(
    env_id,
    target_episodes_per_mission=1000,
    max_total_episodes=None,
    verbose=True,
):
    """
    Collect demos for all mission variants in an environment.
    
    Returns:
        dict: {mission_string: [demos]}
    """
    if max_total_episodes is None:
        max_total_episodes = target_episodes_per_mission * 50  # Safety limit
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Collecting demos for: {env_id}")
        print(f"Target: {target_episodes_per_mission:,} demos per mission variant")
        print(f"Max total: {max_total_episodes:,} demos")
        print(f"{'='*70}\n")
    
    mission_demos = defaultdict(list)
    mission_counts = defaultdict(int)
    total_collected = 0
    
    while total_collected < max_total_episodes:
        # Collect batch
        batch_size = 100
        batch_demos, batch_missions = collect_bot_demos(
            env_id=env_id,
            n_episodes=batch_size,
            verbose=False,
            extract_missions=True,
        )
        
        # Group by mission
        for demo, mission in zip(batch_demos, batch_missions):
            if mission_counts[mission] < target_episodes_per_mission:
                mission_demos[mission].append(demo)
                mission_counts[mission] += 1
        
        total_collected += len(batch_demos)
        
        # Show progress
        if verbose and (total_collected % 500 == 0 or total_collected < 500):
            print(f"\nProgress per mission variant:")
            for mission in sorted(mission_counts.keys()):
                count = mission_counts[mission]
                progress = min(count / target_episodes_per_mission, 1.0)
                bar_len = int(progress * 30)
                bar = '█' * bar_len + '░' * (30 - bar_len)
                
                # Truncate mission if too long
                display_mission = mission if len(mission) <= 45 else mission[:42] + '...'
                print(f"    {bar} {count:5d}/{target_episodes_per_mission:5d}  '{display_mission}'")
        
        # Check if all missions have enough demos
        if mission_counts:
            min_count = min(mission_counts.values())
            if min_count >= target_episodes_per_mission:
                if verbose:
                    print(f"\n✓ All {len(mission_counts)} mission variants have {target_episodes_per_mission:,}+ demos!")
                break
        
        # Safety check
        if total_collected >= max_total_episodes:
            if verbose:
                print(f"\n⚠ Reached max total episodes ({max_total_episodes:,})")
            break
    
    return dict(mission_demos)


def save_multi_mission_demos(env_id, mission_demos_dict):
    """Save each mission variant as a separate task."""
    print(f"\n{'='*70}")
    print(f"Saving {len(mission_demos_dict)} mission variants...")
    print(f"{'='*70}\n")
    
    task_ids = []
    
    for mission, demos in mission_demos_dict.items():
        task_id = TaskRegistry.make_task_id(env_id, mission)
        save_demos(demos, task_id, missions=[mission] * len(demos))
        task_ids.append(task_id)
        
        print(f"  ✓ Saved: {task_id}")
        print(f"    Mission: '{mission}'")
        print(f"    Demos: {len(demos):,}\n")
    
    print(f"{'='*70}")
    print(f"✓ Successfully saved {len(task_ids)} tasks for {env_id}")
    print(f"{'='*70}\n")
    
    return task_ids


# ═══════════════════════════════════════════════════════════════════════════
# MAIN COLLECTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def collect_environment(env_id, demos_per_mission=None, skip_existing=True):
    """
    Collect demos for one environment.
    
    Args:
        env_id: Environment ID
        demos_per_mission: Override default demo count, or None for auto
        skip_existing: Skip if already collected
    """
    # Get complexity info
    complexity, default_demos = get_env_info(env_id)
    target_demos = demos_per_mission if demos_per_mission else default_demos
    
    print(f"\n{'='*70}")
    print(f"ENVIRONMENT: {env_id}")
    print(f"Complexity: {complexity.upper()}")
    print(f"Target demos per variant: {target_demos:,}")
    print(f"{'='*70}")
    
    # Check for existing demos
    if skip_existing:
        # Quick check - discover a few missions and see if they exist
        env = gym.make(env_id, render_mode=None)
        sample_missions = []
        for seed in range(20):
            obs, _ = env.reset(seed=seed)
            sample_missions.append(obs['mission'])
        env.close()
        
        sample_missions = list(set(sample_missions))[:3]  # Check first 3
        existing = sum(1 for m in sample_missions if demos_exist(TaskRegistry.make_task_id(env_id, m)))
        
        if existing == len(sample_missions) and existing > 0:
            print(f"\n✓ Appears to be already collected (found {existing} existing tasks)")
            response = input("Skip this environment? [Y/n]: ")
            if response.lower() != 'n':
                print("Skipped.\n")
                return []
    
    # Collect demos
    mission_demos = collect_multi_mission_demos(
        env_id=env_id,
        target_episodes_per_mission=target_demos,
        max_total_episodes=target_demos * 50,
        verbose=True,
    )
    
    # Save demos
    task_ids = save_multi_mission_demos(env_id, mission_demos)
    
    return task_ids


# ═══════════════════════════════════════════════════════════════════════════
# BATCH COLLECTION
# ═══════════════════════════════════════════════════════════════════════════

def batch_collect(env_ids, demos_per_mission=None, skip_existing=True):
    """
    Collect demos for multiple environments.
    
    Args:
        env_ids: List of environment IDs
        demos_per_mission: Override demo counts, or None for auto-scaling
        skip_existing: Skip already collected environments
    """
    print(f"\n{'='*70}")
    print(f"BATCH COLLECTION: {len(env_ids)} ENVIRONMENTS")
    print(f"{'='*70}\n")
    
    # Show plan
    print("Collection plan:")
    for i, env_id in enumerate(env_ids, 1):
        complexity, default_demos = get_env_info(env_id)
        target = demos_per_mission if demos_per_mission else default_demos
        print(f"  {i}. {env_id:35s} [{complexity:8s}] {target:,} demos/variant")
    
    if demos_per_mission:
        print(f"\nUsing fixed demo count: {demos_per_mission:,} per variant")
    else:
        print(f"\nUsing auto-scaling based on complexity")
    
    print(f"Skip existing: {skip_existing}")
    
    # Confirm
    print(f"\n{'='*70}")
    response = input("Proceed with collection? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Collect each environment
    all_task_ids = []
    
    for i, env_id in enumerate(env_ids, 1):
        print(f"\n\n{'#'*70}")
        print(f"# [{i}/{len(env_ids)}] {env_id}")
        print(f"{'#'*70}")
        
        try:
            task_ids = collect_environment(
                env_id=env_id,
                demos_per_mission=demos_per_mission,
                skip_existing=skip_existing,
            )
            all_task_ids.extend(task_ids)
            
            print(f"\n✓ Completed {env_id}: {len(task_ids)} tasks created")
            
        except KeyboardInterrupt:
            print(f"\n\n⚠ Interrupted by user!")
            print(f"Collected {len(all_task_ids)} tasks so far.")
            response = input("Continue to next environment? [y/N]: ")
            if response.lower() != 'y':
                break
        except Exception as e:
            print(f"\n✗ Error collecting {env_id}: {e}")
            print(f"Continuing to next environment...\n")
    
    # Final summary
    print(f"\n\n{'='*70}")
    print(f"BATCH COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"\nTotal tasks created: {len(all_task_ids)}")
    print(f"Environments processed: {i}/{len(env_ids)}")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print(f"{'='*70}")
    print("1. Generate curriculum entries:")
    print("   python generate_curriculum_entries.py > new_curriculum.txt")
    print()
    print("2. Copy output to utils/il_config.py")
    print()
    print("3. Apply code patches (if not done yet):")
    print("   cp bot_collector_corrected.py trajectory/bot_collector.py")
    print("   # Then edit agents/il_agent.py and agents/curriculum.py")
    print()
    print("4. Train IL on expanded curriculum:")
    print("   python main_il.py --phase curriculum --device cuda")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Collect multi-mission demos with complexity-based scaling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single environment (auto-scaled: 10k demos for simple task)
  python test_simple_env.py BabyAI-GoToObj-v0
  
  # Multiple environments (auto-scaled by complexity)
  python test_simple_env.py BabyAI-GoToObj-v0 BabyAI-Open-v0 BabyAI-PickupLoc-v0
  
  # Override demo count for all environments
  python test_simple_env.py BabyAI-GoToObj-v0 BabyAI-PickupLoc-v0 --demos 15000
  
  # Don't skip existing
  python test_simple_env.py BabyAI-GoToObj-v0 --no-skip

Complexity Tiers:
  Simple (10k):   GoToObj, GoToLocal, Open
  Medium (20k):   PickupLoc, PutNextLocal
  Hard (30k):     GoToSeqS5R2, SynthS5R2
  Extreme (50k):  BossLevel
        """
    )
    
    parser.add_argument(
        'envs',
        nargs='+',
        help='One or more environment IDs (e.g., BabyAI-GoToObj-v0)'
    )
    
    parser.add_argument(
        '--demos',
        type=int,
        help='Override demos per mission variant (default: auto-scale by complexity)'
    )
    
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Re-collect even if demos already exist'
    )
    
    args = parser.parse_args()
    
    # Run collection
    if len(args.envs) == 1:
        # Single environment
        collect_environment(
            env_id=args.envs[0],
            demos_per_mission=args.demos,
            skip_existing=not args.no_skip,
        )
    else:
        # Batch collection
        batch_collect(
            env_ids=args.envs,
            demos_per_mission=args.demos,
            skip_existing=not args.no_skip,
        )