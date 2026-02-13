#!/usr/bin/env python3
"""
Analyze Collected Demos and Identify Trivial Tasks

This script helps you identify which tasks have demos that are too short/easy
and might not be worth including in the curriculum.

Usage:
    python analyze_demos.py
    
Outputs:
    - Summary of demo lengths per task
    - List of tasks that might be too trivial (all demos < 5 steps)
    - Suggested curriculum filtering
"""

import os
import pickle
from glob import glob
import numpy as np
from collections import defaultdict


def infer_env_id(task_id):
    """Infer full BabyAI env id from a task_id like 'GoToObj__go_to_the_box'."""
    env_short = task_id.split("__", 1)[0]
    return f"BabyAI-{env_short}-v0"


def infer_mission(task_id):
    """Infer mission text from task_id like 'GoToObj__go_to_the_blue_box'."""
    parts = task_id.split("__", 1)
    if len(parts) < 2:
        return "Unknown"
    return parts[1].replace("_", " ")


def analyze_demo_file(filepath):
    """Analyze a single demo file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    task_id = data.get('task_id') or os.path.splitext(os.path.basename(filepath))[0].replace("demos_", "", 1)
    env_id = data.get('env_id') or infer_env_id(task_id)
    demos = data.get('demos', [])
    missions = data.get('missions', [])
    
    if not demos:
        return None
    
    # Get episode lengths
    lengths = [len(demo) for demo in demos]
    
    return {
        'task_id': task_id,
        'env_id': env_id,
        'mission': missions[0] if missions else infer_mission(task_id),
        'n_demos': len(demos),
        'min_len': min(lengths),
        'max_len': max(lengths),
        'mean_len': np.mean(lengths),
        'median_len': np.median(lengths),
        'std_len': np.std(lengths),
    }


def analyze_all_demos(demos_dir='demos'):
    """Analyze all collected demos."""
    
    demo_files = glob(os.path.join(demos_dir, 'demos_*.pkl'))
    
    if not demo_files:
        print(f"No demo files found in {demos_dir}/")
        return
    
    print(f"\n{'='*80}")
    print(f"ANALYZING {len(demo_files)} DEMO FILES")
    print(f"{'='*80}\n")
    
    results = []
    
    for filepath in sorted(demo_files):
        try:
            result = analyze_demo_file(filepath)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
    
    # Group by criteria
    trivial = []      # All demos < 5 steps
    short = []        # Mean < 5 steps
    medium = []       # Mean 5-15 steps
    long = []         # Mean > 15 steps
    
    for r in results:
        if r['max_len'] < 5:
            trivial.append(r)
        elif r['mean_len'] < 5:
            short.append(r)
        elif r['mean_len'] < 15:
            medium.append(r)
        else:
            long.append(r)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY BY TASK DIFFICULTY")
    print("="*80)
    
    print(f"\n🟥 TRIVIAL (all demos < 5 steps): {len(trivial)} tasks")
    print("   These might not be worth training on - too easy!")
    if trivial:
        for r in sorted(trivial, key=lambda x: x['mean_len']):
            print(f"     {r['task_id']:50s} mean={r['mean_len']:.1f} [{r['min_len']}-{r['max_len']}]")
    
    print(f"\n🟨 SHORT (mean < 5 steps): {len(short)} tasks")
    print("   Simple tasks, but have some variation")
    if short and len(short) <= 10:
        for r in sorted(short, key=lambda x: x['mean_len']):
            print(f"     {r['task_id']:50s} mean={r['mean_len']:.1f} [{r['min_len']}-{r['max_len']}]")
    elif short:
        print(f"     ({len(short)} tasks - run with --verbose to see all)")
    
    print(f"\n🟩 MEDIUM (mean 5-15 steps): {len(medium)} tasks")
    print("   Good balance of complexity")
    if len(medium) <= 10:
        for r in sorted(medium, key=lambda x: x['mean_len']):
            print(f"     {r['task_id']:50s} mean={r['mean_len']:.1f} [{r['min_len']}-{r['max_len']}]")
    else:
        print(f"     ({len(medium)} tasks - run with --verbose to see all)")
    
    print(f"\n🟦 LONG (mean > 15 steps): {len(long)} tasks")
    print("   Complex, multi-step tasks")
    if long:
        for r in sorted(long, key=lambda x: -x['mean_len']):
            print(f"     {r['task_id']:50s} mean={r['mean_len']:.1f} [{r['min_len']}-{r['max_len']}]")
    
    # Statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    
    all_means = [r['mean_len'] for r in results]
    print(f"Total tasks:        {len(results)}")
    print(f"Mean episode length: {np.mean(all_means):.1f} steps")
    print(f"Median:             {np.median(all_means):.1f} steps")
    print(f"Range:              {min(all_means):.1f} - {max(all_means):.1f} steps")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    if trivial:
        print(f"⚠ Found {len(trivial)} trivial tasks (all demos < 5 steps)")
        print(f"  Consider EXCLUDING these from curriculum - they're too easy!")
        print(f"  They won't help CLIP learn meaningful task structure.")
        print()
    
    if len(short) > len(medium) + len(long):
        print(f"⚠ Most tasks ({len(short)}/{len(results)}) are short (< 5 steps)")
        print(f"  Consider:")
        print(f"    1. Collecting demos from harder environments")
        print(f"    2. Using RL fine-tuning to extend behaviors")
        print(f"    3. Filtering out trivial tasks")
        print()
    
    recommended = medium + long
    print(f"✓ Recommended curriculum: {len(recommended)} tasks")
    print(f"  (Excluding {len(trivial)} trivial + {len(short)} short tasks)")
    
    # Generate filtered curriculum
    print(f"\n{'='*80}")
    print("FILTERED CURRICULUM (copy to utils/il_config.py)")
    print(f"{'='*80}\n")
    
    print("CURRICULUM_STAGES_FILTERED = [")
    print("    # Stage 0 - Medium & Long tasks only")
    print("    [")
    
    for r in sorted(recommended, key=lambda x: (x['env_id'], x['mission'])):
        mission = r['mission']
        env_id = r['env_id']
        print(f'        ("{env_id}", "{mission}"),  # mean={r["mean_len"]:.1f} steps')
    
    print("    ],")
    print("]")
    print()
    print(f"# Original: {len(results)} tasks")
    print(f"# Filtered: {len(recommended)} tasks (removed {len(trivial) + len(short)} trivial/short)")
    
    return {
        'trivial': trivial,
        'short': short,
        'medium': medium,
        'long': long,
        'all': results,
    }


if __name__ == "__main__":
    import sys
    
    verbose = '--verbose' in sys.argv
    
    results = analyze_all_demos()
    
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print()
    print("1. Review the trivial tasks above")
    print("2. Decide if you want to filter them out")
    print("3. If yes: Copy the FILTERED CURRICULUM to utils/il_config.py")
    print("4. If no: Continue with all tasks (they won't hurt, just add noise)")
    print()
    print("For CLIP-RL, having diverse task complexity is actually good!")
    print("Even simple tasks help CLIP learn the overall task space.")
    print()
