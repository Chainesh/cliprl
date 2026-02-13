#!/usr/bin/env python3
"""
Generate Curriculum Entries from Collected Multi-Mission Demos

After collecting multi-mission demos, run this script to automatically
generate the curriculum entries for utils/il_config.py

Usage:
    python generate_curriculum_entries.py > new_curriculum.txt
    # Then copy/paste the output into utils/il_config.py
"""

import os
import pickle
from glob import glob
from collections import defaultdict


def infer_env_id(task_id: str) -> str:
    """Infer full BabyAI env id from a task_id like 'GoToObj__go_to_the_box'."""
    env_short = task_id.split("__", 1)[0]
    return f"BabyAI-{env_short}-v0"

def extract_curriculum_entries(demos_dir="demos"):
    """
    Scan demos directory and generate curriculum entries.
    Groups by environment ID.
    """
    
    # Find all demo files
    demo_files = glob(os.path.join(demos_dir, "demos_*.pkl"))
    
    if not demo_files:
        print(f"# No demo files found in {demos_dir}/")
        return
    
    # Group by environment
    env_tasks = defaultdict(list)
    
    for demo_file in sorted(demo_files):
        try:
            with open(demo_file, 'rb') as f:
                data = pickle.load(f)

            task_id = data.get('task_id') \
                or os.path.splitext(os.path.basename(demo_file))[0].replace("demos_", "", 1)
            env_id = data.get('env_id') or infer_env_id(task_id)
            missions = data.get('missions', [])
            
            if missions:
                mission = missions[0]  # Get the mission for this task
                
                env_tasks[env_id].append({
                    'mission': mission,
                    'task_id': task_id,
                    'n_demos': len(data.get('demos', [])),
                })
        except Exception as e:
            print(f"# Error reading {demo_file}: {e}")
    
    # Print curriculum entries grouped by environment
    print("# Auto-generated curriculum entries from collected demos")
    print("# Copy this into utils/il_config.py\n")
    
    print("CURRICULUM_STAGES_EXPANDED = [")
    print("    # Stage 0 - Navigation")
    print("    [")
    
    # Print tasks grouped by environment
    for env_id in sorted(env_tasks.keys()):
        tasks = env_tasks[env_id]
        env_short = env_id.replace("BabyAI-", "").replace("-v0", "")
        
        print(f"\n        # {env_short} ({len(tasks)} variants)")
        
        for task in sorted(tasks, key=lambda x: x['mission']):
            mission = task['mission']
            n_demos = task['n_demos']
            print(f'        ("{env_id}", "{mission}"),  # {n_demos} demos')
    
    print("\n    ],")
    print("    # TODO: Add Stage 1, 2, etc.")
    print("]")
    print()
    print(f"# Total tasks: {sum(len(tasks) for tasks in env_tasks.values())}")
    print(f"# Environments: {len(env_tasks)}")


def print_summary(demos_dir="demos"):
    """Print summary statistics about collected demos."""
    
    demo_files = glob(os.path.join(demos_dir, "demos_*.pkl"))
    
    total_demos = 0
    total_tasks = 0
    env_counts = defaultdict(int)
    
    print("\n" + "="*70)
    print("COLLECTED DEMOS SUMMARY")
    print("="*70)
    
    for demo_file in demo_files:
        try:
            with open(demo_file, 'rb') as f:
                data = pickle.load(f)
            
            task_id = data.get('task_id') \
                or os.path.splitext(os.path.basename(demo_file))[0].replace("demos_", "", 1)
            env_id = data.get('env_id') or infer_env_id(task_id)
            n_demos = len(data.get('demos', []))
            
            total_demos += n_demos
            total_tasks += 1
            env_counts[env_id] += 1
            
        except Exception:
            pass
    
    print(f"\nTotal Tasks:        {total_tasks}")
    print(f"Total Demos:        {total_demos:,}")
    print(f"Unique Environments: {len(env_counts)}")
    
    print(f"\nTasks per Environment:")
    for env_id, count in sorted(env_counts.items(), key=lambda x: -x[1]):
        env_short = env_id.replace("BabyAI-", "").replace("-v0", "")
        print(f"  {env_short:30s} {count:3d} tasks")
    
    print("="*70 + "\n")


def verify_missions(demos_dir="demos"):
    """Verify that all collected demos have consistent missions."""
    
    print("\n" + "="*70)
    print("MISSION VERIFICATION")
    print("="*70)
    
    demo_files = glob(os.path.join(demos_dir, "demos_*.pkl"))
    
    issues = []
    
    for demo_file in demo_files:
        try:
            with open(demo_file, 'rb') as f:
                data = pickle.load(f)
            
            task_id = data.get('task_id', os.path.basename(demo_file))
            missions = data.get('missions', [])
            
            if not missions:
                issues.append(f"  ⚠ {task_id}: No missions stored")
            else:
                unique_missions = set(missions)
                if len(unique_missions) > 1:
                    issues.append(f"  ⚠ {task_id}: Variable missions ({len(unique_missions)} variants)")
                else:
                    print(f"  ✓ {task_id}")
        
        except Exception as e:
            issues.append(f"  ✗ {demo_file}: Error - {e}")
    
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(issue)
    else:
        print("\n✓ All tasks have consistent missions!")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    # Print summary
    print_summary()
    
    # Verify missions
    verify_missions()
    
    # Generate curriculum
    extract_curriculum_entries()
    
    print("\n# Next steps:")
    print("# 1. Review the generated curriculum above")
    print("# 2. Copy CURRICULUM_STAGES_EXPANDED to utils/il_config.py")
    print("# 3. Organize tasks into appropriate stages (0-5)")
    print("# 4. Run: python main_il.py --phase curriculum --device cuda")
