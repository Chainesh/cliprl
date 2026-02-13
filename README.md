# Complete Guide: From Demo Collection to CLIP Training

## Overview

This guide covers everything from collecting multi-mission demos to training CLIP on the expanded dataset.

## Part 1: Code Changes Required ⚠️ DO THIS FIRST

### Files to Update:

1. **trajectory/bot_collector.py** ✅ Already done
   - Replaced with `bot_collector_corrected.py`
   - Now extracts and returns actual mission strings

2. **agents/il_agent.py** ⚠️ MUST UPDATE
   - See `PATCH_il_agent.py` for exact changes
   - Changes: Handle 6 return values from `demos_to_sequences()`

3. **agents/curriculum.py** ⚠️ MUST UPDATE
   - See `PATCH_curriculum.py` for exact changes
   - Changes: Handle tuple returns, use actual missions

4. **utils/il_config.py** ⚠️ UPDATE AFTER COLLECTION
   - Add all discovered mission variants
   - Use `generate_curriculum_entries.py` to help

---

## Part 2: Quick Test (5 minutes)

### Step 1: Apply Code Changes
```bash
cd ~/Desktop/cliprl

# Copy corrected bot_collector
cp bot_collector_corrected.py trajectory/bot_collector.py

# Apply patches to il_agent.py and curriculum.py
# (See PATCH_il_agent.py and PATCH_curriculum.py for exact changes)
```

### Step 2: Test Collection on Simplest Environment
```bash
# Copy test script
cp test_simple_env.py .

# Run on simplest variable-mission environment
uv run test_simple_env.py BabyAI-GoToObj-v0

# When prompted:
# Episodes per variant: 100 (small for testing)

# Expected: ~12 new task files in demos/
```

### Step 3: Verify It Worked
```bash
# Check created files
ls demos/ | grep GoToObj

# Should see multiple files like:
# demos_GoToObj__go_to_the_blue_ball.pkl
# demos_GoToObj__go_to_the_blue_box.pkl
# etc.

# Generate curriculum entries
python generate_curriculum_entries.py
```

---

## Part 3: Full Collection (Hours)

### Recommended Order (easiest → hardest):

1. **BabyAI-GoToObj-v0** ⭐ Start here
   - ~12 variants
   - Minimal rejections
   - Fast episodes

2. **BabyAI-Open-v0**
   - ~8 door variants
   - Some rejections

3. **BabyAI-PickupLoc-v0**
   - ~10-15 pickup variants
   - More rejections

4. **BabyAI-PutNextLocal-v0**
   - ~15-20 placement variants
   - Many rejections

5. **BabyAI-GoToSeqS5R2-v0**
   - ~20+ sequential variants
   - Many rejections

6. **BabyAI-SynthS5R2-v0**
   - ~20+ composition variants
   - Many rejections

7. **BabyAI-BossLevel-v0** 🔥 Hardest
   - Limit to 30 most common missions
   - Extreme rejections

### Collection Commands:

```bash
# Collect 1000 demos per variant for each environment
uv run test_simple_env.py BabyAI-GoToObj-v0
uv run test_simple_env.py BabyAI-Open-v0
uv run test_simple_env.py BabyAI-PickupLoc-v0
# etc.

# For BossLevel, use special handling:
# - Collect diverse sample
# - Or limit to top 30 missions
```

---

## Part 4: Update Curriculum

### Generate Curriculum Entries:

```bash
# After collecting demos, generate curriculum
python generate_curriculum_entries.py > new_curriculum.txt

# Review the output
cat new_curriculum.txt
```

### Update utils/il_config.py:

Replace old curriculum with expanded version:

```python
# OLD (17 tasks total)
CURRICULUM_STAGES = [
    # Stage 0
    [
        ("BabyAI-GoToRedBall-v0", "go to the red ball"),
        ("BabyAI-GoToObj-v0", "go to the blue box"),  # Just 1 variant
        # ...
    ],
]

# NEW (100+ tasks total)
CURRICULUM_STAGES = [
    # Stage 0 - Navigation (EXPANDED)
    [
        # Fixed-mission tasks
        ("BabyAI-GoToRedBall-v0", "go to the red ball"),
        
        # ALL GoToObj variants (12 tasks from 1 environment!)
        ("BabyAI-GoToObj-v0", "go to the blue ball"),
        ("BabyAI-GoToObj-v0", "go to the blue box"),
        ("BabyAI-GoToObj-v0", "go to the blue door"),
        ("BabyAI-GoToObj-v0", "go to the blue key"),
        ("BabyAI-GoToObj-v0", "go to the green ball"),
        ("BabyAI-GoToObj-v0", "go to the green box"),
        ("BabyAI-GoToObj-v0", "go to the green door"),
        ("BabyAI-GoToObj-v0", "go to the green key"),
        ("BabyAI-GoToObj-v0", "go to the red ball"),
        ("BabyAI-GoToObj-v0", "go to the red box"),
        ("BabyAI-GoToObj-v0", "go to the red door"),
        ("BabyAI-GoToObj-v0", "go to the red key"),
        
        # ... repeat for other expanded environments
    ],
    # Stage 1, 2, etc.
]
```

---

## Part 5: Train IL on Expanded Curriculum

```bash
# Train IL policies on ALL expanded tasks
python main_il.py --phase curriculum --device cuda

# This will take MUCH longer than before (100+ tasks vs 17)
# But you'll have way more base tasks for CLIP!

# Expected output:
# Stage 0: Training 35 tasks (was 4)
# Stage 1: Training 25 tasks (was 4)
# etc.
```

---

## Part 6: Collect Trajectories

```bash
# After IL training completes, collect trajectories
python main_il.py --phase il_trajectories --device cuda

# Collects trajectories from ALL 100+ trained policies
# Each policy contributes to CLIP training
```

---

## Part 7: Train CLIP on Expanded Dataset

```bash
# CLIP now trains on 100+ (instruction, policy) pairs!
python main_il.py --phase clip --epochs 100 --device cuda

# CLIP learns:
# - "go to red ball" and "go to blue box" have similar policies
# - Task structure matters more than specific objects
# - Color/object are surface variations
```

---

## Part 8: Evaluate Transfer

```bash
# Test transfer to new tasks
python main_il.py --phase transfer --seeds 3 --device cuda

# Expected improvement:
# - Better zero-shot transfer to new color/object combos
# - More robust to instruction variations
# - Higher success rates on target tasks
```

---

## Expected Results

### Before Multi-Mission Expansion:
- **Base tasks:** 17
- **CLIP training pairs:** 17
- **Transfer:** Limited to exact instruction matches

### After Multi-Mission Expansion:
- **Base tasks:** 100-150
- **CLIP training pairs:** 100-150
- **Transfer:** Generalizes across color/object variations

### Example Transfer Improvement:

**Target task:** "go to the purple key"

**Before:** 
- No similar base task
- Random initialization
- Transfer success: ~30%

**After:**
- Has "go to red key", "go to blue key", "go to green key" in base tasks
- CLIP learns color is surface variation
- Transfer success: ~75%

---

## Files Reference

### Code Changes:
- `bot_collector_corrected.py` - Fixed collector (already applied)
- `PATCH_il_agent.py` - Changes for il_agent.py
- `PATCH_curriculum.py` - Changes for curriculum.py

### Testing:
- `test_simple_env.py` - Test collection on one environment
- `TESTING_STRATEGY.md` - Explains rejections and testing order
- `QUICK_START.md` - 5-minute quick test

### Helpers:
- `generate_curriculum_entries.py` - Auto-generate curriculum entries
- `verify_missions.py` - Check which envs have variable missions

### Documentation:
- `NEXT_STEPS_AND_CHANGES.md` - Complete workflow (this file)
- `MULTI_MISSION_STRATEGY.md` - Strategy explanation
- `FINAL_SOLUTION.md` - High-level overview

---

## Troubleshooting

### Issue: "No sequences could be extracted"
- **Cause:** All demos too short for seq_len
- **Fix:** Corrected bot_collector auto-adjusts seq_len

### Issue: "Mission mismatch during training"
- **Cause:** Using hardcoded missions vs actual
- **Fix:** Applied patches use actual missions from demos

### Issue: "Variable missions detected"
- **Cause:** Environment generates random missions
- **Fix:** This is GOOD! Each variant becomes a task

### Issue: Too many "Sampling rejected" messages
- **Cause:** Complex environment constraints
- **Fix:** Normal! Just wait, it will succeed

---

## Summary Checklist

### Setup (Do Once):
- [ ] Replace `trajectory/bot_collector.py` with corrected version
- [ ] Apply patches to `agents/il_agent.py`
- [ ] Apply patches to `agents/curriculum.py`
- [ ] Test on one simple environment (5 min)

### Collection (Hours):
- [ ] Collect GoToObj-v0 (100 demos/variant for test)
- [ ] Verify it creates multiple task files
- [ ] Collect all simple environments (1000 demos/variant)
- [ ] Collect harder environments (optional)

### Curriculum (30 min):
- [ ] Run `generate_curriculum_entries.py`
- [ ] Copy output to `utils/il_config.py`
- [ ] Organize tasks into stages

### Training (Days):
- [ ] Train IL on expanded curriculum
- [ ] Collect trajectories from all policies
- [ ] Train CLIP on expanded dataset
- [ ] Evaluate transfer

---

## Next Steps After This Guide

Once you have CLIP trained on 100+ tasks:

1. **Analyze learned representations**
   - Visualize task embeddings
   - Check if similar tasks cluster together

2. **Test on diverse target tasks**
   - New color/object combinations
   - Multi-step compositions
   - Evaluate generalization

3. **Iterate and improve**
   - Collect more mission variants if needed
   - Add more diverse base tasks
   - Fine-tune CLIP training hyperparameters

---

## Quick Command Reference

```bash
# 1. Initial setup and test
cp bot_collector_corrected.py trajectory/bot_collector.py
cp test_simple_env.py .
uv run test_simple_env.py BabyAI-GoToObj-v0

# 2. Generate curriculum after collection
python generate_curriculum_entries.py > new_curriculum.txt

# 3. Train on expanded curriculum
python main_il.py --phase curriculum --device cuda

# 4. CLIP training
python main_il.py --phase il_trajectories --device cuda
python main_il.py --phase clip --epochs 100 --device cuda

# 5. Evaluate
python main_il.py --phase transfer --seeds 3 --device cuda
```

---

## Expected Timeline

- **Setup & Testing:** 30 minutes
- **Collection (6 environments, 1000/variant):** 6-12 hours
- **IL Training (100 tasks):** 1-3 days
- **Trajectory Collection:** 2-6 hours
- **CLIP Training:** 4-8 hours
- **Transfer Evaluation:** 2-4 hours

**Total:** ~2-5 days for complete pipeline

But you get 5-10× better CLIP training data!

---

## Questions?

If you run into issues:

1. Check the relevant PATCH file for exact code changes
2. Verify files with `python -m py_compile filename.py`
3. Run `generate_curriculum_entries.py` to see what was collected
4. Test on one simple environment first before scaling up

The most important thing is getting the code changes right FIRST, then collection and training will work smoothly!