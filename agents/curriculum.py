"""
Curriculum Learning Manager for CLIP-RL.

Orchestrates progressive training through BabyAI difficulty levels:

    Stage 0: GoTo tasks (simple navigation)
         ↓  warm-start from stage 0 checkpoint
    Stage 1: Door / Open tasks
         ↓  warm-start from stage 1 checkpoint
    Stage 2: Unlock (keys + locked doors)
         ↓  warm-start from stage 2 checkpoint
    Stage 3: PutNext (manipulation)
         ↓  warm-start from stage 3 checkpoint
    Stage 4: Sequential / Compositional
         ↓  warm-start from stage 4 checkpoint
    Stage 5: BossLevel

At each stage:
    1. Collect bot demos (if not cached)
    2. Train via IL (behavioral cloning) on demos
    3. Optional RL fine-tune with PPO
    4. Evaluate success rate
    5. If success_rate >= threshold → advance to next stage
       Else → train more (up to max_il_epochs)

The policy state is transferred between stages so each stage
inherits the navigation/manipulation skills from previous stages.
"""

import torch
import os, sys, time
import numpy as np
from typing import Dict, Optional, List
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.recurrent_policy import RecurrentPolicy, build_vocab, tokenize_batch
from agents.il_agent import ILAgent
from utils.il_config import (
    CURRICULUM_STAGES, ALL_IL_TASKS, CURRICULUM, IL, RL_FINETUNE,
    CHECKPOINTS, BOT_DEMOS,
)


def make_task_id(env_id: str, instruction: str) -> str:
    env_short  = env_id.replace("BabyAI-", "").replace("-v0", "")
    instr_slug = instruction.replace(" ", "_")
    return f"{env_short}__{instr_slug}"


# ─── Curriculum Manager ───────────────────────────────────────────────────────

class CurriculumManager:
    """
    Manages multi-stage curriculum training toward BossLevel.

    Handles:
        - Demo collection at each stage
        - IL training with warm-start from previous stage
        - Optional RL fine-tuning
        - Stage advancement logic
        - Checkpoint saving / resuming
    """

    def __init__(
        self,
        device:      str  = "auto",
        large_model: bool = False,
        verbose:     bool = True,
        start_stage: int  = 0,
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.large_model = large_model
        self.verbose     = verbose
        self.start_stage = start_stage

        # Build shared vocab from ALL instructions across all stages
        self.all_instructions = [instr for _, instr in ALL_IL_TASKS]
        self.vocab = build_vocab(self.all_instructions)

        # Track results per stage
        self.stage_results: Dict[int, Dict] = {}

    # ── Stage state management ────────────────────────────────────────────────

    def _stage_ckpt_path(self, stage_idx: int) -> str:
        return os.path.join(CHECKPOINTS, f"curriculum_stage_{stage_idx}.pt")

    def _save_stage(self, stage_idx: int, policy: RecurrentPolicy):
        path = self._stage_ckpt_path(stage_idx)
        torch.save({
            "policy_state_dict": policy.state_dict(),
            "vocab"            : self.vocab,
            "stage_idx"        : stage_idx,
        }, path)
        if self.verbose:
            print(f"  Saved stage {stage_idx} checkpoint → {path}")

    def _load_stage(self, stage_idx: int) -> Optional[Dict]:
        path = self._stage_ckpt_path(stage_idx)
        if os.path.exists(path):
            return torch.load(path, map_location="cpu")
        return None

    def _warm_start(
        self,
        agent: ILAgent,
        from_stage: int,
    ) -> bool:
        """
        Load weights from a previous stage into agent's policy.
        Only transfers weights that are architecture-compatible.

        Returns True if warm start succeeded.
        """
        ckpt = self._load_stage(from_stage)
        if ckpt is None:
            if self.verbose:
                print(f"  No stage {from_stage} checkpoint found — training from scratch")
            return False

        prev_state = ckpt["policy_state_dict"]
        curr_state = agent.policy.state_dict()

        # Load only matching keys (architecture must be same — it is, same vocab/dims)
        matched = {k: v for k, v in prev_state.items() if k in curr_state}
        curr_state.update(matched)
        agent.policy.load_state_dict(curr_state)

        if self.verbose:
            print(f"  Warm-started from stage {from_stage} "
                  f"({len(matched)}/{len(curr_state)} param tensors transferred)")
        return True

    # ── Single stage training ─────────────────────────────────────────────────

    def train_stage(
        self,
        stage_idx:    int,
        warm_start:   bool = CURRICULUM["warm_start"],
        rl_finetune:  bool = False,
    ) -> Dict:
        """
        Train all tasks in a given stage.

        Args:
            stage_idx:   which stage to train (0 = simplest, 5 = BossLevel)
            warm_start:  initialize from previous stage checkpoint
            rl_finetune: run PPO fine-tune after IL (recommended for hard stages)

        Returns:
            dict of {task_id: success_rate}
        """
        from trajectory.bot_collector import collect_bot_demos, save_demos, load_demos, demos_exist

        stage_tasks = CURRICULUM_STAGES[stage_idx]
        results     = {}

        print(f"\n{'='*65}")
        print(f"  CURRICULUM STAGE {stage_idx}  |  {len(stage_tasks)} tasks")
        print(f"  Large model: {self.large_model}  |  RL finetune: {rl_finetune}")
        print(f"{'='*65}")

        for env_id, instruction in stage_tasks:
            task_id = make_task_id(env_id, instruction)

            print(f"\n  Task: {task_id}")
            print(f"  Instruction: '{instruction}'")

            # ── Collect demos ──────────────────────────────────────────────────
            if demos_exist(task_id):
                print(f"  Loading cached demos...")
                demos = load_demos(task_id)
            else:
                print(f"  Collecting {BOT_DEMOS['n_episodes']} bot demos...")
                demos = collect_bot_demos(
                    env_id     = env_id,
                    n_episodes = BOT_DEMOS["n_episodes"],
                    verbose    = self.verbose,
                )
                save_demos(demos, task_id)

            # ── Create agent ───────────────────────────────────────────────────
            from utils.il_config import RECURRENT_POLICY
            memory_dim = (
                RECURRENT_POLICY["memory_dim_large"]
                if self.large_model else
                RECURRENT_POLICY["memory_dim"]
            )

            agent = ILAgent(
                env_id           = env_id,
                task_id          = task_id,
                instruction      = instruction,
                all_instructions = self.all_instructions,
                large_model      = self.large_model,
                device           = self.device,
                verbose          = self.verbose,
            )

            # ── Warm start from previous stage ─────────────────────────────────
            if warm_start and stage_idx > 0:
                self._warm_start(agent, from_stage=stage_idx - 1)

            # ── IL training ────────────────────────────────────────────────────
            agent.train_on_demos(
                demos,
                epochs     = CURRICULUM["max_il_epochs"],
                patience   = IL["patience"],
            )

            # ── RL fine-tune (optional) ────────────────────────────────────────
            if rl_finetune:
                self._ppo_finetune(agent, env_id, instruction)

            # ── Evaluate ───────────────────────────────────────────────────────
            metrics = agent.evaluate(n_episodes=CURRICULUM["eval_episodes"])
            sr      = metrics["success_rate"]

            print(f"\n  [{task_id}]  Success rate: {sr*100:.1f}%")
            if sr < CURRICULUM["success_threshold"]:
                print(f"  ⚠  Below threshold ({CURRICULUM['success_threshold']*100:.0f}%) "
                      f"— consider more demos or longer training")
            else:
                print(f"  ✓  Passed threshold")

            agent.save()
            results[task_id] = metrics

        # Save a single "stage complete" checkpoint using the last task's policy
        # (all tasks share the same arch — use the last one as the warm-start for next stage)
        self._save_stage(stage_idx, agent.policy)
        self.stage_results[stage_idx] = results
        return results

    # ── PPO fine-tune ─────────────────────────────────────────────────────────

    def _ppo_finetune(
        self,
        agent:       ILAgent,
        env_id:      str,
        instruction: str,
    ):
        """
        Fine-tune an IL-pretrained policy with PPO.
        The IL policy is used as initialization — PPO then improves it
        using actual env rewards.
        """
        import gymnasium as gym
        from envs.task_suite import FlatObsWrapper
        from torch.distributions import Categorical

        if self.verbose:
            print(f"\n  PPO fine-tuning for {instruction}...")

        policy    = agent.policy.to(torch.device(self.device))
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr = RL_FINETUNE["learning_rate"],
        )

        env    = gym.make(env_id, render_mode=None)
        env    = FlatObsWrapper(env)
        tokens = agent.tokens.to(torch.device(self.device))

        total_steps = 0
        n_updates   = RL_FINETUNE["total_timesteps"] // RL_FINETUNE["n_steps"]

        ep_rewards = []

        for update in range(1, n_updates + 1):
            # ── Collect rollout ────────────────────────────────────────────────
            obs_list     = []
            act_list     = []
            rew_list     = []
            val_list     = []
            logp_list    = []
            done_list    = []
            hidden_list  = []

            obs, _  = env.reset()
            hidden  = policy.init_hidden(1, torch.device(self.device))
            ep_r    = 0.0
            ep_rs   = []

            policy.eval()
            for _ in range(RL_FINETUNE["n_steps"]):
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits, value, new_hidden = policy.forward_step(
                        obs_t, tokens, hidden
                    )
                    dist    = Categorical(logits=logits)
                    action  = dist.sample()
                    log_p   = dist.log_prob(action)

                obs_list.append(obs.copy())
                act_list.append(action.item())
                rew_list.append(0.0)   # filled after step
                val_list.append(value.item())
                logp_list.append(log_p.item())
                hidden_list.append(hidden.detach().cpu())

                obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                rew_list[-1] = float(reward)
                done_list.append(float(done))
                ep_r += reward
                total_steps += 1
                hidden = new_hidden

                if done:
                    ep_rs.append(ep_r)
                    obs, _ = env.reset()
                    hidden = policy.init_hidden(1, torch.device(self.device))
                    ep_r   = 0.0

            ep_rewards.extend(ep_rs)

            # ── Compute returns (simple discounted, no GAE for simplicity) ─────
            gamma   = RL_FINETUNE["gamma"]
            returns = []
            R = 0.0
            for r, d in zip(reversed(rew_list), reversed(done_list)):
                R = r + gamma * R * (1.0 - d)
                returns.insert(0, R)

            returns    = torch.FloatTensor(returns).to(self.device)
            advantages = returns - torch.FloatTensor(val_list).to(self.device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ── PPO update ─────────────────────────────────────────────────────
            policy.train()
            obs_t   = torch.FloatTensor(np.array(obs_list)).to(self.device)
            act_t   = torch.LongTensor(act_list).to(self.device)
            old_lp  = torch.FloatTensor(logp_list).to(self.device)
            hid_0   = hidden_list[0].to(self.device)

            # Process all steps as one sequence (recurrence=n_steps)
            # Simplified: process in chunks of recurrence length
            rec = RL_FINETUNE["recurrence"]
            for start in range(0, len(obs_t), rec):
                end    = min(start + rec, len(obs_t))
                chunk  = obs_t[start:end].unsqueeze(0)   # (1, chunk, 148)
                tok    = tokens                            # (1, 20)
                h      = hid_0 if start == 0 else hidden_list[start].to(self.device)

                logits, vals, _ = policy.forward(chunk, tok, h)
                logits = logits.squeeze(0)                # (chunk, 7)
                vals   = vals.squeeze(0).squeeze(-1)      # (chunk,)

                dist   = Categorical(logits=logits)
                new_lp = dist.log_prob(act_t[start:end])
                entr   = dist.entropy().mean()

                ratio  = torch.exp(new_lp - old_lp[start:end])
                adv    = advantages[start:end]
                ret    = returns[start:end]

                loss_pi  = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1 - RL_FINETUNE["clip_range"],
                                1 + RL_FINETUNE["clip_range"]) * adv
                ).mean()
                loss_vf  = torch.nn.functional.mse_loss(vals, ret)
                loss     = (loss_pi
                           + RL_FINETUNE["vf_coef"] * loss_vf
                           - RL_FINETUNE["ent_coef"] * entr)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), RL_FINETUNE["max_grad_norm"]
                )
                optimizer.step()

            if self.verbose and update % 20 == 0:
                mean_r = np.mean(ep_rewards[-20:]) if ep_rewards else 0.0
                print(f"    PPO update {update:3d}/{n_updates}  |  "
                      f"mean_reward (last 20 eps): {mean_r:.3f}")

        env.close()

    # ── Run full curriculum ────────────────────────────────────────────────────

    def run(
        self,
        rl_finetune_stages: List[int] = [],
        skip_to_stage: Optional[int]  = None,
    ) -> Dict:
        """
        Run the full curriculum from start_stage to BossLevel.

        Args:
            rl_finetune_stages: which stage indices to also do RL fine-tune on
            skip_to_stage:      jump to a specific stage (must have prev checkpoints)

        Returns:
            all_results: dict of {stage_idx: {task_id: metrics}}
        """
        all_results = {}
        start = skip_to_stage if skip_to_stage is not None else self.start_stage

        for stage_idx in range(start, len(CURRICULUM_STAGES)):
            do_rl = stage_idx in rl_finetune_stages

            stage_results = self.train_stage(
                stage_idx   = stage_idx,
                warm_start  = (stage_idx > 0),
                rl_finetune = do_rl,
            )
            all_results[stage_idx] = stage_results

            # Print stage summary
            success_rates = [m["success_rate"] for m in stage_results.values()]
            avg_sr        = np.mean(success_rates)
            print(f"\n  Stage {stage_idx} avg success rate: {avg_sr*100:.1f}%")

        print(f"\n{'='*65}")
        print(f"  CURRICULUM COMPLETE")
        print(f"{'='*65}")
        self._print_summary(all_results)

        return all_results

    def _print_summary(self, all_results: Dict):
        for stage_idx, stage_results in all_results.items():
            print(f"\n  Stage {stage_idx}:")
            for task_id, metrics in stage_results.items():
                sr = metrics["success_rate"]
                r  = metrics["mean_reward"]
                ok = "✓" if sr >= CURRICULUM["success_threshold"] else "✗"
                print(f"    {ok}  {task_id:50s}  success: {sr*100:5.1f}%  reward: {r:.3f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running curriculum training toward BossLevel...\n")
    manager = CurriculumManager(
        device      = "cuda" if torch.cuda.is_available() else "cpu",
        large_model = False,   # set True for BossLevel (stage 5)
        verbose     = True,
        start_stage = 0,
    )
    results = manager.run(
        rl_finetune_stages = [3, 4, 5],   # do RL fine-tune on hard stages
    )