"""
IL (Imitation Learning) config for CLIP-RL.
Drop this alongside utils/config.py — it extends the base config with
everything needed for bot demo collection, behavioral cloning, and
curriculum learning on harder BabyAI levels.
"""

import os
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
RESULTS     = os.path.join(ROOT, "results")
DEMOS_DIR   = os.path.join(ROOT, "demos")

os.makedirs(DEMOS_DIR,   exist_ok=True)
os.makedirs(CHECKPOINTS, exist_ok=True)


# ─── Curriculum: task progression toward BossLevel ───────────────────────────
# Each stage unlocks after the policy reaches SUCCESS_THRESHOLD on it.
# Policies are warm-started from the previous stage checkpoint.
# Format: (env_id, instruction, stage_idx)

CURRICULUM_STAGES = [
    # Stage 0 - Navigation
    [
        ("BabyAI-GoToObj-v0", "go to the blue ball"),   # mean=5.0 steps
        ("BabyAI-GoToObj-v0", "go to the blue key"),    # mean=5.0 steps
        ("BabyAI-GoToObj-v0", "go to the grey key"),    # mean=5.2 steps
        ("BabyAI-GoToObj-v0", "go to the purple ball"), # mean=6.0 steps
        ("BabyAI-GoToObj-v0", "go to the purple box"),  # mean=8.0 steps
        ("BabyAI-GoToObj-v0", "go to the purple key"),  # mean=7.5 steps
        ("BabyAI-GoToObj-v0", "go to the red box"),     # mean=6.0 steps
        ("BabyAI-GoToObj-v0", "go to the yellow box"),  # mean=8.0 steps
    ],
    # Stage 1 - Easy door interactions
    [
        ("BabyAI-Open-v0", "open a red door"),                # mean=9.0 steps
        ("BabyAI-Open-v0", "open the red door"),              # mean=7.3 steps
        ("BabyAI-OpenDoor-v0", "open the door on your left"), # mean=7.4 steps
    ],
    # Stage 2 - Medium door interactions
    [
        ("BabyAI-Open-v0", "open a green door"),   # mean=23.3 steps
        ("BabyAI-Open-v0", "open a grey door"),    # mean=18.6 steps
        ("BabyAI-Open-v0", "open a purple door"),  # mean=24.0 steps
        ("BabyAI-Open-v0", "open a yellow door"),  # mean=27.5 steps
        ("BabyAI-Open-v0", "open the door"),       # mean=30.5 steps
        ("BabyAI-Open-v0", "open the purple door"),# mean=26.0 steps
        ("BabyAI-Open-v0", "open a blue door"),    # mean=31.2 steps
    ],
    # Stage 3 - Hard door interactions
    [
        ("BabyAI-Open-v0", "open the blue door"),  # mean=65.0 steps
        ("BabyAI-Open-v0", "open the green door"), # mean=65.7 steps
        ("BabyAI-Open-v0", "open the grey door"),  # mean=78.0 steps
    ],
]

# Flat list of all IL tasks (all stages combined) for easy iteration
ALL_IL_TASKS = [task for stage in CURRICULUM_STAGES for task in stage]


# ─── Bot demo collection ──────────────────────────────────────────────────────

BOT_DEMOS = dict(
    n_episodes     = 10_000,   # fallback demos per task
    n_episodes_by_stage = {
        0: 5_000,
        1: 10_000,
        2: 15_000,
        3: 20_000,
    },
    max_steps      = 256,      # max steps per episode
    save_every     = 1_000,    # checkpoint demos every N episodes
    timeout        = 30,       # seconds before giving up on one episode
)


def get_stage_demo_count(stage_idx: int) -> int:
    """Return demos per task for a stage, with fallback to global default."""
    return BOT_DEMOS["n_episodes_by_stage"].get(stage_idx, BOT_DEMOS["n_episodes"])


# ─── Recurrent Policy Architecture ───────────────────────────────────────────

RECURRENT_POLICY = dict(
    # Instruction encoding
    word_emb_dim   = 32,       # token embedding dimension
    instr_gru_dim  = 128,      # biGRU hidden size (output = 256 bidirectional)
    instr_dim      = 256,      # final instruction embedding dim

    # Obs encoding
    obs_proj_dim   = 128,      # project flat obs to this before FiLM

    # FiLM conditioning
    film_dim       = 128,      # FiLM applies to obs_proj_dim features

    # Memory GRU
    memory_dim     = 256,      # GRUCell hidden dim (use 2048 for BossLevel)
    memory_dim_large = 2048,   # large model for hard levels

    # Output
    embed_dim      = 128,      # shared CLIP embedding dim (same as flat policy)
)


# ─── Imitation Learning (Behavioral Cloning) ─────────────────────────────────

IL = dict(
    epochs         = 20,       # epochs over the demo dataset
    lr             = 1e-4,
    batch_size     = 128,      # (obs, action) pairs per batch
    seq_len        = 40,       # TBPTT sequence length (truncated backprop through time)
    val_split      = 0.1,      # fraction of demos held out for validation
    patience       = 5,        # early stopping patience (epochs without val improvement)
    max_grad_norm  = 0.5,
    weight_decay   = 1e-5,
)


# ─── Curriculum Training ─────────────────────────────────────────────────────

CURRICULUM = dict(
    success_threshold  = 0.70,   # success rate to unlock next stage
    eval_episodes      = 50,     # episodes used to measure success rate
    max_il_epochs      = 30,     # max IL epochs per stage before moving on
    warm_start         = True,   # initialize from previous stage checkpoint
    demo_refresh       = True,   # re-collect demos after RL fine-tune
)


# ─── RL Fine-tune (after IL warm start) ───────────────────────────────────────
# Same PPO config but with recurrent policy and longer horizon

RL_FINETUNE = dict(
    total_timesteps  = 1_000_000,
    n_steps          = 4096,
    batch_size       = 128,
    n_epochs         = 10,
    gamma            = 0.99,
    gae_lambda       = 0.95,
    clip_range       = 0.1,      # tighter clip range for fine-tuning
    learning_rate    = 1e-4,
    ent_coef         = 0.05,     # higher entropy to keep exploring
    vf_coef          = 0.5,
    max_grad_norm    = 0.5,
    recurrence       = 20,       # TBPTT steps for recurrent policy
)
