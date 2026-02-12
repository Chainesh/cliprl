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
    # Stage 0 — pure navigation, single room
    [
        ("BabyAI-GoToRedBall-v0",        "go to the red ball"),
        ("BabyAI-GoToRedBallNoDists-v0", "go to the red ball with no distractors"),
        ("BabyAI-GoToObj-v0",            "go to the blue box"),
        ("BabyAI-GoToLocal-v0",          "go to the green key"),
    ],
    # Stage 1 — doors, multi-object, slightly larger rooms
    [
        ("BabyAI-GoToDoor-v0",           "go to the blue door"),
        ("BabyAI-OpenRedDoor-v0",        "open the red door"),
        ("BabyAI-Open-v0",               "open the door"),
        ("BabyAI-OpenDoor-v0",           "open the door on your left"),
    ],
    # Stage 2 — keys + locked doors
    [
        ("BabyAI-Unlock-v0",             "open the locked door"),
        ("BabyAI-UnlockLocal-v0",        "open the locked door in this room"),
        ("BabyAI-KeyInBox-v0",           "use the key to open the box"),
    ],
    # Stage 3 — pick-and-place, object manipulation
    [
        ("BabyAI-PickupLoc-v0",          "pick up the ball in front of you"),
        ("BabyAI-PutNextLocal-v0",       "put the ball next to the key"),
        ("BabyAI-PutNextS5N3-v0",        "put the red ball next to the blue key"),
    ],
    # Stage 4 — sequential / compositional
    [
        ("BabyAI-GoToSeqS5R2-v0",        "go to a red door then go to a blue ball"),
        ("BabyAI-SynthS5R2-v0",          "pick up the red ball after opening the door"),
    ],
    # Stage 5 — BossLevel: everything combined
    [
        ("BabyAI-BossLevel-v0",          "boss level task"),
    ],
]

# Flat list of all IL tasks (all stages combined) for easy iteration
ALL_IL_TASKS = [task for stage in CURRICULUM_STAGES for task in stage]


# ─── Bot demo collection ──────────────────────────────────────────────────────

BOT_DEMOS = dict(
    n_episodes     = 10_000,   # demos per task for IL pretraining
    max_steps      = 256,      # max steps per episode
    save_every     = 1_000,    # checkpoint demos every N episodes
    timeout        = 30,       # seconds before giving up on one episode
)


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