"""
Central config for CLIP-RL.
All hyperparameters and task definitions live here.
"""

# ─── Environment ──────────────────────────────────────────────────────────────

# BabyAI observation: partial 7x7 grid view, each cell = (object, color, state)
OBS_IMAGE_SHAPE = (7, 7, 3)          # raw obs["image"] shape
OBS_IMAGE_DIM   = 7 * 7 * 3         # = 147 flattened
DIRECTION_DIM   = 1                  # obs["direction"] scalar, 0-3
OBS_DIM         = OBS_IMAGE_DIM + DIRECTION_DIM  # = 148 total input size

ACTION_DIM      = 7                  # Discrete(7): turn_left, turn_right,
                                     # forward, pickup, drop, toggle, done

# ─── Tasks ────────────────────────────────────────────────────────────────────
# (env_id, instruction)
# Instructions are the exact strings the language encoder will embed.
# Base tasks: trained to convergence, used for CLIP alignment.
# Target tasks: never trained, used to evaluate transfer.

BASE_TASKS = [
    ("BabyAI-GoToRedBall-v0",      "go to the red ball"),
    ("BabyAI-GoToRedBallGrey-v0",  "go to the red ball in the grey room"),
    ("BabyAI-GoToRedBallNoDists-v0","go to the red ball with no distractors"),
    ("BabyAI-GoToObj-v0",          "go to the blue box"),
    ("BabyAI-GoToLocal-v0",        "go to the green key"),
    ("BabyAI-GoToDoor-v0",         "go to the blue door"),
    ("BabyAI-OpenRedDoor-v0",      "open the red door"),
    ("BabyAI-Open-v0",             "open the door"),
]

TARGET_TASKS = [
    # Unseen object/color combinations — tests genuine transfer
    ("BabyAI-GoToObj-v0",   "go to the red key"),      # new color+object combo
    ("BabyAI-GoToObj-v0",   "go to the grey ball"),
    ("BabyAI-Open-v0",      "open the blue door"),
]

# ─── PPO Hyperparameters ───────────────────────────────────────────────────────

PPO = dict(
    total_timesteps  = 500_000,    # steps per base task
    n_steps          = 2048,       # steps per rollout buffer
    batch_size       = 64,
    n_epochs         = 10,
    gamma            = 0.99,
    gae_lambda       = 0.95,
    clip_range       = 0.2,
    learning_rate    = 3e-4,
    ent_coef         = 0.01,       # entropy bonus — important for exploration
    vf_coef          = 0.5,
    max_grad_norm    = 0.5,
)

# ─── Policy Network ───────────────────────────────────────────────────────────

POLICY_NET = dict(
    hidden_dims = [256, 128],      # two hidden layers
    activation  = "relu",
)

# ─── Trajectory Collection ────────────────────────────────────────────────────

TRAJECTORY = dict(
    n_episodes      = 50,          # episodes per policy for behavioral embedding
    max_steps       = 200,         # max steps per episode (BabyAI default is ~200)
)

# ─── Trajectory Encoder (GRU) ─────────────────────────────────────────────────

TRAJ_ENCODER = dict(
    input_dim    = OBS_DIM + ACTION_DIM,  # 148 + 7 = 155 per timestep
    proj_dim     = 128,            # per-step projection before GRU
    gru_hidden   = 256,
    gru_layers   = 2,
    embed_dim    = 128,            # final policy embedding size (shared CLIP space)
    dropout      = 0.1,
)

# ─── Language Encoder ─────────────────────────────────────────────────────────

LANG_ENCODER = dict(
    model_name   = "all-MiniLM-L6-v2",  # sentence-transformers model
    sbert_dim    = 384,                  # output dim of MiniLM
    embed_dim    = 128,                  # projected dim (shared CLIP space)
)

# ─── CLIP Training ────────────────────────────────────────────────────────────

CLIP_TRAIN = dict(
    epochs          = 100,
    lr              = 1e-4,
    temperature     = 0.07,        # δ — learnable but initialized here
    learn_temp      = True,        # whether to make temperature a learnable param
    batch_size      = len(BASE_TASKS),  # all pairs at once (N is small)
)

# ─── Paths ────────────────────────────────────────────────────────────────────

import os
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS  = os.path.join(ROOT, "checkpoints")
RESULTS      = os.path.join(ROOT, "results")

os.makedirs(CHECKPOINTS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

def policy_checkpoint_path(task_id: str) -> str:
    return os.path.join(CHECKPOINTS, f"policy_{task_id}.zip")

def trajectory_path(task_id: str) -> str:
    return os.path.join(CHECKPOINTS, f"trajectories_{task_id}.pkl")

def clip_checkpoint_path() -> str:
    return os.path.join(CHECKPOINTS, "clip_model.pt")