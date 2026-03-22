"""
config.py
Central configuration for CoopIDS.
All hyperparameters, paths, and constants in one place.
Change values here — everything else reads from here.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# Safe BASE_DIR for both script and notebook
# ─────────────────────────────────────────────
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
DATA_DIR     = BASE_DIR / "BCC-2024"
ARTIFACT_DIR = BASE_DIR / "artifacts"
VERSION      = "v1"

PREPROCESS_DIR = ARTIFACT_DIR / "preprocessing" / VERSION
LGBM_DIR       = ARTIFACT_DIR / "lgbm"          / VERSION
MARL_DIR       = ARTIFACT_DIR / "marl"           / VERSION
EVAL_DIR       = ARTIFACT_DIR / "evaluation"     / VERSION

# CSV filenames
CSV_THURSDAY = DATA_DIR / "Thursday_14_Dec_2023.csv"
CSV_SATURDAY = DATA_DIR / "Saturday_16_Dec_2023.csv"
CSV_MONDAY   = DATA_DIR / "Monday_18_Dec_2023.csv"
CSV_TUESDAY  = DATA_DIR / "Tuesday_19_Dec_2023.csv"

# LightGBM bundle (trained on Monday only — day split)
LGBM_BUNDLE  = BASE_DIR / "Risk_Model" / "artifacts" / "vae" / "v1" / "models" / "lgbm_bundle_monday.pkl"
FEATURES_JSON= BASE_DIR / "Risk_Model" / "artifacts" / "vae" / "v1" / "selected_features.json"

# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────
WINDOW_SIZE  = 30     # seconds per window
VAL_RATIO    = 0.2    # time-ordered val split
RANDOM_SEED  = 42
N_AGENTS     = 3

# ─────────────────────────────────────────────
# LightGBM
# ─────────────────────────────────────────────
LGBM_LABEL_MAP = {
    "Benign":     0,
    "Suspicious": 1,
    "Attack":     2,
}
LGBM_IDX_TO_LABEL = {0: "benign", 1: "suspicious", 2: "attack"}

# Best threshold from threshold sweep (day split, best F1)
LGBM_THRESHOLD = 0.35

# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────

# Server capacity in bytes/s
# Used for Malialis-style reward normalization
Us = 1e8

# IP risk table — how long before an entry expires (seconds)
IP_TIMEOUT = 300

# Minimum P(attack) for a flow to be added to IP risk table
BASE_THRESHOLD = 0.35

# Victim machine IPs (from paper Table A1)
VICTIM_IPS = {
    "3.96.128.96",    # Windows-machine-1
    "99.79.45.168",   # Linux-admin-1
    "15.222.45.224",  # Linux-webserver
    "35.182.194.19",  # Windows-machine-2
    "3.99.186.200",   # Windows-machine-3
    "35.183.15.52",   # Windows-machine-4
}

# Puzzle: probability a flow passes challenge given its true label
PUZZLE_PASS = {
    "Benign":     0.95,   # legitimate users almost always solve
    "Suspicious": 0.50,   # uncertain — 50/50
    "Attack":     0.05,   # bots almost never solve
}

# Drop probabilities for throttle actions
# action_id → drop probability
DROP_PROBS = {
    2: 0.3,   # throttle_light
    3: 0.6,   # throttle_medium
    4: 0.9,   # throttle_heavy
    5: 1.0,   # block
}

# Action space
ACTIONS = {
    0: "forward",
    1: "puzzle",
    2: "throttle_light",
    3: "throttle_medium",
    4: "throttle_heavy",
    5: "block",
}
ACTION_DIM = len(ACTIONS)  # 6

# ─────────────────────────────────────────────
# MARL — State dimensions
# ─────────────────────────────────────────────
# Per-agent state: [p_attack, p_benign, p_suspicious, p_anomaly, load_norm]
STATE_DIM  = 5
JOINT_DIM  = STATE_DIM * N_AGENTS   # 15  (centralized critic input)

# ─────────────────────────────────────────────
# MARL — Training
# ─────────────────────────────────────────────
N_EPISODES    = 500
GAMMA         = 0.99       # discount factor
GAE_LAMBDA    = 0.95       # GAE lambda for advantage estimation
LR_ACTOR      = 3e-4       # actor learning rate
LR_CRITIC     = 1e-3       # critic learning rate
CLIP_EPS      = 0.2        # PPO clip epsilon
ENTROPY_COEF  = 0.01       # entropy bonus coefficient
VALUE_COEF    = 0.5        # value loss coefficient
MAX_GRAD_NORM = 0.5        # gradient clipping
UPDATE_EVERY  = 10         # episodes between MAPPO updates
EPOCHS_PER_UPDATE = 4      # PPO epochs per update
BATCH_SIZE    = 64
HIDDEN_DIM    = 128        # actor/critic hidden layer size

# Checkpoint every N episodes
CHECKPOINT_EVERY = 50

# ─────────────────────────────────────────────
# Ablation study stage names
# ─────────────────────────────────────────────
ABLATION_STAGES = {
    "S1": "Single agent, load-only state, blind throttle (Malialis baseline)",
    "S2": "Single agent, 5 raw features, blind throttle",
    "S3": "Single agent, LightGBM risk, blind throttle",
    "S4": "Single agent, LightGBM risk, selective IP blocking",
    "S5": "Single agent, LightGBM risk, selective + puzzle",
    "S6": "3 agents, LightGBM risk, selective + puzzle, IQL",
    "S7": "3 agents, LightGBM risk, selective + puzzle, IQL no CTDE",
    "S8": "3 agents, LightGBM risk, selective + puzzle, MAPPO + CTDE",
}
