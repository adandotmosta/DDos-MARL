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

PREPROCESS_DIR    = ARTIFACT_DIR / "preprocessing" / VERSION
LGBM_DIR          = ARTIFACT_DIR / "lgbm"          / VERSION
MARL_DIR          = ARTIFACT_DIR / "marl"           / VERSION
MAPL_DIR          = MARL_DIR / "ablation" / "S9"    # MAPPO + local reward
E13_DIR           = MARL_DIR / "ablation" / "S10"   # MAPPO + extended state (E13)
S11_DIR           = MARL_DIR / "ablation" / "S11"   # MAPPO + 10-action space
S12_DIR           = MARL_DIR / "ablation" / "S12"   # MAPPO + no puzzle, 9-action space
EVAL_DIR          = ARTIFACT_DIR / "evaluation"     / VERSION

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

# State normalisation constants
# Max flows expected in a single 30s window (attack windows ~600-800 flows)
FLOW_COUNT_MAX = 1000
# Soft ceiling for IP risk table size (used to normalise ip_table_ratio)
IP_TABLE_MAX   = 500
# Max unique src IPs expected in a 30s window (DDoS windows ~500+ sources)
UNIQUE_IP_MAX  = 500

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

# Reward weights
# reward = REWARD_LEGIT_W * legit_ratio - REWARD_ATTACK_W * attack_ratio
# Symmetric baseline: both = 1.0
# Asymmetric (attacks more important): REWARD_ATTACK_W > 1.0
REWARD_LEGIT_W  = 1.0
REWARD_ATTACK_W = 1.0

# Per-agent local reward experiment
# False → shared global reward (default, all experiments above)
# True  → each agent rewarded on its own router's flows only
LOCAL_REWARD = False

# Drop probabilities for throttle actions
# action_id → drop probability
DROP_PROBS = {
    2: 0.10,  # throttle_10
    3: 0.20,  # throttle_20
    4: 0.30,  # throttle_30
    5: 0.50,  # throttle_50
    6: 0.70,  # throttle_70
    7: 0.90,  # throttle_90
    8: 0.95,  # throttle_95
    9: 1.00,  # block
}

# Action space
ACTIONS = {
    0: "forward",
    1: "puzzle",
    2: "throttle_10",
    3: "throttle_20",
    4: "throttle_30",
    5: "throttle_50",
    6: "throttle_70",
    7: "throttle_90",
    8: "throttle_95",
    9: "block",
}
ACTION_DIM = len(ACTIONS)  # 10

# ─────────────────────────────────────────────
# MARL — State dimensions
# ─────────────────────────────────────────────
# E13 — Extended state with spatial features
# False → 5-dim state (baseline): [p_attack, p_benign, p_suspicious, ip_table_ratio, flow_count_norm]
# True  → 7-dim state (E13):      above + [unique_src_ips_norm, attack_ip_ratio]
EXTENDED_STATE = False

STATE_DIM  = 7 if EXTENDED_STATE else 5
JOINT_DIM  = STATE_DIM * N_AGENTS   # 21 if extended, 15 baseline

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
    "S7": "3 agents, LightGBM risk, selective + puzzle, IPPO (local critics, no CTDE)",
    "S8": "3 agents, LightGBM risk, selective + puzzle, MAPPO + CTDE",
    "S9": "3 agents, LightGBM risk, selective + puzzle, MAPPO + local reward",
    "S10": "3 agents, LightGBM risk, selective + puzzle, MAPPO + extended state (7-dim)",
    "S11": "3 agents, LightGBM risk, selective + puzzle, MAPPO + 10-action space",
    "S12": "3 agents, LightGBM risk, no puzzle, MAPPO + 9-action space",
}
