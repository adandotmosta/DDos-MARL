from pathlib import Path

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# Paths
DATA_DIR     = BASE_DIR / "BCC-2024"
ARTIFACT_DIR = BASE_DIR / "artifacts"
VERSION      = "v1"

PREPROCESS_DIR = ARTIFACT_DIR / "preprocessing" / VERSION
LGBM_DIR       = ARTIFACT_DIR / "lgbm"          / VERSION
MARL_DIR       = ARTIFACT_DIR / "marl"           / VERSION
EVAL_DIR       = ARTIFACT_DIR / "evaluation"     / VERSION

# Ablation stage directories
S11_DIR  = MARL_DIR / "ablation" / "S11"
S12_DIR  = MARL_DIR / "ablation" / "S12"
S13_DIR  = MARL_DIR / "ablation" / "S13"
S14_DIR  = MARL_DIR / "ablation" / "S14"
S15_DIR  = MARL_DIR / "ablation" / "S15"
S16_DIR  = MARL_DIR / "ablation" / "S16"
S17_DIR  = MARL_DIR / "ablation" / "S17"
S18_DIR  = MARL_DIR / "ablation" / "S18"
S19_DIR  = MARL_DIR / "ablation" / "S19"
S21_DIR  = MARL_DIR / "ablation" / "S21"
S22_DIR  = MARL_DIR / "ablation" / "S22"
S23_DIR  = MARL_DIR / "ablation" / "S23"
S24_DIR  = MARL_DIR / "ablation" / "S24"
S25_DIR  = MARL_DIR / "ablation" / "S25"
S26_DIR  = MARL_DIR / "ablation" / "S26"
S26B_DIR = MARL_DIR / "ablation" / "S26b"
S27_DIR  = MARL_DIR / "ablation" / "S27"
S28_DIR  = MARL_DIR / "ablation" / "S28"
S29_DIR  = MARL_DIR / "ablation" / "S29"
S30_DIR  = MARL_DIR / "ablation" / "S30"
S31_DIR  = MARL_DIR / "ablation" / "S31"
S32_DIR  = MARL_DIR / "ablation" / "S32"
S33_DIR  = MARL_DIR / "ablation" / "S33"
S34_DIR  = MARL_DIR / "ablation" / "S34"
S35_DIR  = MARL_DIR / "ablation" / "S35"
S36_DIR  = MARL_DIR / "ablation" / "S36"
S37_DIR  = MARL_DIR / "ablation" / "S37"
S38_DIR  = MARL_DIR / "ablation" / "S38"
S39_DIR  = MARL_DIR / "ablation" / "S39"
S40_DIR  = MARL_DIR / "ablation" / "S40"
MAPL_DIR = MARL_DIR / "ablation" / "S9"
E13_DIR  = MARL_DIR / "ablation" / "S10"

# Data files
CSV_THURSDAY = DATA_DIR / "Thursday_14_Dec_2023.csv"
CSV_SATURDAY = DATA_DIR / "Saturday_16_Dec_2023.csv"
CSV_MONDAY   = DATA_DIR / "Monday_18_Dec_2023.csv"
CSV_TUESDAY  = DATA_DIR / "Tuesday_19_Dec_2023.csv"

# LightGBM bundles
LGBM_BUNDLE          = BASE_DIR / "Risk_Model" / "artifacts" / "vae" / "v1" / "models" / "lgbm_bundle_monday.pkl"
LGBM_BUNDLE_COMBINED = BASE_DIR / "Risk_Model" / "artifacts" / "vae" / "v1" / "models" / "lgbm_bundle_monday_tuesday.pkl"
FEATURES_JSON        = BASE_DIR / "Risk_Model" / "artifacts" / "vae" / "v1" / "selected_features.json"

# Preprocessing
WINDOW_SIZE = 30      # seconds per window
VAL_RATIO   = 0.2
RANDOM_SEED = 42
N_AGENTS    = 3

# LightGBM
LGBM_LABEL_MAP     = {"Benign": 0, "Suspicious": 1, "Attack": 2}
LGBM_IDX_TO_LABEL  = {0: "benign", 1: "suspicious", 2: "attack"}
LGBM_THRESHOLD     = 0.35

# Environment
Us             = 1e8      # server capacity (bytes/s)
IP_TIMEOUT     = 300
BASE_THRESHOLD = 0.35
FLOW_COUNT_MAX = 1000
IP_TABLE_MAX   = 500
UNIQUE_IP_MAX  = 500

VICTIM_IPS = {
    "3.96.128.96",
    "99.79.45.168",
    "15.222.45.224",
    "35.182.194.19",
    "3.99.186.200",
    "35.183.15.52",
}

PUZZLE_PASS = {
    "Benign":     0.95,
    "Suspicious": 0.50,
    "Attack":     0.05,
}

REWARD_LEGIT_W  = 1.0
REWARD_ATTACK_W = 1.0
PUZZLE_COST     = 0.0
LOCAL_REWARD    = False

DROP_PROBS = {
    2: 0.10, 3: 0.20, 4: 0.30, 5: 0.50,
    6: 0.70, 7: 0.90, 8: 0.95, 9: 1.00,
}

ACTIONS = {
    0: "forward", 1: "puzzle",
    2: "throttle_10", 3: "throttle_20", 4: "throttle_30",
    5: "throttle_50", 6: "throttle_70", 7: "throttle_90",
    8: "throttle_95", 9: "block",
}
ACTION_DIM = len(ACTIONS)

# State dimensions
EXTENDED_STATE = False
STATE_DIM  = 7 if EXTENDED_STATE else 5
JOINT_DIM  = STATE_DIM * N_AGENTS

# MARL training
N_EPISODES        = 500
GAMMA             = 0.99
GAE_LAMBDA        = 0.95
LR_ACTOR          = 3e-4
LR_CRITIC         = 1e-3
CLIP_EPS          = 0.2
ENTROPY_COEF      = 0.01
VALUE_COEF        = 0.5
MAX_GRAD_NORM     = 0.5
UPDATE_EVERY      = 10
EPOCHS_PER_UPDATE = 4
BATCH_SIZE        = 64
HIDDEN_DIM        = 128
CHECKPOINT_EVERY  = 50

ABLATION_STAGES = {
    "S1":  "Single agent, load-only state, blind throttle (Malialis baseline)",
    "S2":  "Single agent, 5 raw features, blind throttle",
    "S3":  "Single agent, LightGBM risk, blind throttle",
    "S4":  "Single agent, LightGBM risk, selective IP blocking",
    "S5":  "Single agent, LightGBM risk, selective + puzzle",
    "S6":  "3 agents, IQL",
    "S7":  "3 agents, IPPO (local critics)",
    "S8":  "3 agents, MAPPO + CTDE",
    "S9":  "3 agents, MAPPO + local reward",
    "S10": "3 agents, MAPPO + extended state (7-dim)",
    "S11": "3 agents, MAPPO + 10-action space",
    "S12": "3 agents, MAPPO + no puzzle, 9-action space",
    "S13": "3 agents, MAPPO + puzzle cost 0.05",
    "S14": "3 agents, MAPPO + puzzle_block_rate state",
    "S15": "3 agents, MAPPO + risk-aware throttle",
    "S16": "3 agents, MAPPO + dynamic puzzle",
    "S17": "3 agents, MAPPO + entropy state",
    "S18": "3 agents, MAPPO + curriculum learning",
    "S19": "3 agents, MAPPO + asymmetric reward (attack_w=3)",
    "S21": "3 agents, MAPPO + lower threshold (0.20)",
    "S22": "3 agents, MAPPO + counterfactual reward (COMA-style)",
    "S23": "3 agents, MAPPO + selective puzzle (suspicious-only)",
    "S24": "3 agents, LightGBM (Mon+Tue), selective puzzle",
    "S25": "3 agents, LightGBM (Mon+Tue), puzzle on flagged IPs",
    "S26": "3 agents, LightGBM (Mon+Tue), tiered profiles (p_attack routing)",
    "S26b":"3 agents, LightGBM (Mon only), tiered profiles (p_attack routing)",
    "S27": "3 agents, LightGBM (Mon only), tiered profiles (p_risk routing)  — FINAL MODEL",
    "S28": "3 agents, p_risk tiered, train on Monday+Thursday",
    "S29": "3 agents, IPPO + p_risk tiered — cooperation ablation",
    "S30": "3 agents, LightGBM (Mon+Tue), p_risk tiered — sensor ablation",
    "S31": "6 agents, IPPO + p_risk tiered",
    "S32": "3 agents, IPPO + 1-dim state [p_risk only]",
    "S33": "3 agents, MAPPO + shared IP blocklist",
    "S34": "3 agents, IPPO + shared IP blocklist",
    "S35": "3 agents, MAPPO + victim routing + shared IP blocklist",
    "S36": "3 agents, IPPO + victim routing + shared IP blocklist",
    "S37": "6 agents, MAPPO + p_risk tiered — scaling ablation",
    "S38": "3 agents, IPPO + ECMP 5-tuple routing",
    "S39": "3 agents, MAPPO + end-to-end FlowEncoder (no LightGBM)",
    "S40": "3 agents, MAPPO + global reward — reward ablation",
}
