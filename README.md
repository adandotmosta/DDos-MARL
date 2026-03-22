





# CoopIDS — Project Handoff Document
# For Claude Code / New Session

---

## 1. PROJECT OVERVIEW

**Name**: CoopIDS — Cooperative Multi-Agent Intrusion Detection System  
**Goal**: Graduate research project (solo, 4 weeks remaining)  
**Output**: ACM paper (5-11 pages)  
**Dataset**: BCCC-cPacket-Cloud-DDoS-2024 (real cloud traffic, 4 CSV days)

### One-Line Summary
Three cooperative MARL agents (MAPPO + CTDE) deployed at ISP edge routers learn
adaptive DDoS mitigation policies using LightGBM risk scores as state observations.

---

## 2. DATASET

### Files (all in BCC-2024/)
```
Thursday_14_Dec_2023.csv   105,087 flows   pure benign (no attacks)
Saturday_16_Dec_2023.csv   189,678 flows   pure benign (no attacks)
Monday_18_Dec_2023.csv     320,530 flows   TRAINING DAY (attack + benign + suspicious)
Tuesday_19_Dec_2023.csv     85,479 flows   TEST DAY (5 unseen attack types)
```

### Monday Label Distribution
```
Attack:     220,276  (12 attack types, sequential 20min blocks)
Benign:      68,444
Suspicious:  31,810  (flows during rest intervals between attacks)
```

### Monday Attack Schedule (important for understanding temporal structure)
```
09:00-09:20  TCP-Valid-SYN      → Windows-machine-1
09:30-09:50  TCP-BYPass-V1      → Linux-admin
10:00-10:20  TCP-Killall-v2     → Linux-webserver
10:32-10:52  TCP-IGMP           → Windows-machine-2
11:00-11:20  TCP-SYN            → Linux-admin
11:30-11:50  Killer-TCP         → Windows-machine-4
13:00-13:20  TCP-Control        → Linux-webserver
13:30-13:50  TCP-Flag-MIX       → Linux-admin
14:00-14:20  TCP-Flag-SYN       → Windows-machine-2
14:30-14:50  TCP-Flag-ACK       → Windows-machine-3
15:00-15:20  TCP-Flag-SYN-ACK   → Linux-webserver
15:30-15:50  TCP-Flag-ACK-PSH   → Windows-machine-4
```

### Tuesday Attacks (unseen — test generalization)
```
TCP-Flag-RST-ACK, TCP-Flag-SYN-TFO, TCP-Flag-SYN-TIME, TCP-Flag-OSYN, TCP-Flag-OSYNP
```

### Victim Machine IPs
```python
VICTIM_IPS = {
    "3.96.128.96",    # Windows-machine-1
    "99.79.45.168",   # Linux-admin
    "15.222.45.224",  # Linux-webserver
    "35.182.194.19",  # Windows-machine-2
    "3.99.186.200",   # Windows-machine-3
    "35.183.15.52",   # Windows-machine-4
}
```

### Timestamp Format
```
dtype: str
example: "2023-12-18 08:56:39.743361"
IMPORTANT: pandas 3.0 astype(np.int64) returns microseconds not nanoseconds.
CORRECT conversion: (pd.to_datetime(ts) - pd.Timestamp("1970-01-01")).dt.total_seconds()
```

---

## 3. RISK MODEL (Phase 1 — DONE)

### What Was Built
LightGBM 3-class classifier trained on Monday only (day split).
Outputs P(attack), P(benign), P(suspicious) per flow.

### Training Split
```
Train: Thursday + Saturday + Monday first 80% (time-ordered, no shuffle)
Test:  Tuesday (completely held out — never seen during training)
```

### Performance
```
Monday (training data):
  window-level gap (attack vs benign p_attack): 0.765  → Outcome A

Tuesday (unseen attacks):
  window-level gap: 0.268 (attack signal weak)
  BUT p_suspicious = 0.550 during attack windows
  Combined p_anomaly = p_attack + p_suspicious = 0.826 vs benign 0.446
  → System degrades gracefully: throttle→puzzle on unseen attacks
```

### Bundle Location
```
Risk_Model/artifacts/vae/v1/models/lgbm_bundle_monday.pkl
```

### Bundle Contents
```python
bundle = {
    "features":          list of 42 feature names,
    "handshake_mapping": dict str→int for categorical encoding,
    "imputer":           fitted SimpleImputer(strategy="median"),
    "model_3class":      fitted LGBMClassifier (3-class),
    "model_binary":      fitted LGBMClassifier (binary),
    "classes_3class":    ["attack", "benign", "suspicious"],
    "classes_binary":    ["attack", "benign"],
    "recommended_threshold_f1": 0.35,
}
```

---

## 4. MARL ARCHITECTURE (Phase 2 — IN PROGRESS)

### Topology
```
src_ip → router assignment: hash(src_ip) % 3  (fixed, permanent)

r1: ~1/3 of src_IPs ──┐
r2: ~1/3 of src_IPs ──┼──→ 6 victim machines
r3: ~1/3 of src_IPs ──┘
```

### State (per agent, 5-dimensional)
```python
obs_i = [
    p_attack,      # mean P(attack) over flows in 30s window
    p_benign,      # mean P(benign)
    p_suspicious,  # mean P(suspicious)
    p_anomaly,     # p_attack + p_suspicious (helps on Tuesday unseen attacks)
    load_norm,     # sum(bytes_rate) / Us, clipped [0,1]
]
STATE_DIM = 5
```

### Joint State (centralized critic only, training only)
```python
joint_obs = [obs_r1, obs_r2, obs_r3]  # 15-dimensional
JOINT_DIM = 15
```

### Action Space (6 actions)
```python
0: forward          # pass all flows
1: puzzle           # challenge flows (pass rate depends on true label)
2: throttle_light   # drop 30% of flagged IPs
3: throttle_medium  # drop 60% of flagged IPs
4: throttle_heavy   # drop 90% of flagged IPs
5: block            # drop 100% of flagged IPs
```

### Puzzle Mechanism (most novel contribution)
```python
# Pass rates when puzzle is issued (based on true label)
PUZZLE_PASS = {
    "Benign":     0.95,  # legit users almost always pass
    "Suspicious": 0.50,  # uncertain → 50/50
    "Attack":     0.05,  # bots almost never pass
}
# Directly motivated by dataset's "suspicious" label
# Used by real systems: Cloudflare, AWS Shield
```

### IP Risk Table (per agent, persistent within episode)
```python
ip_risk_table = {}  # {src_ip: (risk_score, timestamp_sec)}
# Populated: flows with P(attack) >= 0.35 → added to table
# Throttle/block actions only affect IPs IN this table
# Decay: entries older than 300 seconds are removed
```

### Reward (global, shared across all agents)
```python
server_load = sum(bytes_rate for forwarded flows to victim_ips)
if server_load > Us:          # Us = 1e8 bytes/s
    reward = -(server_load / (2 * Us))   # proportional punishment
else:
    reward = legit_passed / legit_total  # legitimate ratio [0,1]
```

### Algorithm: MAPPO + CTDE
```
Training:  Centralized (critic sees joint_obs of all 3 agents)
Execution: Decentralized (each actor sees only local obs_i)
Update:    PPO clipped objective, GAE advantage estimation
```

### Training Loop
```
One episode = full replay of Monday (08:56 → 17:00)
Window size = 30 seconds
Windows per episode = ~966
N_episodes = 500
Expected training time = ~2-3 hours (vectorized)
```

---

## 5. FOLDER STRUCTURE

```
Research_project/
│
├── BCC-2024/                          ← raw CSV files (READ ONLY)
│   ├── Thursday_14_Dec_2023.csv
│   ├── Saturday_16_Dec_2023.csv
│   ├── Monday_18_Dec_2023.csv
│   └── Tuesday_19_Dec_2023.csv
│
├── Risk_Model/                        ← Phase 1 (DONE)
│   ├── train_eval_save_lgbm.py        ← LightGBM training script
│   ├── eval_lgbm_daily_split.py       ← evaluation script
│   ├── compute_window_risk_stats.py   ← window-level separation check
│   ├── lgbm_runtime_infer.py          ← inference demo
│   └── artifacts/
│       └── vae/v1/
│           ├── selected_features.json ← 42 feature names
│           └── models/
│               ├── lgbm_bundle_monday.pkl  ← MAIN BUNDLE (train on Monday only)
│               └── lgbm_bundle.pkl         ← older bundle (random split, DO NOT USE)
│
├── artifacts/                         ← MARL outputs
│   ├── marl/v1/
│   │   ├── actor_r1.pt               ← saved after training
│   │   ├── actor_r2.pt
│   │   ├── actor_r3.pt
│   │   ├── critic.pt
│   │   ├── best/                     ← best episode checkpoint
│   │   └── training_log.json
│   └── evaluation/v1/
│       ├── monday_results.json
│       ├── tuesday_results.json
│       └── ablation_results.json
│
├── src/                               ← Phase 2 (IN PROGRESS)
│   ├── __init__.py
│   ├── environment.py                 ← RouterEnvironment + MultiAgentEnv
│   ├── mappo.py                       ← Actor + Critic + RolloutBuffer + MAPPO
│   ├── train_marl.py                  ← training loop
│   └── evaluate.py                    ← evaluation + ablation
│
├── config.py                          ← ALL hyperparameters
├── main.py                            ← pipeline entry point
└── requirements.txt
```

---

## 6. CURRENT BUG (needs fixing first)

### Symptom
```
Approx windows per episode: 307502498   ← should be ~966
```

### Root Cause
pandas 3.0 breaking change: `astype(np.int64)` on datetime now returns
microseconds instead of nanoseconds. Dividing by 1e9 gives wrong seconds.

### Fix Required (2 places in src/environment.py)

**Place 1** — RouterEnvironment.__init__ (~line 105):
```python
# WRONG (old):
self._ts_sec = pd.to_datetime(self.flows["timestamp"]).astype(np.int64).to_numpy() / 1e9

# CORRECT:
_epoch = pd.Timestamp("1970-01-01")
self._ts_sec = (
    (pd.to_datetime(self.flows["timestamp"]) - _epoch)
    .dt.total_seconds()
    .to_numpy()
    .astype(np.float64)
)
```

**Place 2** — MultiAgentEnv.__init__ (~line 325):
```python
# WRONG (old):
ts_sec = pd.to_datetime(df["timestamp"]).astype(np.int64).to_numpy() / 1e9

# CORRECT:
_epoch = pd.Timestamp("1970-01-01")
ts_sec = (
    (pd.to_datetime(df["timestamp"]) - _epoch)
    .dt.total_seconds()
    .to_numpy()
    .astype(np.float64)
)
```

### Verify Fix Works
```bash
python -c "
import src.environment as e
import inspect
print('Loading from:', inspect.getfile(e))
"
# Should print path to YOUR src/environment.py

python main.py --stage marl
# Should print: Approx windows per episode: 966
```

---

## 7. KNOWN ISSUES (after timestamp fix)

### Issue: sklearn warning on every prediction
```
UserWarning: X does not have valid feature names, but LGBMClassifier
was fitted with feature names
```
**Fix**: Pass DataFrame not numpy array to predict_proba, OR suppress warning.
```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
```
Add this at top of environment.py.

---

## 8. HYPERPARAMETERS (all in config.py)

```python
# Environment
WINDOW_SIZE    = 30       # seconds
Us             = 1e8      # server capacity bytes/s
IP_TIMEOUT     = 300      # seconds before IP table entry expires
BASE_THRESHOLD = 0.35     # P(attack) threshold for IP table

# MARL
N_EPISODES     = 500
STATE_DIM      = 5        # per agent
JOINT_DIM      = 15       # 5 × 3 agents
ACTION_DIM     = 6
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
LR_ACTOR       = 3e-4
LR_CRITIC      = 1e-3
CLIP_EPS       = 0.2
ENTROPY_COEF   = 0.01
UPDATE_EVERY   = 10       # episodes between PPO updates
HIDDEN_DIM     = 128
```

---

## 9. ABLATION STUDY (Week 3)

| Stage | State | Action | Algorithm | Purpose |
|-------|-------|--------|-----------|---------|
| S1 | load only | blind throttle | Single agent | Malialis baseline |
| S2 | 5 raw features | blind throttle | Single agent | Richer state |
| S3 | LightGBM risk | blind throttle | Single agent | Supervised risk value |
| S4 | LightGBM risk | selective IP | Single agent | IP blocking value |
| S5 | LightGBM risk | selective+puzzle | Single agent | Puzzle action value |
| S6 | LightGBM risk | selective+puzzle | IQL 3 agents | Multi-agent value |
| S7 | LightGBM risk | selective+puzzle | MAPPO no CTDE | Coordination value |
| S8 | LightGBM risk | selective+puzzle | MAPPO + CTDE | Full system |

---

## 10. EVALUATION METRICS

```python
TPR         = attack_blocked / total_attack        # detection rate
FPR         = benign_blocked / total_benign        # false alarm rate
legit_ratio = benign_passed  / total_benign        # legitimate preservation
F1          = 2 * precision * TPR / (precision + TPR)
reward      = episode cumulative reward
```

### Test Scenarios
```
1. Monday held-out (last 20% by timestamp) → seen attack types
2. Tuesday full day                         → unseen attack types
3. Ablation S1→S8                          → contribution of each component
```

---

## 11. PAPER NARRATIVE

*"We propose CoopIDS — a cooperative MARL framework for DDoS detection and
mitigation at ISP edge routers. Three distributed agents learn adaptive
response policies using LightGBM-based risk scoring on real cloud DDoS traffic
(BCCC-cPacket-Cloud-DDoS-2024). Unlike Malialis who applies blind throttling on
synthetic traffic, our agents learn a three-tier response: forwarding benign
traffic, issuing puzzle challenges to suspicious sources (directly motivated by
the dataset's suspicious label), and selectively throttling confirmed attackers
by IP. The CTDE framework enables agents to recognize coordinated attack
signatures spanning multiple ingress points. Evaluated on 12 TCP-based attack
types with generalization testing on 5 unseen attack variants, CoopIDS
demonstrates strong detection while preserving legitimate traffic."*

### Key Contributions Over Prior Work
| Aspect | Malialis | BCCC Paper (RF) | CoopIDS |
|--------|----------|-----------------|---------|
| Data | Synthetic ns-2 | Real BCCC-2024 | Real BCCC-2024 |
| Risk score | Load only | None | LightGBM 3-class |
| Action | Blind throttle | None | Forward/Puzzle/Throttle/Block |
| Suspicious | Ignored | Labeled only | Puzzle challenge |
| Architecture | MARL throttling | Centralized RF | Cooperative MARL |
| Coordination | Shared reward | None | CTDE joint patterns |
| Generalization | Not tested | Not tested | Tuesday unseen attacks |

---

## 12. REMAINING TIMELINE (4 weeks)

```
Week 1 (NOW):
  ✅ Dataset understood
  ✅ LightGBM trained + validated (gap=0.765 Monday, 0.741 Tuesday)
  ✅ Architecture designed
  ✅ Code written (config, environment, mappo, train_marl, evaluate)
  ❌ FIX: timestamp bug in environment.py (307M windows → 966)
  ❌ RUN: python main.py --stage marl (first episode sanity check)

Week 2:
  ❌ Full MARL training (500 episodes, ~2-3 hours)
  ❌ Validate reward curve is learning (not flat)
  ❌ Monday held-out evaluation

Week 3:
  ❌ Ablation study S1-S8
  ❌ Tuesday generalization evaluation
  ❌ Comparison table (all stages + metrics)

Week 4:
  ❌ ACM paper writing (5-11 pages)
  ❌ Figures: reward curve, confusion matrix, ablation table
  ❌ Final submission
```

---

## 13. ENVIRONMENT (Python)

```
Python:  3.14
pandas:  3.0.1   ← has breaking change on datetime astype(int64)
lightgbm: latest
torch:   latest
sklearn: latest
virtualenv: my_env
```

### Run Command
```bash
cd /Users/abdenourbelarbi/Documents/MultiAgents/MARL_Cybersecurity/Research_project
source my_env/bin/activate
python main.py --stage marl
```