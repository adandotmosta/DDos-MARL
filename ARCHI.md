# CoopIDS — Architecture Overview

**Project**: Cooperative Multi-Agent Reinforcement Learning for Intrusion Detection
**Status**: Phase 2 (MARL training in progress)
**Timeline**: 4 weeks | ACM paper submission

---

## 1. SYSTEM CONTEXT

### Goal
Three cooperative MARL agents deployed at ISP edge routers learn adaptive DDoS mitigation policies using LightGBM risk scores as state observations.

### One-Liner
Three agents at ISP routers learn to detect and mitigate TCP-based DDoS attacks on real cloud traffic, balancing attack detection with legitimate traffic preservation.

### Novel Contributions
- **Risk Scoring**: LightGBM 3-class classifier (attack/benign/suspicious) replaces blind heuristics
- **Puzzle Challenges**: Selective rate-limiting for suspicious sources (novel action type)
- **Coordination**: CTDE training enables agents to recognize multi-ingress attack patterns
- **Real Data**: Evaluated on BCCC-cPacket-Cloud-DDoS-2024 with generalization to unseen attacks

---

## 2. DATA LAYER

### Dataset: BCCC-2024 (Cloud DDoS Traffic)

| File | Flows | Purpose |
|------|-------|---------|
| **Thursday_14_Dec_2023.csv** | 105K | Training (benign baseline) |
| **Saturday_16_Dec_2023.csv** | 190K | Training (benign baseline) |
| **Monday_18_Dec_2023.csv** | 321K | Training (12 attack types + benign + suspicious) |
| **Tuesday_19_Dec_2023.csv** | 85K | Testing (5 unseen attack variants) |

### Monday Attack Schedule (Temporal Structure)
```
09:00–09:20  TCP-Valid-SYN       → Windows-machine-1
09:30–09:50  TCP-BYPass-V1       → Linux-admin
10:00–10:20  TCP-Killall-v2      → Linux-webserver
... (12 total types, 20-min blocks, rest intervals with "suspicious" label)
```

### Label Distribution (Monday)
- **Attack**: 220,276 flows (68.7% — 12 sequential DDoS types)
- **Benign**: 68,444 flows (21.4%)
- **Suspicious**: 31,810 flows (9.9% — temporal "gray zone" between attacks)

### Victim Machines
```python
VICTIM_IPS = {
    "3.96.128.96",      # Windows-machine-1
    "99.79.45.168",     # Linux-admin
    "15.222.45.224",    # Linux-webserver
    "35.182.194.19",    # Windows-machine-2
    "3.99.186.200",     # Windows-machine-3
    "35.183.15.52",     # Windows-machine-4
}
```

---

## 3. RISK MODEL (Phase 1 ✅ Complete)

### Component: LightGBM 3-Class Classifier

#### Purpose
Pre-process raw flow features into risk probabilities:
- `P(attack)` — likelihood flow is part of DDoS attack
- `P(benign)` — likelihood flow is legitimate
- `P(suspicious)` — likelihood flow is in attack transition zone

#### Training Split
```
Train: Thursday + Saturday + first 80% of Monday (time-ordered, no shuffle)
Test:  Tuesday (5 unseen attack types)
```

#### Performance
```
Monday (seen attacks):
  → window-level P(attack) gap = 0.765 (strong separation)

Tuesday (unseen attacks):
  → P(attack) gap = 0.268 (weak, model hasn't seen these types)
  → BUT: P(attack) + P(suspicious) = 0.826 vs benign 0.446
  → Graceful degradation: system escalates to puzzle → throttle on unknowns
```

#### Bundle Contents
```python
{
    "features":           [42 feature names],
    "handshake_mapping":  {str → int} for categorical encoding,
    "imputer":            SimpleImputer(strategy="median"),
    "model_3class":       LGBMClassifier (attack/benign/suspicious),
    "model_binary":       LGBMClassifier (attack/benign fallback),
    "classes_3class":     ["attack", "benign", "suspicious"],
    "recommended_threshold": 0.35,
}
```

#### Location
```
Risk_Model/artifacts/vae/v1/models/lgbm_bundle_monday.pkl
```

---

## 4. ENVIRONMENT LAYER

### Architecture: 3-Router ISP Topology

```
Source IP Assignment (Hash-based, permanent):
    src_ip → router_id = hash(src_ip) % 3

    Router-1 (r1) — 1/3 of source IPs ──┐
    Router-2 (r2) — 1/3 of source IPs ──├──→ 6 victim machines
    Router-3 (r3) — 1/3 of source IPs ──┘
```

**Rationale**: Realistic routing model. Same IP always enters via same router.

### State Space (Per Agent)

#### Baseline State (5-dimensional)
```python
obs_i = [
    p_attack,           # mean P(attack) over flows in 30s window
    p_benign,           # mean P(benign)
    p_suspicious,       # mean P(suspicious)
    ip_table_ratio,     # |IP risk table| / IP_TABLE_MAX, clipped [0,1]
    flow_count_norm,    # |flows| / FLOW_COUNT_MAX, clipped [0,1]
]
STATE_DIM = 5
```

#### Extended State Option (7-dimensional, E13 ablation)
```python
obs_i_extended = [
    ... (5 baseline features above) +
    unique_src_ips_norm,  # |unique sources| / UNIQUE_IP_MAX
    attack_ip_ratio,      # |attack sources in table| / |total in table|
]
STATE_DIM = 7 (if EXTENDED_STATE=True)
```

#### Joint State (Training Only, CTDE)
```python
joint_obs = [obs_r1, obs_r2, obs_r3]  # concatenated
JOINT_DIM = 15 (baseline) or 21 (extended)
# Used by centralized critic during training
# Actors NEVER see this at execution (true CTDE)
```

### Action Space (6 or 10 actions)

#### Baseline Actions (6-action)
```python
{
    0: "forward",          # pass all flows (no-op)
    1: "puzzle",           # challenge flagged IPs
    2: "throttle_light",   # drop 30% of flagged IPs
    3: "throttle_medium",  # drop 60%
    4: "throttle_heavy",   # drop 90%
    5: "block",            # drop 100% (complete blacklist)
}
ACTION_DIM = 6
```

#### Extended Actions (10-action, S11 ablation)
```python
{
    0: "forward",
    1: "puzzle",
    2: "throttle_10", 3: "throttle_20", ..., 8: "throttle_95",
    9: "block",
}
ACTION_DIM = 10
```

### Key Mechanisms

#### IP Risk Table (Per Router, Persistent)
```python
ip_risk_table = {}  # {src_ip: (risk_score, timestamp_sec)}

# Insertion:
if P(attack) >= BASE_THRESHOLD (0.35):
    ip_risk_table[src_ip] = (p_attack, current_time_sec)

# Decay:
for ip in list(ip_risk_table.keys()):
    if (current_time - timestamp) > IP_TIMEOUT (300s):
        del ip_risk_table[ip]

# Actions apply ONLY to IPs in this table
# Forward/puzzle assume flow's src_ip → decide passage
```

#### Puzzle Challenge Mechanism
```python
PUZZLE_PASS = {
    "Benign":     0.95,  # legitimate users solve ~95% of puzzles
    "Suspicious": 0.50,  # uncertain → 50/50 pass rate
    "Attack":     0.05,  # bots solve ~5% (rate-limiting effective)
}

# When puzzle action taken on a flow:
if random() < PUZZLE_PASS[true_label]:
    flow_passes = True
else:
    flow_passes = False  # dropped (simulated rate-limit)
```

**Motivation**: Dataset has explicit "Suspicious" label for attack transition zones. Puzzles (CAPTCHAs, proof-of-work) are used by real systems (Cloudflare, AWS Shield) to mitigate.

#### Reward Function (Global, Shared)

```python
# Server load from forwarded flows (bytes/second)
server_load = sum(bytes_rate for flows to victim_ips)

if server_load > Us (1e8 bytes/s):  # capacity exceeded
    reward = -(server_load / (2 * Us))  # penalty proportional to overload
else:
    # Reward legitimate traffic preservation
    legit_passed = count(benign flows that passed)
    legit_total  = count(all benign flows)
    reward = legit_passed / legit_total  ∈ [0, 1]
```

**Interpretation**:
- If under capacity: maximize legitimate throughput (preserve users)
- If over capacity: minimize excess load (prevent server crash)
- All agents share same reward (cooperative training)

---

## 5. MARL LAYER (Phase 2)

### Algorithm: MAPPO (Multi-Agent Proximal Policy Optimization)

#### Training Paradigm: CTDE (Centralized Training, Decentralized Execution)
```
Training Phase:
  ┌─────────────────────────────────────────┐
  │ Centralized Critic                      │
  │ Input: joint_obs (r1, r2, r3 combined) │
  │ Output: value V(joint_obs)              │
  └─────────────────────────────────────────┘
                    ↑
       Computes advantages & policy gradients
                    ↑
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Actor r1 │  │ Actor r2 │  │ Actor r3 │
  │ π(·|o₁)  │  │ π(·|o₂)  │  │ π(·|o₃)  │
  └──────────┘  └──────────┘  └──────────┘

Execution Phase:
  Each agent observes only LOCAL obs_i
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Actor r1 │  │ Actor r2 │  │ Actor r3 │
  │ Input: o1│  │ Input: o2│  │ Input: o3│
  │ Output: a│  │ Output: a│  │ Output: a│
  └──────────┘  └──────────┘  └──────────┘
  (NO access to other agents' observations)
```

**Why CTDE?**
- Critic can learn joint patterns (e.g., coordinated attacks spanning multiple ingress)
- Actors remain decentralized (deployable at separate routers without communication)

#### PPO Update Step
```python
# For each episode batch:
for epoch in range(EPOCHS_PER_UPDATE):
    for mini_batch in batches:
        # Actor update
        log_probs, entropies = actors.evaluate(obs_batch, action_batch)
        ratio = exp(log_probs - old_log_probs)
        policy_loss = -clip(ratio, 1-eps, 1+eps) * advantage_batch
        actor_loss = policy_loss - entropy_coef * entropies

        # Critic update
        value_pred = critic(joint_obs_batch)
        value_loss = (value_pred - returns_batch)²

        # Gradient step
        actor_optim.step(actor_loss)
        critic_optim.step(value_loss)
```

### Network Architecture

#### Actor (per agent)
```
Input: local obs (STATE_DIM = 5 or 7)
  ↓
Linear(STATE_DIM → HIDDEN_DIM=128) + Tanh
  ↓
Linear(HIDDEN_DIM → HIDDEN_DIM=128) + Tanh
  ↓
Linear(HIDDEN_DIM → ACTION_DIM=6 or 10)
Output: logits → Categorical distribution
```

#### Critic (shared, centralized)
```
Input: joint obs (JOINT_DIM = 15 or 21)
  ↓
Linear(JOINT_DIM → HIDDEN_DIM=128) + Tanh
  ↓
Linear(HIDDEN_DIM → HIDDEN_DIM=128) + Tanh
  ↓
Linear(HIDDEN_DIM → 1)
Output: scalar value V(joint_obs)
```

### Training Loop Structure

```python
for episode in range(N_EPISODES=500):
    # Reset environment to Monday 08:56 AM
    obs = env.reset()

    # Rollout one full episode (Monday 08:56 → 17:00)
    for window_t in range(~966 windows):
        # Step all 3 agents
        actions = [actor_i.act(obs[i]) for i in 0..2]

        # Environment step (30-second window)
        obs_next, reward, done = env.step(actions)

        # Store transition in RolloutBuffer
        buffer.push(obs, actions, reward, done, obs_next)

        obs = obs_next

    # PPO update every UPDATE_EVERY=10 episodes
    if episode % UPDATE_EVERY == 0:
        # Compute GAE advantages
        advantages = gae(buffer.rewards, buffer.values)
        returns = advantages + buffer.values

        # Normalize advantages
        advantages = (advantages - mean) / (std + eps)

        # Mini-batch PPO epochs
        ppo_update(buffer, advantages, returns)

    # Checkpoint every CHECKPOINT_EVERY=50 episodes
    if episode % 50 == 0:
        save_checkpoint(actors, critic, episode)
```

### Hyperparameters (All in `config.py`)

```python
# Environment
WINDOW_SIZE    = 30           # seconds per observation window
Us             = 1e8          # server capacity (bytes/s)
IP_TIMEOUT     = 300          # IP table entry expiration (sec)
BASE_THRESHOLD = 0.35         # P(attack) for IP table entry

# MARL Training
N_EPISODES     = 500
GAMMA          = 0.99         # discount factor
GAE_LAMBDA     = 0.95         # GAE advantage smoothing
LR_ACTOR       = 3e-4
LR_CRITIC      = 1e-3
CLIP_EPS       = 0.2          # PPO clipping range
ENTROPY_COEF   = 0.01         # exploration bonus
UPDATE_EVERY   = 10           # episodes between PPO updates
EPOCHS_PER_UPDATE = 4         # PPO passes per update
BATCH_SIZE     = 64           # mini-batch size
HIDDEN_DIM     = 128          # actor/critic hidden layer
MAX_GRAD_NORM  = 0.5          # gradient clipping
```

---

## 6. EVALUATION FRAMEWORK

### Metrics

```python
# Detection metrics
TPR = true_positive_rate = attacks_blocked / total_attacks
FPR = false_positive_rate = benign_blocked / total_benign

# Preservation metric
Legit_Ratio = benign_passed / total_benign  ∈ [0, 1]

# Combined F1
Precision = attacks_blocked / (attacks_blocked + benign_blocked)
F1 = 2 * Precision * TPR / (Precision + TPR)

# Training metric
Cumulative_Reward = sum of rewards over episode
```

### Test Scenarios

| Scenario | Data | Purpose |
|----------|------|---------|
| **Seen Attacks** | Monday held-out (last 20%) | Detect known attack patterns |
| **Unseen Attacks** | Tuesday (full day) | Generalization to 5 new attack types |
| **Ablation S1–S12** | Monday + Tuesday | Contribution of each component |

---

## 7. ABLATION STUDY (Stages S1–S12)

Progressive complexity to isolate each contribution:

| Stage | Agents | State | Action | Algorithm | Purpose |
|-------|--------|-------|--------|-----------|---------|
| **S1** | 1 | load only | blind throttle | — | Malialis baseline |
| **S2** | 1 | 5 raw features | blind throttle | — | Richer state value |
| **S3** | 1 | LightGBM P(attack) | blind throttle | — | Supervised risk score |
| **S4** | 1 | LightGBM | selective IP block | — | IP targeting value |
| **S5** | 1 | LightGBM | selective + puzzle | — | Puzzle mechanism value |
| **S6** | 3 | LightGBM | selective + puzzle | IQL | Multi-agent value |
| **S7** | 3 | LightGBM | selective + puzzle | IPPO (local critics) | Coordination without CTDE |
| **S8** | 3 | LightGBM | selective + puzzle | MAPPO + CTDE | Full system (baseline) |
| **S9** | 3 | LightGBM | selective + puzzle | MAPPO + local reward | Local vs global reward |
| **S10 (E13)** | 3 | LightGBM + extended (7-dim) | selective + puzzle | MAPPO + CTDE | Extended state features |
| **S11** | 3 | LightGBM | selective + 10-action | MAPPO + CTDE | Fine-grained throttle |
| **S12** | 3 | LightGBM | selective (no puzzle) | MAPPO + CTDE | Puzzle necessity |

---

## 8. CODEBASE STRUCTURE

```
Research_project/
│
├── BCC-2024/                      ← Data (READ ONLY)
│   ├── Thursday_14_Dec_2023.csv
│   ├── Saturday_16_Dec_2023.csv
│   ├── Monday_18_Dec_2023.csv
│   └── Tuesday_19_Dec_2023.csv
│
├── Risk_Model/                    ← Phase 1: LightGBM Training (✅ DONE)
│   ├── train_eval_save_lgbm.py    ← Main training script
│   ├── eval_lgbm_daily_split.py   ← Evaluation on day splits
│   ├── compute_window_risk_stats.py ← Verify Monday/Tuesday separation
│   ├── lgbm_runtime_infer.py      ← Inference demo
│   └── artifacts/
│       └── vae/v1/
│           ├── selected_features.json    ← 42 feature names
│           └── models/
│               ├── lgbm_bundle_monday.pkl ← PRIMARY (train on Monday)
│               └── lgbm_bundle.pkl        ← OUTDATED (random split)
│
├── src/                           ← Phase 2: MARL (IN PROGRESS)
│   ├── __init__.py
│   ├── environment.py             ← RouterEnvironment + MultiAgentEnv
│   │                              ← Houses simulation logic
│   ├── mappo.py                   ← Actor + Critic + RolloutBuffer + MAPPO
│   │                              ← Reinforcement learning algorithm
│   ├── train_marl.py              ← Main training loop (entry point)
│   └── evaluate.py                ← Evaluation + ablation runs
│
├── artifacts/                     ← Outputs
│   ├── marl/v1/
│   │   ├── actor_r1.pt, actor_r2.pt, actor_r3.pt  ← Saved policies
│   │   ├── critic.pt                              ← Saved value function
│   │   ├── best/                                  ← Best episode checkpoint
│   │   └── training_log.json                      ← Learning curves
│   │
│   └── evaluation/v1/
│       ├── monday_results.json    ← Seen attack performance
│       ├── tuesday_results.json   ← Unseen attack performance
│       └── ablation_results.json  ← S1–S12 comparison
│
├── config.py                      ← ALL hyperparameters & paths
├── main.py                        ← Pipeline orchestration (--stage flag)
├── requirements.txt               ← Python dependencies
└── ARCHI.md                       ← This file
```

### Key Files Explained

| File | Responsibility |
|------|-----------------|
| [config.py](config.py) | Central source of truth for hyperparameters, paths, constants |
| [src/environment.py](src/environment.py) | RouterEnvironment (single router), MultiAgentEnv (3 routers), state/action handling |
| [src/mappo.py](src/mappo.py) | Actor, Critic networks, RolloutBuffer, MAPPO training algorithm |
| [src/train_marl.py](src/train_marl.py) | Main training loop, logging, checkpointing |
| [src/evaluate.py](src/evaluate.py) | Evaluation on Monday/Tuesday, ablation runs |
| [main.py](main.py) | Entry point, `--stage` flag for pipeline control |

---

## 9. DATA FLOW DIAGRAM

```
                                  ┌──────────────────┐
                                  │   BCC-2024 CSVs  │
                                  │ (Mon, Tues data) │
                                  └────────┬─────────┘
                                           │
                                           ↓
                                  ┌──────────────────────────┐
                                  │   Load flows + assign    │
                                  │   router (hash(src_ip))  │
                                  └────────┬─────────────────┘
                                           │
                                           ↓
                                  ┌──────────────────────────┐
                                  │  LightGBM Bundle         │
                                  │  (lgbm_bundle_monday)    │
                                  └────────┬─────────────────┘
                                           │
                    ┌──────────────────────┴───────────────────────┐
                    ↓                                               ↓
        ┌────────────────────┐                        ┌────────────────────────┐
        │  RouterEnvironment  │                        │  MultiAgentEnv         │
        │  (single router)    │                        │  (3 routers + sync)    │
        └────────────────────┘                        └────────────────────────┘
                    ↑                                               ↑
                    │                                               │
        ┌────────────────────┐                        ┌────────────────────────┐
        │ For i in [0..2]:    │                        │   MAIN environment for │
        │ - Compute obs       │←──────Step per agent──→│   MARL training        │
        │ - Apply action      │                        │   - Window iteration   │
        │ - Update IP table   │                        │   - Episode rollout    │
        │ - Return obs', r, d │                        │   - RolloutBuffer push │
        └────────────────────┘                        └────────────────────────┘
                    ↑                                               ↑
                    │                                               │
        ┌────────────────────┐                        ┌────────────────────────┐
        │  Actor (per router) │                        │  RolloutBuffer         │
        │  π(action | obs_i)  │←───────Sample action──→  {obs, acts, rews,     │
        │                     │                        │   dones, values, ...}  │
        └────────────────────┘                        └────────────────────────┘
                                                              ↓
                                                      ┌──────────────────┐
                                                      │ GAE computation  │
                                                      │ (advantages &    │
                                                      │ returns)         │
                                                      └────────┬─────────┘
                                                              ↓
                                        ┌──────────────────────────────────┐
                                        │  PPO Update (every UPDATE_EVERY) │
                                        │  - Actor policy gradient         │
                                        │  - Critic value loss             │
                                        │  - Clipped objective             │
                                        └────────┬───────────────────────┘
                                                 ↓
                                        ┌─────────────────────┐
                                        │ Save checkpoint     │
                                        │ best_actor_X.pt     │
                                        │ best_critic.pt      │
                                        │ training_log.json   │
                                        └─────────────────────┘
```

---

## 10. KNOWN ISSUES & FIXES

### Issue 1: Timestamp Conversion Bug (CRITICAL)
**Symptom**: "Approx windows per episode: 307,502,498" (should be ~966)

**Root Cause**: pandas 3.0 breaking change — `astype(np.int64)` on datetime returns microseconds, not nanoseconds.

**Fix** (2 locations in `src/environment.py`):
```python
# WRONG
ts_sec = pd.to_datetime(df["timestamp"]).astype(np.int64).to_numpy() / 1e9

# CORRECT
_epoch = pd.Timestamp("1970-01-01")
ts_sec = (
    (pd.to_datetime(df["timestamp"]) - _epoch)
    .dt.total_seconds()
    .to_numpy()
    .astype(np.float64)
)
```

### Issue 2: sklearn UserWarning (Non-Critical)
**Symptom**: "X does not have valid feature names" on every LightGBM prediction

**Fix**: Already suppressed at top of `environment.py`:
```python
warnings.filterwarnings("ignore", category=UserWarning,
                        message="X does not have valid feature names")
```

---

## 11. TRAINING TIMELINE

```
Total Training Time: ~2–3 hours (vectorized, 500 episodes)
One Episode: ~20 seconds (full Monday replay)
Windows per Episode: 966 (30s windows, 08:56–17:00)

Expected Progress:
  Episode 0–100:    Reward curve volatile (exploration)
  Episode 100–300:  Convergence toward optimal policy
  Episode 300–500:  Fine-tuning, potential plateau
```

---

## 12. EXECUTION COMMANDS

### Training from Scratch
```bash
cd /Users/abdenourbelarbi/Documents/MultiAgents/MARL_Cybersecurity/Research_project
source my_env/bin/activate
python main.py --stage marl
```

### Evaluation (Post-Training)
```bash
python main.py --stage eval
```

### Ablation Studies
```bash
python main.py --stage ablation --ablation_stage S1
python main.py --stage ablation --ablation_stage S8
# (etc. for S2–S12)
```

---

## 13. INTEGRATION POINTS

### With Risk Model
- **Inputs**: 42 features from raw flows (precomputed in CSV)
- **Outputs**: P(attack), P(benign), P(suspicious) probabilities
- **How**: Loaded once at env init from `lgbm_bundle_monday.pkl`

### With Dataset
- **Reads**: Monday (training), Tuesday (evaluation)
- **Ignores**: Thursday, Saturday (used only for LightGBM pre-training)
- **Window Loop**: Synchronously replay flows in 30-second windows

### With Ablation Framework
- **Config Toggles**: `EXTENDED_STATE`, `LOCAL_REWARD`, `PUZZLE_ENABLED`
- **Stage Names**: S1–S12 correspond to different config combinations
- **Results**: Stored in `artifacts/evaluation/v1/ablation_results.json`

---

## 14. KEY DESIGN DECISIONS

| Decision | Rationale |
|----------|-----------|
| **CTDE** | Critic learns joint patterns; actors remain deployable separately |
| **Shared Global Reward** | DDoS mitigation is network-wide problem; local rewards create conflicting incentives |
| **IP Risk Table** | Realistic: routers track suspicious sources; decay prevents stale entries |
| **Puzzle Mechanism** | Directly motivated by dataset's "suspicious" label; matches real systems (Cloudflare) |
| **Hash-Based Router Assignment** | Permanent routing: same source IP always uses same ingress path |
| **30-Second Windows** | Balance between state granularity and computational cost |
| **LightGBM Risk Scores** | Interpretable probabilities; supervised learning ground truth from Monday labels |

---

## 15. FUTURE EXTENSIONS

- **Communication**: Add explicit multi-agent communication (message passing)
- **Temporal Attention**: Use RNN/Transformer for time-series state
- **Curriculum Learning**: Start with single attack type, gradually increase complexity
- **Real Deployment**: Integrate with actual router firmware (OSPF notifications, BGP updates)

---

**Document Version**: v1
**Last Updated**: 2026-03-21
**Contact**: abdenourbelarbi@gmail.com
