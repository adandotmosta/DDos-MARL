# CoopIDS — Cooperative MARL for DDoS Mitigation at ISP Edge Routers

Multi-Agent PPO (MAPPO) with Centralized Training / Decentralized Execution (CTDE). Three agents deployed at ISP edge routers learn adaptive per-flow DDoS filtering using a pre-trained LightGBM risk scorer and tiered action profiles.

## Project Structure

```
CoopIDS/
├── config.py                      # All hyperparameters and paths
├── main.py                        # Pipeline entry point
├── requirements.txt
├── Risk_Model/
│   ├── train_eval_save_lgbm.py    # Train the LightGBM risk scorer
│   └── artifacts/vae/v1/
│       ├── models/                # lgbm_bundle_monday.pkl  (gitignored)
│       └── selected_features.json
├── src/
│   ├── mappo.py                   # MAPPO: Actor, Critic, RolloutBuffer
│   ├── ippo.py                    # IPPO (cooperation ablation)
│   ├── iql.py                     # IQL (early baseline)
│   ├── environment_tiered_v2.py   # Main environment — p_risk tiered routing
│   ├── evaluate.py                # Full evaluation across all ablation stages
│   ├── train_s27.py               # Final model
│   ├── train_s29.py               # Ablation: IPPO (no CTDE)
│   ├── train_s30.py               # Ablation: retrained sensor (Mon+Tue)
│   ├── train_s37.py               # Ablation: 6 agents
│   ├── train_s39.py               # Ablation: end-to-end FlowEncoder
│   ├── train_s40.py               # Ablation: global reward
│   └── train_s*.py                # All other ablation stages (S11–S38)
├── scripts/                       # Analysis and utility scripts
└── BCC-2024/                      # Dataset CSVs (gitignored — download separately)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the BCCC-cPacket-Cloud-DDoS-2024 CSV files in `BCC-2024/`.

## Step 1 — Train the Risk Scorer

```bash
python Risk_Model/train_eval_save_lgbm.py \
    --split-mode day \
    --backend lightgbm \
    --model-bundle-out Risk_Model/artifacts/vae/v1/models/lgbm_bundle_monday.pkl
```

Trains a 3-class LightGBM (benign / suspicious / attack) on Monday data only and saves a bundle used by all S27+ experiments.

For the retrained-sensor ablation (S30), also run:

```bash
python scripts/train_lgbm_combined.py
```

## Step 2 — Train the MARL Policy

**Final model — S27 (MAPPO + p\_risk tiered profiles, Monday-only sensor):**

```bash
python main.py --stage s27
```

Weights saved to `artifacts/marl/v1/ablation/S27/`. Training takes ~23 minutes on a single CPU (500 episodes).

Other key stages:

| Command | Stage | Description |
|---------|-------|-------------|
| `python main.py --stage s29` | S29 | IPPO — no CTDE (cooperation ablation) |
| `python main.py --stage s30` | S30 | MAPPO + retrained sensor (Mon+Tue) |
| `python main.py --stage s37` | S37 | MAPPO + 6 agents (scaling ablation) |
| `python main.py --stage s39` | S39 | MAPPO + FlowEncoder (no LightGBM) |
| `python main.py --stage s40` | S40 | MAPPO + global reward (reward ablation) |

## Step 3 — Run Inference / Evaluation

Evaluate trained models on Monday (seen attacks) and Tuesday (unseen, zero-shot):

```bash
python main.py --stage eval --tag s27_results
```

To evaluate specific models only:

```bash
python main.py --stage eval --tag my_run --models s27,s29,s30
```

Results are saved as JSON to `artifacts/evaluation/v1/<tag>/`.

### Output metrics (per model, per day)

| Metric | Description |
|--------|-------------|
| TPR | Fraction of attack flows blocked |
| FPR | Fraction of legitimate flows incorrectly blocked |
| Precision | Of blocked flows, fraction that are actual attacks |
| F1 | Harmonic mean of TPR and Precision |
| Legit ratio | Fraction of legitimate traffic forwarded (= 1 − FPR) |

## Key Results

**S27 — Monday-only sensor, zero-shot Tuesday:**

| Day | TPR | FPR | F1 |
|-----|-----|-----|----|
| Monday (12 seen attack types) | 99.52% | 2.91% | — |
| Tuesday (5 unseen attack types) | 85.78% | 6.01% | 77.14% |

**S30 — Same MARL weights, sensor retrained on Mon+Tue:**

| Day | TPR | FPR | F1 |
|-----|-----|-----|----|
| Tuesday | 94.68% | 1.17% | 93.83% |

The 16.7pp F1 gain (S27→S30) from sensor retraining alone identifies the risk scorer as the generalisation bottleneck, not the RL policy.

## Tiered Action Profiles

Each agent selects one of 6 profiles per 30-second window.
Routing signal: `p_risk = 1 − P(benign)`.

| ID | Profile | T\_block | T\_puzzle |
|----|---------|---------|---------|
| 0 | Passive | 1.00 | 1.00 |
| 1 | Light | 0.90 | 0.60 |
| 2 | Moderate | 0.80 | 0.40 |
| 3 | Firm | 0.70 | 0.25 |
| 4 | Aggressive | 0.60 | 0.15 |
| 5 | Lockdown | 0.50 | 0.10 |

`p_risk ≥ T_block` → drop; `p_risk ≥ T_puzzle` → CAPTCHA (attack bots fail 95%); below T_puzzle → forward.

## Dataset

**BCCC-cPacket-Cloud-DDoS-2024** — real cloud traffic from AWS infrastructure.

| File | Flows | Role |
|------|-------|------|
| `Monday_18_Dec_2023.csv` | 320,530 | Training (12 attack types) |
| `Tuesday_19_Dec_2023.csv` | 85,479 | Zero-shot test (5 unseen types) |
| `Thursday_14_Dec_2023.csv` | 105,087 | Pure benign (mixed-day ablations) |
| `Saturday_16_Dec_2023.csv` | 189,678 | Pure benign |

Download: https://www.unb.ca/cic/datasets/ddos-2024.html
