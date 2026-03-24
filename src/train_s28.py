"""
train_s28.py
MAPPO + p_risk tiered profiles + Monday-only LightGBM + mixed-density training (S28).

Identical to S27 EXCEPT training data = Monday (69% attack) + Thursday (100% benign).

Purpose: fix the Thursday/Saturday FPR problem observed in S27.
In S27, agents were trained only on Monday (attack-heavy) so they never learned
to stand down on benign-only days.  By including Thursday in training, the agent
experiences both regimes:
  - High p_risk windows (Monday attacks)  → be aggressive/firm
  - Low p_risk windows  (Thursday benign) → use passive
The reward function already provides the correct signal: using lockdown on a
benign-only window gives low legit_ratio → low reward → agent learns to relax.

Comparison chain:
  S27  (Monday-only training) → catastrophic FPR on Thursday/Saturday
  S28  (Monday+Thursday training) → same design, mixed-density training

Evaluation:
  Monday   — sanity check (seen attacks, seen benign)
  Tuesday  — primary test (unseen attack types)
  Saturday — KEY test (unseen benign-only day — did stand-down generalise?)
  Thursday — EXCLUDED (seen during training)

Actions:
  0 = passive    (forward all)
  1 = light      (T_block=0.90, T_puzzle=0.60)
  2 = moderate   (T_block=0.80, T_puzzle=0.40)
  3 = firm       (T_block=0.70, T_puzzle=0.25)
  4 = aggressive (T_block=0.60, T_puzzle=0.15)
  5 = lockdown   (T_block=0.50, T_puzzle=0.10)

Routing:  p_risk = 1 - p_benign
State:    4-dim [p_attack, p_benign, p_suspicious, flow_count_norm]
Bundle:   lgbm_bundle_monday.pkl (Monday-only)
Train on: Monday (320,530 flows) + Thursday (105,087 flows)
Saves to: artifacts/marl/v1/ablation/S28/

Usage:
    python main.py --stage s28
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import (
        CSV_MONDAY, CSV_THURSDAY, LGBM_BUNDLE,
        MARL_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )
    from src.environment_tiered_v2 import MultiAgentEnv, STATE_DIM, JOINT_DIM, ACTION_DIM
    from src.mappo import MAPPO, RolloutBuffer
except ImportError:
    from coopids.config import (
        CSV_MONDAY, CSV_THURSDAY, LGBM_BUNDLE,
        MARL_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )
    from coopids.src.environment_tiered_v2 import MultiAgentEnv, STATE_DIM, JOINT_DIM, ACTION_DIM
    from coopids.src.mappo import MAPPO, RolloutBuffer

_S28_DIR = MARL_DIR / "ablation" / "S28"


def load_bundle(bundle_path: Path) -> dict:
    with open(bundle_path, "rb") as f:
        return pickle.load(f)


def load_day(csv_path: Path, features: list) -> pd.DataFrame:
    needed = list(dict.fromkeys(
        features + ["label", "src_ip", "dst_ip", "timestamp", "bytes_rate"]
    ))
    all_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    needed = [c for c in needed if c in all_cols]
    df = pd.read_csv(csv_path, usecols=needed, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def train() -> None:
    _S28_DIR.mkdir(parents=True, exist_ok=True)

    if not LGBM_BUNDLE.exists():
        raise FileNotFoundError(
            f"Monday LightGBM bundle not found: {LGBM_BUNDLE}\n"
            "Run: python Risk_Model/train_eval_save_lgbm.py"
        )

    print("S28 config: p_RISK TIERED PROFILES + MONDAY-ONLY LightGBM + MIXED-DENSITY TRAINING")
    print("  Routing signal: p_risk = 1 - p_benign")
    print("  Training data:  Monday (attack-heavy) + Thursday (benign-only)")
    print("  Fix target:     Thursday/Saturday FPR — agent learns to stand down")
    print("  Evaluation:     Monday (sanity), Tuesday (unseen attacks), Saturday (unseen benign)")
    print(f"  State dim: {STATE_DIM}, Joint dim: {JOINT_DIM}, Action dim: {ACTION_DIM}")

    print(f"\nLoading Monday-only LGBM bundle from {LGBM_BUNDLE}...")
    bundle   = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]
    print(f"Features: {len(features)}")

    print(f"\nLoading Monday data...")
    df_monday = load_day(CSV_MONDAY, features)
    print(f"  Monday: {len(df_monday):,} flows")
    print(df_monday["label"].value_counts().to_string())

    print(f"\nLoading Thursday data (benign-only)...")
    if not CSV_THURSDAY.exists():
        raise FileNotFoundError(f"Thursday CSV not found: {CSV_THURSDAY}")
    df_thursday = load_day(CSV_THURSDAY, features)
    print(f"  Thursday: {len(df_thursday):,} flows")
    print(df_thursday["label"].value_counts().to_string())

    # Retime Thursday to immediately follow Monday (1-minute gap).
    # Without this, Thursday (Dec 14) and Monday (Dec 18) are 4 days apart:
    # the environment would create ~11,500 empty 30-second windows in the gap,
    # wasting 90% of training steps on dead time with no flows.
    monday_end      = df_monday["timestamp"].max()
    thursday_start  = df_thursday["timestamp"].min()
    time_offset     = monday_end - thursday_start + pd.Timedelta(minutes=1)
    df_thursday     = df_thursday.copy()
    df_thursday["timestamp"] = df_thursday["timestamp"] + time_offset
    print(f"  Thursday retimed to start at: {df_thursday['timestamp'].min()}")
    print(f"  (immediately after Monday end: {monday_end})")

    # Combine and sort — agent sees Monday attacks then Thursday benign in each episode
    df_train = pd.concat([df_monday, df_thursday], ignore_index=True)
    df_train = df_train.sort_values("timestamp").reset_index(drop=True)
    print(f"\nCombined training set: {len(df_train):,} flows")
    print(df_train["label"].value_counts().to_string())
    attack_frac = (df_train["label"] == "Attack").mean()
    benign_frac = (df_train["label"] == "Benign").mean()
    print(f"  Attack density: {attack_frac:.1%}  (was 69% Monday-only)")
    print(f"  Benign density: {benign_frac:.1%}  (was 21% Monday-only)")

    span_hours = (df_train["timestamp"].max() - df_train["timestamp"].min()).total_seconds() / 3600
    est_windows = int(span_hours * 3600 / WINDOW_SIZE)
    print(f"  Combined time span: {span_hours:.1f} hours → ~{est_windows} windows/episode")

    print("\nBuilding MultiAgentEnv (p_risk routing, mixed training data)...")
    env = MultiAgentEnv(
        df=df_train, bundle=bundle,
        n_agents=N_AGENTS, window_size=WINDOW_SIZE,
    )
    print(f"Approx windows per episode: {env.n_windows}")

    print(f"\nBuilding MAPPO (state_dim={STATE_DIM}, joint_dim={JOINT_DIM}, action_dim={ACTION_DIM})...")
    mappo  = MAPPO(state_dim=STATE_DIM, joint_dim=JOINT_DIM, action_dim=ACTION_DIM)
    buffer = RolloutBuffer(n_agents=N_AGENTS)
    buffer.clear()

    log         = []
    best_reward = -np.inf

    print(f"\nStarting training for {N_EPISODES} episodes...\n")
    print(f"{'Episode':>8} {'Reward':>10} {'Steps':>7} "
          f"{'Actor L':>9} {'Critic L':>9} {'Entropy':>9} {'Time':>7}")
    print("-" * 65)

    for episode in range(1, N_EPISODES + 1):
        t0 = time.time()
        local_states, joint_obs = env.reset()
        ep_reward = 0.0
        ep_steps  = 0
        done      = False

        while not done:
            actions, log_probs, value = mappo.act(local_states, joint_obs)
            next_local, next_joint, rewards, done = env.step(actions)
            mean_reward = float(np.mean(rewards))
            buffer.store(
                local_obs=local_states, joint_obs=joint_obs,
                actions=actions, log_probs=log_probs,
                reward=mean_reward, done=done, value=value,
            )
            local_states = next_local
            joint_obs    = next_joint
            ep_reward   += mean_reward
            ep_steps    += 1

        update_info = {}
        if episode % UPDATE_EVERY == 0 and len(buffer) > 0:
            _, _, last_val = mappo.act(local_states, joint_obs)
            update_info = mappo.update(buffer, last_value=last_val)
            buffer.clear()

        elapsed = time.time() - t0
        entry = {
            "episode": episode, "reward": round(ep_reward, 4),
            "steps": ep_steps,
            "actor_loss":  round(update_info.get("actor_loss",  0.0), 6),
            "critic_loss": round(update_info.get("critic_loss", 0.0), 6),
            "entropy":     round(update_info.get("entropy",     0.0), 6),
            "time_s":      round(elapsed, 2),
        }
        log.append(entry)

        if episode % 10 == 0 or episode == 1:
            print(f"{episode:>8d} {ep_reward:>10.4f} {ep_steps:>7d} "
                  f"{entry['actor_loss']:>9.6f} {entry['critic_loss']:>9.6f} "
                  f"{entry['entropy']:>9.6f} {elapsed:>6.1f}s")

        if episode % CHECKPOINT_EVERY == 0:
            mappo.save(_S28_DIR)
            with open(_S28_DIR / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)
            print(f"  Checkpoint saved at episode {episode}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            mappo.save(_S28_DIR / "best")

    mappo.save(_S28_DIR)
    with open(_S28_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining complete. Best reward: {best_reward:.4f}")
    print(f"Models saved to: {_S28_DIR}")


if __name__ == "__main__":
    train()
