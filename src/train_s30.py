"""
train_s30.py
MAPPO + p_risk tiered profiles + Combined LightGBM (Monday+Tuesday) (S30).

Identical to S27 EXCEPT the LightGBM bundle is trained on Monday+Tuesday attacks.

Purpose: isolate whether S27's Monday→Tuesday performance gap is caused by
sensor degradation (LightGBM never saw Tuesday attacks) or by a fundamental
policy limitation (MARL converges to near-fixed policies regardless).

Comparison chain:
  S27  (p_risk routing, Monday-only scorer)    ← sensor degraded on Tuesday
  S30  (p_risk routing, combined scorer)       ← sensor sees all attack types

If S30 Tuesday TPR >> S27 Tuesday TPR:
  → The gap was sensor quality. Better LightGBM fixes it.
  → MARL policy is sound; the limitation was the sensor.

If S30 Tuesday TPR ≈ S27 Tuesday TPR:
  → Sensor quality is not the bottleneck.
  → MARL converges to near-fixed policies regardless of scorer quality.
  → The limitation is structural (dataset/routing architecture).

Also compare S30 MARL vs S30 static baseline:
  If S30 MARL >> S30 static → MARL genuinely adds adaptive value with good sensor.
  If S30 MARL ≈ S30 static → MARL adds nothing even with perfect signal.

Routing:  p_risk = 1 - p_benign
State:    4-dim [p_attack, p_benign, p_suspicious, flow_count_norm]
Actions:  passive / light / moderate / firm / aggressive / lockdown
Bundle:   lgbm_bundle_monday_tuesday.pkl (combined — sees Tuesday attacks)
Train:    Monday data only (policy generalisation test stays valid)
Saves to: artifacts/marl/v1/ablation/S30/

Usage:
    python main.py --stage s30
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
        CSV_MONDAY, LGBM_BUNDLE_COMBINED,
        MARL_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )
    from src.environment_tiered_v2 import MultiAgentEnv, STATE_DIM, JOINT_DIM, ACTION_DIM
    from src.mappo import MAPPO, RolloutBuffer
except ImportError:
    from coopids.config import (
        CSV_MONDAY, LGBM_BUNDLE_COMBINED,
        MARL_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )
    from coopids.src.environment_tiered_v2 import MultiAgentEnv, STATE_DIM, JOINT_DIM, ACTION_DIM
    from coopids.src.mappo import MAPPO, RolloutBuffer

_S30_DIR = MARL_DIR / "ablation" / "S30"


def load_bundle(bundle_path: Path) -> dict:
    with open(bundle_path, "rb") as f:
        return pickle.load(f)


def load_monday(csv_path: Path, features: list) -> pd.DataFrame:
    needed = list(dict.fromkeys(
        features + ["label", "src_ip", "dst_ip", "timestamp", "bytes_rate"]
    ))
    all_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    needed = [c for c in needed if c in all_cols]
    df = pd.read_csv(csv_path, usecols=needed, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def train() -> None:
    _S30_DIR.mkdir(parents=True, exist_ok=True)

    if not LGBM_BUNDLE_COMBINED.exists():
        raise FileNotFoundError(
            f"Combined LightGBM bundle not found: {LGBM_BUNDLE_COMBINED}\n"
            "Run: python main.py --stage lgbm_combined"
        )

    print("S30 config: p_RISK TIERED PROFILES + COMBINED LightGBM (Monday+Tuesday)")
    print("  Identical to S27 except LightGBM trained on Monday+Tuesday attacks")
    print("  Purpose: isolate sensor quality vs policy quality")
    print("  Routing signal: p_risk = 1 - p_benign")
    print("  State:          4-dim [p_attack, p_benign, p_suspicious, flow_count_norm]")
    print("  LightGBM:       combined bundle (Monday+Tuesday)")
    print("  Train data:     Monday only (policy generalisation test stays valid)")
    print(f"  State dim: {STATE_DIM}, Joint dim: {JOINT_DIM}, Action dim: {ACTION_DIM}")

    print(f"\nLoading combined LGBM bundle from {LGBM_BUNDLE_COMBINED}...")
    bundle   = load_bundle(LGBM_BUNDLE_COMBINED)
    features = bundle["features"]
    print(f"Features: {len(features)}")

    print(f"\nLoading Monday data from {CSV_MONDAY}...")
    df_monday = load_monday(CSV_MONDAY, features)
    print(f"  Monday: {len(df_monday):,} flows")
    print(df_monday["label"].value_counts().to_string())

    print("\nBuilding MultiAgentEnv (p_risk routing, combined LightGBM)...")
    env = MultiAgentEnv(
        df=df_monday, bundle=bundle,
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
            mappo.save(_S30_DIR)
            with open(_S30_DIR / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)
            print(f"  Checkpoint saved at episode {episode}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            mappo.save(_S30_DIR / "best")

    mappo.save(_S30_DIR)
    with open(_S30_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining complete. Best reward: {best_reward:.4f}")
    print(f"Models saved to: {_S30_DIR}")


if __name__ == "__main__":
    train()
