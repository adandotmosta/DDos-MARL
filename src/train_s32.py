"""
train_s32.py
IPPO + p_risk tiered profiles + Monday-only LightGBM + 1-dim state (S32).

Identical to S29 EXCEPT: state reduced to [p_risk] only (1-dim).
p_risk = 1 - p_benign, extracted from the 4-dim state at index 1.

Purpose: test whether the full 4-dim state adds anything over a single
risk score. If S32 ≈ S29, agents were ignoring p_attack, p_suspicious,
and flow_count — they reduced the state to a threshold on p_risk anyway.

This closes the loop with the static baseline:
  Static rule:       fixed p_risk threshold, no learning
  S32 (1-dim state): learned p_risk threshold, 1 feature
  S29 (4-dim state): learned policy, 4 features

If all three produce similar Tuesday TPR → the multi-dimensional state
adds nothing. The agents learn what static rules do, automatically.

Saves to: artifacts/marl/v1/ablation/S32/

Usage:
    python main.py --stage s32
    python src/train_s32.py
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
        CSV_MONDAY, LGBM_BUNDLE,
        MARL_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )
    from src.environment_tiered_v2 import MultiAgentEnv, ACTION_DIM
    from src.ippo import IPPO
except ImportError:
    from coopids.config import (
        CSV_MONDAY, LGBM_BUNDLE,
        MARL_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )
    from coopids.src.environment_tiered_v2 import MultiAgentEnv, ACTION_DIM
    from coopids.src.ippo import IPPO

STATE_DIM_S32 = 1   # only p_risk = 1 - p_benign
_S32_DIR = MARL_DIR / "ablation" / "S32"


def to_prisk_state(local_states: list) -> list:
    """Extract p_risk = 1 - p_benign from 4-dim state [p_atk, p_ben, p_sus, flow]."""
    return [np.array([1.0 - s[1]], dtype=np.float32) for s in local_states]


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
    _S32_DIR.mkdir(parents=True, exist_ok=True)

    if not LGBM_BUNDLE.exists():
        raise FileNotFoundError(f"Bundle not found: {LGBM_BUNDLE}")

    print("S32 config: IPPO + p_risk TIERED PROFILES + 1-DIM STATE [p_risk only]")
    print("  State:     1-dim [p_risk = 1 - p_benign]  (vs S29: 4-dim)")
    print("  Algorithm: IPPO — identical to S29")
    print("  Question:  Does 4-dim state add anything over p_risk alone?")
    print(f"  State dim: {STATE_DIM_S32}, Action dim: {ACTION_DIM}")

    bundle   = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]

    print(f"\nLoading Monday data...")
    df_monday = load_day(CSV_MONDAY, features)
    print(f"  Monday: {len(df_monday):,} flows")

    env  = MultiAgentEnv(df=df_monday, bundle=bundle,
                         n_agents=N_AGENTS, window_size=WINDOW_SIZE)
    ippo = IPPO(state_dim=STATE_DIM_S32, action_dim=ACTION_DIM, n_agents=N_AGENTS)

    log         = []
    best_reward = -np.inf
    N_EPISODES_S32 = 250   # 1-dim state converges fast

    print(f"\nStarting training for {N_EPISODES_S32} episodes...\n")
    print(f"{'Episode':>8} {'Reward':>10} {'Steps':>7} "
          f"{'Actor L':>9} {'Critic L':>9} {'Entropy':>9} {'Time':>7}")
    print("-" * 65)

    for episode in range(1, N_EPISODES_S32 + 1):
        t0 = time.time()

        local_states_full, _ = env.reset()
        local_states = to_prisk_state(local_states_full)

        ep_reward = 0.0
        ep_steps  = 0
        done      = False

        while not done:
            actions, log_probs, values = ippo.act(local_states)
            next_full, _, rewards, done = env.step(actions)
            mean_reward = float(np.mean(rewards))

            ippo.store(
                local_obs=local_states,
                actions=actions,
                log_probs=log_probs,
                reward=mean_reward,
                done=done,
                values=values,
            )

            local_states = to_prisk_state(next_full)
            ep_reward   += mean_reward
            ep_steps    += 1

        update_info = {}
        if episode % UPDATE_EVERY == 0 and len(ippo.buffers[0]) > 0:
            _, _, last_values = ippo.act(local_states)
            update_info = ippo.update(last_values=last_values)
            ippo.clear_buffers()

        elapsed = time.time() - t0
        entry = {
            "episode":     episode,
            "reward":      round(ep_reward, 4),
            "steps":       ep_steps,
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
            ippo.save(_S32_DIR)
            with open(_S32_DIR / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)

        if ep_reward > best_reward:
            best_reward = ep_reward
            ippo.save(_S32_DIR / "best")

    ippo.save(_S32_DIR)
    with open(_S32_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining complete. Best reward: {best_reward:.4f}")
    print(f"Models saved to: {_S32_DIR}")


if __name__ == "__main__":
    train()
