"""
train_s34.py
IPPO + p_risk tiered profiles + shared IP blocklist (S34).

Identical to S33 EXCEPT: IPPO instead of MAPPO (no joint critic).

Each agent sees the same 5-dim state including shared_threat_level,
but has its own local critic — no centralized coordination.

Purpose: isolate whether CTDE adds value when shared signal exists.
  S33 (MAPPO): joint critic can coordinate responses to shared signal
  S34 (IPPO):  independent critics, agents use shared signal individually

  If S33 > S34 → CTDE + shared signal = genuine coordination
  If S33 ≈ S34 → independent use of shared signal is sufficient
  Both comparisons vs S29 (4-dim, no shared signal) show whether
  the shared threat level itself improves detection.

Saves to: artifacts/marl/v1/ablation/S34/

Usage:
    python main.py --stage s34
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
    from src.environment_tiered_v3 import MultiAgentEnvV3, STATE_DIM, ACTION_DIM
    from src.ippo import IPPO
except ImportError:
    from coopids.config import (
        CSV_MONDAY, LGBM_BUNDLE,
        MARL_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )
    from coopids.src.environment_tiered_v3 import MultiAgentEnvV3, STATE_DIM, ACTION_DIM
    from coopids.src.ippo import IPPO

_S34_DIR = MARL_DIR / "ablation" / "S34"


def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_day(csv_path, features):
    needed = list(dict.fromkeys(
        features + ["label", "src_ip", "dst_ip", "timestamp", "bytes_rate"]
    ))
    all_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    needed = [c for c in needed if c in all_cols]
    df = pd.read_csv(csv_path, usecols=needed, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def train():
    _S34_DIR.mkdir(parents=True, exist_ok=True)

    if not LGBM_BUNDLE.exists():
        raise FileNotFoundError(f"Bundle not found: {LGBM_BUNDLE}")

    print("S34: IPPO + SHARED IP BLOCKLIST + 5-DIM STATE (no CTDE)")
    print("  State:     5-dim [p_atk, p_ben, p_sus, flow_count, shared_threat]")
    print("  Algorithm: IPPO — local critics only, NO joint observation")
    print("  Question:  Does shared signal help without CTDE coordination?")
    print(f"  State dim: {STATE_DIM}, Action dim: {ACTION_DIM}")

    bundle   = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]

    print("\nLoading Monday data...")
    df = load_day(CSV_MONDAY, features)
    print(f"  {len(df):,} flows")

    env  = MultiAgentEnvV3(df=df, bundle=bundle, n_agents=N_AGENTS,
                            window_size=WINDOW_SIZE)
    ippo = IPPO(state_dim=STATE_DIM, action_dim=ACTION_DIM, n_agents=N_AGENTS)

    log         = []
    best_reward = -np.inf

    print(f"\nTraining for {N_EPISODES} episodes...")
    print(f"{'Episode':>8} {'Reward':>10} {'Steps':>7} "
          f"{'Actor L':>9} {'Critic L':>9} {'Entropy':>9} {'Time':>7}")
    print("-" * 65)

    for episode in range(1, N_EPISODES + 1):
        t0 = time.time()
        local_states, _ = env.reset()
        ep_reward = ep_steps = 0
        done = False

        while not done:
            actions, log_probs, values = ippo.act(local_states)
            next_local, _, rewards, done = env.step(actions)
            mean_reward = float(np.mean(rewards))
            ippo.store(
                local_obs=local_states, actions=actions,
                log_probs=log_probs, reward=mean_reward,
                done=done, values=values,
            )
            local_states = next_local
            ep_reward   += mean_reward
            ep_steps    += 1

        update_info = {}
        if episode % UPDATE_EVERY == 0 and len(ippo.buffers[0]) > 0:
            _, _, last_values = ippo.act(local_states)
            update_info = ippo.update(last_values=last_values)
            ippo.clear_buffers()

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
            ippo.save(_S34_DIR)
            with open(_S34_DIR / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)

        if ep_reward > best_reward:
            best_reward = ep_reward
            ippo.save(_S34_DIR / "best")

    ippo.save(_S34_DIR)
    with open(_S34_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone. Best reward: {best_reward:.4f}")
    print(f"Saved to: {_S34_DIR}")


if __name__ == "__main__":
    train()
