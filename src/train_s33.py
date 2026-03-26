"""
train_s33.py
MAPPO + p_risk tiered profiles + shared IP blocklist (S33).

State: 5-dim [p_attack, p_benign, p_suspicious, flow_count_norm, shared_threat_level]
  shared_threat_level = total flagged IPs across all routers / 500

Joint state (CTDE critic): 15-dim (5 × 3 agents)

This is the first experiment where CTDE coordination could genuinely help:
  - Each agent sees the aggregate IP flagging rate across the whole system
  - The joint critic can learn: "when agents 1+2 are both flagging many IPs,
    agent 3 should pre-emptively escalate even before its own subnet is hit"
  - IPPO (S34) sees the same 5-dim state but has no joint critic

Comparison:
  S29 (IPPO, 4-dim, no shared signal) → baseline
  S33 (MAPPO, 5-dim, shared signal)   → does CTDE exploit the shared signal?
  S34 (IPPO, 5-dim, shared signal)    → does shared signal alone help?

  If S33 > S34 > S29 → CTDE + shared signal = genuine coordination
  If S33 ≈ S34 > S29 → shared signal helps, but CTDE not needed
  If S33 ≈ S34 ≈ S29 → shared signal adds nothing

Saves to: artifacts/marl/v1/ablation/S33/

Usage:
    python main.py --stage s33
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
    from src.environment_tiered_v3 import MultiAgentEnvV3, STATE_DIM, JOINT_DIM, ACTION_DIM
    from src.mappo import MAPPO, RolloutBuffer
except ImportError:
    from coopids.config import (
        CSV_MONDAY, LGBM_BUNDLE,
        MARL_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )
    from coopids.src.environment_tiered_v3 import MultiAgentEnvV3, STATE_DIM, JOINT_DIM, ACTION_DIM
    from coopids.src.mappo import MAPPO, RolloutBuffer

_S33_DIR = MARL_DIR / "ablation" / "S33"


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
    _S33_DIR.mkdir(parents=True, exist_ok=True)

    if not LGBM_BUNDLE.exists():
        raise FileNotFoundError(f"Bundle not found: {LGBM_BUNDLE}")

    print("S33: MAPPO + SHARED IP BLOCKLIST + 5-DIM STATE")
    print("  State:       5-dim [p_atk, p_ben, p_sus, flow_count, shared_threat]")
    print("  Joint state: 15-dim (CTDE critic sees all agents)")
    print("  Question:    Does CTDE exploit shared threat signal for coordination?")
    print(f"  State dim: {STATE_DIM}, Joint dim: {JOINT_DIM}, Action dim: {ACTION_DIM}")

    bundle   = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]

    print("\nLoading Monday data...")
    df = load_day(CSV_MONDAY, features)
    print(f"  {len(df):,} flows")

    env    = MultiAgentEnvV3(df=df, bundle=bundle, n_agents=N_AGENTS,
                              window_size=WINDOW_SIZE)
    mappo  = MAPPO(state_dim=STATE_DIM, joint_dim=JOINT_DIM, action_dim=ACTION_DIM)
    buffer = RolloutBuffer(n_agents=N_AGENTS)
    buffer.clear()

    log         = []
    best_reward = -np.inf

    print(f"\nTraining for {N_EPISODES} episodes...")
    print(f"{'Episode':>8} {'Reward':>10} {'Steps':>7} "
          f"{'Actor L':>9} {'Critic L':>9} {'Entropy':>9} {'Time':>7}")
    print("-" * 65)

    for episode in range(1, N_EPISODES + 1):
        t0 = time.time()
        local_states, joint_obs = env.reset()
        ep_reward = ep_steps = 0
        done = False

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
            mappo.save(_S33_DIR)
            with open(_S33_DIR / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)

        if ep_reward > best_reward:
            best_reward = ep_reward
            mappo.save(_S33_DIR / "best")

    mappo.save(_S33_DIR)
    with open(_S33_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone. Best reward: {best_reward:.4f}")
    print(f"Saved to: {_S33_DIR}")


if __name__ == "__main__":
    train()
