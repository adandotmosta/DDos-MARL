"""
train_s23.py
MAPPO + selective puzzle (S23).

Your idea: puzzle only challenges flows that LightGBM predicts as Suspicious.
  - Benign-predicted flows  → pass_rate = 1.0 (no challenge needed)
  - Suspicious-predicted flows → pass_rate = 0.5 (genuine uncertainty, verify)
  - Attack-predicted flows  → pass_rate = 1.0 (puzzle useless — must use throttle/block)

This fundamentally breaks puzzle dominance: for high-confidence attack windows
puzzle does nothing, so the agent is forced to learn throttle/block to get any TPR.
Puzzle effectiveness now depends on LightGBM quality (good on Monday, degrades Tuesday)
which is the correct real-world behavior.

State:     5-dim baseline (same as S9)
Actions:   6 (same as S9)
Reward:    local per-agent (same as S9)
Saves to:  artifacts/marl/v1/ablation/S23/

Usage:
    python main.py --stage s23
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
    import src.environment as _env_module
    from src.environment import MultiAgentEnv
    from src.mappo import MAPPO, RolloutBuffer
except ImportError:
    from coopids.config import (
        CSV_MONDAY, LGBM_BUNDLE,
        MARL_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )
    import coopids.src.environment as _env_module
    from coopids.src.environment import MultiAgentEnv
    from coopids.src.mappo import MAPPO, RolloutBuffer

_S23_DIR = MARL_DIR / "ablation" / "S23"


def load_bundle(bundle_path: Path) -> dict:
    with open(bundle_path, "rb") as f:
        return pickle.load(f)


def load_monday(csv_path: Path, features: list) -> pd.DataFrame:
    needed = features + ["label", "src_ip", "dst_ip", "timestamp", "bytes_rate"]
    needed = list(dict.fromkeys(needed))
    df = pd.read_csv(csv_path, usecols=needed, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Monday loaded: {df.shape}")
    print(df["label"].value_counts().to_string())
    return df


def train() -> None:
    _S23_DIR.mkdir(parents=True, exist_ok=True)

    # Patch environment: selective puzzle + local reward
    _env_module.LOCAL_REWARD      = True
    _env_module.EXTENDED_STATE    = False
    _env_module.PUZZLE_ENABLED    = True
    _env_module.PUZZLE_COST       = 0.0
    _env_module.SELECTIVE_PUZZLE  = True

    print("S23 config: SELECTIVE_PUZZLE=True, LOCAL_REWARD=True")
    print("  Puzzle only challenges Suspicious-predicted flows.")
    print("  Attack-predicted flows must be stopped by throttle/block.")

    print(f"Loading LGBM bundle from {LGBM_BUNDLE}...")
    bundle = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]
    print(f"Features: {len(features)}")

    print(f"\nLoading Monday data from {CSV_MONDAY}...")
    df_monday = load_monday(CSV_MONDAY, features)

    print("\nBuilding MultiAgentEnv (selective puzzle)...")
    env = MultiAgentEnv(
        df=df_monday, bundle=bundle,
        n_agents=N_AGENTS, window_size=WINDOW_SIZE,
    )
    print(f"Approx windows per episode: {env.n_windows}")

    print("\nBuilding MAPPO (state_dim=5, joint_dim=15, action_dim=10)...")
    mappo  = MAPPO(state_dim=5, joint_dim=15, action_dim=10)
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
            mappo.save(_S23_DIR)
            with open(_S23_DIR / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)
            print(f"  Checkpoint saved at episode {episode}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            mappo.save(_S23_DIR / "best")

    # Restore default
    _env_module.SELECTIVE_PUZZLE = False

    mappo.save(_S23_DIR)
    with open(_S23_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining complete. Best reward: {best_reward:.4f}")
    print(f"Models saved to: {_S23_DIR}")


if __name__ == "__main__":
    train()
