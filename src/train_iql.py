"""
train_iql.py
IQL training loop for CoopIDS ablation study (S6).

Trains 3 independent DQN agents on Monday data.
Saves to artifacts/marl/v1/ablation/S6/ for comparison with MAPPO (S8).

Usage:
    python src/train_iql.py
    python main.py --stage iql
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
        MARL_DIR, N_EPISODES, N_AGENTS, WINDOW_SIZE,
        CHECKPOINT_EVERY,
    )
    from src.environment import MultiAgentEnv, assign_agents
    from src.iql import IQL, TARGET_UPDATE
except ImportError:
    from coopids.config import (
        CSV_MONDAY, LGBM_BUNDLE,
        MARL_DIR, N_EPISODES, N_AGENTS, WINDOW_SIZE,
        CHECKPOINT_EVERY,
    )
    from coopids.src.environment import MultiAgentEnv, assign_agents
    from coopids.src.iql import IQL, TARGET_UPDATE

# IQL saves to the ablation S6 directory
IQL_DIR = MARL_DIR / "ablation" / "S6"


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
    IQL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load bundle and data
    print(f"Loading LGBM bundle from {LGBM_BUNDLE}...")
    bundle   = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]
    print(f"Features: {len(features)}")

    print(f"\nLoading Monday data from {CSV_MONDAY}...")
    df_monday = load_monday(CSV_MONDAY, features)

    # ── Build environment
    print("\nBuilding MultiAgentEnv...")
    env = MultiAgentEnv(
        df         = df_monday,
        bundle     = bundle,
        n_agents   = N_AGENTS,
        window_size= WINDOW_SIZE,
    )
    print(f"Approx windows per episode: {env.n_windows}")

    # ── Build IQL
    print("\nBuilding IQL (3 independent DQN agents)...")
    iql = IQL()

    # ── Training log
    log         = []
    best_reward = -np.inf

    print(f"\nStarting IQL training for {N_EPISODES} episodes...\n")
    print(f"{'Episode':>8} {'Reward':>10} {'Steps':>7} "
          f"{'Q Loss':>9} {'Epsilon':>9} {'Time':>7}")
    print("-" * 58)

    for episode in range(1, N_EPISODES + 1):
        t0 = time.time()

        local_states, _ = env.reset()
        ep_reward = 0.0
        ep_steps  = 0
        done      = False
        ep_loss   = 0.0
        n_updates = 0

        while not done:
            # Epsilon-greedy action selection
            actions = iql.act(local_states, greedy=False)

            # Step environment
            next_local, _, reward, done = env.step(actions)

            # Store in replay buffers
            iql.store(local_states, actions, reward, next_local, done)

            # Update Q-networks every step (DQN style)
            info = iql.update()
            if info:
                ep_loss   += info["q_loss"]
                n_updates += 1

            local_states  = next_local
            ep_reward    += reward
            ep_steps     += 1

        # Decay epsilon after each episode
        iql.decay_epsilon()

        # Hard update target networks every TARGET_UPDATE episodes
        if episode % TARGET_UPDATE == 0:
            iql.sync_targets()

        # ── Logging
        avg_loss = ep_loss / max(n_updates, 1)
        elapsed  = time.time() - t0
        entry = {
            "episode":  episode,
            "reward":   round(ep_reward, 4),
            "steps":    ep_steps,
            "q_loss":   round(avg_loss, 6),
            "epsilon":  round(iql.epsilon, 4),
            "time_s":   round(elapsed, 2),
        }
        log.append(entry)

        if episode % 10 == 0 or episode == 1:
            print(
                f"{episode:>8d} {ep_reward:>10.4f} {ep_steps:>7d} "
                f"{avg_loss:>9.6f} {iql.epsilon:>9.4f} {elapsed:>6.1f}s"
            )

        # ── Checkpoint
        if episode % CHECKPOINT_EVERY == 0:
            iql.save(IQL_DIR)
            with open(IQL_DIR / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)
            print(f"  ✓ Checkpoint saved at episode {episode}")

        # ── Best model
        if ep_reward > best_reward:
            best_reward = ep_reward
            iql.save(IQL_DIR / "best")

    # ── Final save
    iql.save(IQL_DIR)
    with open(IQL_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nIQL Training complete.")
    print(f"Best episode reward: {best_reward:.4f}")
    print(f"Models saved to:     {IQL_DIR}")
    print(f"Log saved to:        {IQL_DIR / 'training_log.json'}")


if __name__ == "__main__":
    train()
