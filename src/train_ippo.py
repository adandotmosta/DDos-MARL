"""
train_ippo.py
IPPO training loop for CoopIDS ablation study (S7).

Trains 3 independent PPO agents on Monday data.
Each agent has its own local actor + local critic — no centralized critic.
Saves to artifacts/marl/v1/ablation/S7/ for comparison with MAPPO (S8) and IQL (S6).

Usage:
    python src/train_ippo.py
    python main.py --stage ippo
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
        CHECKPOINT_EVERY, UPDATE_EVERY,
    )
    from src.environment import MultiAgentEnv
    from src.ippo import IPPO
except ImportError:
    from coopids.config import (
        CSV_MONDAY, LGBM_BUNDLE,
        MARL_DIR, N_EPISODES, N_AGENTS, WINDOW_SIZE,
        CHECKPOINT_EVERY, UPDATE_EVERY,
    )
    from coopids.src.environment import MultiAgentEnv
    from coopids.src.ippo import IPPO

IPPO_DIR = MARL_DIR / "ablation" / "S7"


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
    IPPO_DIR.mkdir(parents=True, exist_ok=True)

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
        df          = df_monday,
        bundle      = bundle,
        n_agents    = N_AGENTS,
        window_size = WINDOW_SIZE,
    )
    print(f"Approx windows per episode: {env.n_windows}")

    # ── Build IPPO
    print("\nBuilding IPPO (3 independent PPO agents, local critics)...")
    ippo = IPPO()

    # ── Training log
    log         = []
    best_reward = -np.inf

    print(f"\nStarting IPPO training for {N_EPISODES} episodes...\n")
    print(f"{'Episode':>8} {'Reward':>10} {'Steps':>7} "
          f"{'Actor L':>9} {'Critic L':>9} {'Entropy':>9} {'Time':>7}")
    print("-" * 65)

    for episode in range(1, N_EPISODES + 1):
        t0 = time.time()

        local_states, _ = env.reset()
        ep_reward = 0.0
        ep_steps  = 0
        done      = False

        while not done:
            # Act — local obs only, returns per-agent values
            actions, log_probs, values = ippo.act(local_states)

            # Step environment
            next_local, _, reward, done = env.step(actions)

            # Store in each agent's buffer
            ippo.store(
                local_obs = local_states,
                actions   = actions,
                log_probs = log_probs,
                reward    = reward,
                done      = done,
                values    = values,
            )

            local_states  = next_local
            ep_reward    += reward
            ep_steps     += 1

        # ── PPO update every UPDATE_EVERY episodes
        update_info = {}
        if episode % UPDATE_EVERY == 0 and len(ippo.buffers[0]) > 0:
            # Per-agent bootstrap values from last local state
            _, _, last_values = ippo.act(local_states)
            update_info = ippo.update(last_values)
            ippo.clear_buffers()

        # ── Logging
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
            print(
                f"{episode:>8d} {ep_reward:>10.4f} {ep_steps:>7d} "
                f"{entry['actor_loss']:>9.6f} {entry['critic_loss']:>9.6f} "
                f"{entry['entropy']:>9.6f} {elapsed:>6.1f}s"
            )

        # ── Checkpoint
        if episode % CHECKPOINT_EVERY == 0:
            ippo.save(IPPO_DIR)
            with open(IPPO_DIR / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)
            print(f"  ✓ Checkpoint saved at episode {episode}")

        # ── Best model
        if ep_reward > best_reward:
            best_reward = ep_reward
            ippo.save(IPPO_DIR / "best")

    # ── Final save
    ippo.save(IPPO_DIR)
    with open(IPPO_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nIPPO Training complete.")
    print(f"Best episode reward: {best_reward:.4f}")
    print(f"Models saved to:     {IPPO_DIR}")
    print(f"Log saved to:        {IPPO_DIR / 'training_log.json'}")


if __name__ == "__main__":
    train()
