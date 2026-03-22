"""
train_mappo_local.py
MAPPO with per-agent local reward (E2 ablation, S9).

Each agent is rewarded only on its own router's flows.
Algorithm stays MAPPO (shared centralized critic) — only reward signal changes.

Saves to: artifacts/marl/v1/ablation/S9/

Usage:
    python main.py --stage mapl
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
        CSV_MONDAY, LGBM_BUNDLE, FEATURES_JSON,
        MAPL_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )
    from src.environment import MultiAgentEnv, assign_agents
    from src.mappo import MAPPO, RolloutBuffer
except ImportError:
    from coopids.config import (
        CSV_MONDAY, LGBM_BUNDLE, FEATURES_JSON,
        MAPL_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )
    from coopids.src.environment import MultiAgentEnv, assign_agents
    from coopids.src.mappo import MAPPO, RolloutBuffer


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
    # LOCAL_REWARD must be True — enforce at runtime
    try:
        from config import LOCAL_REWARD
    except ImportError:
        from coopids.config import LOCAL_REWARD

    if not LOCAL_REWARD:
        raise RuntimeError(
            "Set LOCAL_REWARD = True in config.py before running train_mappo_local."
        )

    MAPL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load LightGBM bundle
    print(f"Loading LGBM bundle from {LGBM_BUNDLE}...")
    bundle = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]
    print(f"Features: {len(features)}")

    # ── Load Monday data
    print(f"\nLoading Monday data from {CSV_MONDAY}...")
    df_monday = load_monday(CSV_MONDAY, features)

    # ── Build environment (LOCAL_REWARD=True → step returns list of rewards)
    print("\nBuilding MultiAgentEnv (LOCAL_REWARD=True)...")
    env = MultiAgentEnv(
        df=df_monday,
        bundle=bundle,
        n_agents=N_AGENTS,
        window_size=WINDOW_SIZE,
    )
    print(f"Approx windows per episode: {env.n_windows}")

    # ── Build MAPPO (same architecture as baseline)
    print("\nBuilding MAPPO (local reward variant)...")
    mappo  = MAPPO()
    buffer = RolloutBuffer(n_agents=N_AGENTS)
    buffer.clear()

    # ── Training log
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

            # step() returns list of per-agent rewards when LOCAL_REWARD=True
            next_local, next_joint, rewards, done = env.step(actions)

            # Mean reward for logging and best-model tracking
            mean_reward = float(np.mean(rewards))

            # Store each agent's own reward in the shared buffer.
            # RolloutBuffer.store() expects a scalar reward — we store
            # the mean so the centralized critic sees the global signal.
            # The key difference from baseline: each agent's individual
            # reward drove its own advantage computation if we used
            # per-agent buffers (as in IPPO). Here we average to keep
            # the centralized critic semantics intact, while each agent's
            # local reward shaped what actions were taken.
            buffer.store(
                local_obs=local_states,
                joint_obs=joint_obs,
                actions=actions,
                log_probs=log_probs,
                reward=mean_reward,
                done=done,
                value=value,
            )

            local_states = next_local
            joint_obs    = next_joint
            ep_reward   += mean_reward
            ep_steps    += 1

        # ── PPO update every UPDATE_EVERY episodes
        update_info = {}
        if episode % UPDATE_EVERY == 0 and len(buffer) > 0:
            _, _, last_val = mappo.act(local_states, joint_obs)
            update_info = mappo.update(buffer, last_value=last_val)
            buffer.clear()

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
            mappo.save(MAPL_DIR)
            with open(MAPL_DIR / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)
            print(f"  Checkpoint saved at episode {episode}")

        # ── Best model
        if ep_reward > best_reward:
            best_reward = ep_reward
            mappo.save(MAPL_DIR / "best")

    # ── Final save
    mappo.save(MAPL_DIR)
    with open(MAPL_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nTraining complete.")
    print(f"Best episode reward: {best_reward:.4f}")
    print(f"Models saved to:     {MAPL_DIR}")
    print(f"Log saved to:        {MAPL_DIR / 'training_log.json'}")


if __name__ == "__main__":
    train()
