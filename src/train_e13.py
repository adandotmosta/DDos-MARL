"""
train_e13.py
MAPPO with extended 7-dim state (E13 / ablation S10).

New state features added to baseline 5-dim:
  - unique_src_ips_norm : fraction of distinct source IPs (DDoS signature)
  - attack_ip_ratio     : fraction of flows from already-flagged IPs

Saves to: artifacts/marl/v1/ablation/S10/

Usage:
    python main.py --stage e13
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
        E13_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )
    from src.environment import MultiAgentEnv
    from src.mappo import MAPPO, RolloutBuffer
except ImportError:
    from coopids.config import (
        CSV_MONDAY, LGBM_BUNDLE,
        E13_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )
    from coopids.src.environment import MultiAgentEnv
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
    # EXTENDED_STATE must be True — enforce at runtime
    try:
        from config import EXTENDED_STATE
    except ImportError:
        from coopids.config import EXTENDED_STATE

    if not EXTENDED_STATE:
        raise RuntimeError(
            "Set EXTENDED_STATE = True in config.py before running train_e13."
        )

    E13_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading LGBM bundle from {LGBM_BUNDLE}...")
    bundle = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]
    print(f"Features: {len(features)}")

    print(f"\nLoading Monday data from {CSV_MONDAY}...")
    df_monday = load_monday(CSV_MONDAY, features)

    print("\nBuilding MultiAgentEnv (EXTENDED_STATE=True, STATE_DIM=7)...")
    env = MultiAgentEnv(
        df=df_monday,
        bundle=bundle,
        n_agents=N_AGENTS,
        window_size=WINDOW_SIZE,
    )
    print(f"Approx windows per episode: {env.n_windows}")

    print("\nBuilding MAPPO (7-dim state)...")
    mappo  = MAPPO()
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
            next_local, next_joint, reward, done = env.step(actions)

            buffer.store(
                local_obs=local_states,
                joint_obs=joint_obs,
                actions=actions,
                log_probs=log_probs,
                reward=reward,
                done=done,
                value=value,
            )

            local_states = next_local
            joint_obs    = next_joint
            ep_reward   += reward
            ep_steps    += 1

        update_info = {}
        if episode % UPDATE_EVERY == 0 and len(buffer) > 0:
            _, _, last_val = mappo.act(local_states, joint_obs)
            update_info = mappo.update(buffer, last_value=last_val)
            buffer.clear()

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

        if episode % CHECKPOINT_EVERY == 0:
            mappo.save(E13_DIR)
            with open(E13_DIR / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)
            print(f"  Checkpoint saved at episode {episode}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            mappo.save(E13_DIR / "best")

    mappo.save(E13_DIR)
    with open(E13_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nTraining complete.")
    print(f"Best episode reward: {best_reward:.4f}")
    print(f"Models saved to:     {E13_DIR}")
    print(f"Log saved to:        {E13_DIR / 'training_log.json'}")


if __name__ == "__main__":
    train()
