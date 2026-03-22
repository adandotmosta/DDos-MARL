"""
train_s12.py
MAPPO without puzzle action (9-action space, no CAPTCHA challenge).

Action space (no puzzle, compact IDs 0–8):
  0  forward
  1  throttle_10  (10% drop)
  2  throttle_20  (20% drop)
  3  throttle_30  (30% drop)
  4  throttle_50  (50% drop)
  5  throttle_70  (70% drop)
  6  throttle_90  (90% drop)
  7  throttle_95  (95% drop)
  8  block        (100% drop)

Puzzle is disabled via PUZZLE_ENABLED=False patch on the environment module.
Action ID 1 therefore acts as throttle_10 (first entry in DROP_PROBS).

State: 5-dim baseline (same as S8).
Saves to: artifacts/marl/v1/ablation/S12/

Usage:
    python main.py --stage s12
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
        S12_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )
    import src.environment as _env_module
    from src.environment import MultiAgentEnv
    from src.mappo import MAPPO, RolloutBuffer
except ImportError:
    from coopids.config import (
        CSV_MONDAY, LGBM_BUNDLE,
        S12_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )
    import coopids.src.environment as _env_module
    from coopids.src.environment import MultiAgentEnv
    from coopids.src.mappo import MAPPO, RolloutBuffer

# S12 local constants — not taken from config to avoid global mutation
_S12_DROP_PROBS = {
    1: 0.10,  # throttle_10
    2: 0.20,  # throttle_20
    3: 0.30,  # throttle_30
    4: 0.50,  # throttle_50
    5: 0.70,  # throttle_70
    6: 0.90,  # throttle_90
    7: 0.95,  # throttle_95
    8: 1.00,  # block
}
_S12_ACTION_DIM = 9
_S12_STATE_DIM  = 5
_S12_JOINT_DIM  = 15  # STATE_DIM * N_AGENTS


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
    S12_DIR.mkdir(parents=True, exist_ok=True)

    # Patch environment module: disable puzzle, use S12 action space
    _env_module.PUZZLE_ENABLED = False
    _env_module.EXTENDED_STATE = False
    _env_module.DROP_PROBS     = _S12_DROP_PROBS

    print(f"Loading LGBM bundle from {LGBM_BUNDLE}...")
    bundle = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]
    print(f"Features: {len(features)}")

    print(f"\nLoading Monday data from {CSV_MONDAY}...")
    df_monday = load_monday(CSV_MONDAY, features)

    print("\nBuilding MultiAgentEnv (no puzzle, ACTION_DIM=9)...")
    env = MultiAgentEnv(
        df=df_monday,
        bundle=bundle,
        n_agents=N_AGENTS,
        window_size=WINDOW_SIZE,
    )
    print(f"Approx windows per episode: {env.n_windows}")

    print("\nBuilding MAPPO (state_dim=5, joint_dim=15, action_dim=9)...")
    mappo  = MAPPO(state_dim=_S12_STATE_DIM, joint_dim=_S12_JOINT_DIM, action_dim=_S12_ACTION_DIM)
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
            mappo.save(S12_DIR)
            with open(S12_DIR / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)
            print(f"  Checkpoint saved at episode {episode}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            mappo.save(S12_DIR / "best")

    mappo.save(S12_DIR)
    with open(S12_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nTraining complete.")
    print(f"Best episode reward: {best_reward:.4f}")
    print(f"Models saved to:     {S12_DIR}")
    print(f"Log saved to:        {S12_DIR / 'training_log.json'}")


if __name__ == "__main__":
    train()
