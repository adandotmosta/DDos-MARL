"""
train_s18.py
MAPPO + local reward + curriculum learning (S18).

Two-phase training:
  Phase 1 (ep 1–250):  PUZZLE_ENABLED=False → agents learn throttle/block
  Phase 2 (ep 251–500): PUZZLE_ENABLED=True  → agents add puzzle to repertoire

Agents develop a non-puzzle foundation first, then learn to use puzzle
as a supplement rather than a crutch.

State:     5-dim baseline (same as S9)
Actions:   6 (same as S9; action 1 is no-op in Phase 1)
Reward:    local per-agent (same as S9)
Saves to:  artifacts/marl/v1/ablation/S18/

Usage:
    python main.py --stage s18
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

_S18_DIR = MARL_DIR / "ablation" / "S18"
_PHASE1_EPISODES = 250  # no puzzle
_PHASE2_EPISODES = 250  # puzzle enabled


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
    _S18_DIR.mkdir(parents=True, exist_ok=True)

    # Patch environment: local reward (puzzle toggled per phase)
    _env_module.LOCAL_REWARD    = True
    _env_module.EXTENDED_STATE  = False
    _env_module.PUZZLE_COST     = 0.0

    print(f"S18 config: LOCAL_REWARD=True, curriculum learning")
    print(f"  Phase 1 (ep 1–{_PHASE1_EPISODES}): PUZZLE_ENABLED=False")
    print(f"  Phase 2 (ep {_PHASE1_EPISODES+1}–{N_EPISODES}): PUZZLE_ENABLED=True")

    print(f"Loading LGBM bundle from {LGBM_BUNDLE}...")
    bundle = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]
    print(f"Features: {len(features)}")

    print(f"\nLoading Monday data from {CSV_MONDAY}...")
    df_monday = load_monday(CSV_MONDAY, features)

    print("\nBuilding MultiAgentEnv (local reward, curriculum)...")
    env = MultiAgentEnv(
        df=df_monday, bundle=bundle,
        n_agents=N_AGENTS, window_size=WINDOW_SIZE,
    )
    print(f"Approx windows per episode: {env.n_windows}")

    print("\nBuilding MAPPO (state_dim=5, joint_dim=15, action_dim=6)...")
    mappo  = MAPPO(state_dim=5, joint_dim=15, action_dim=6)
    buffer = RolloutBuffer(n_agents=N_AGENTS)
    buffer.clear()

    log         = []
    best_reward = -np.inf

    print(f"\nStarting training for {N_EPISODES} episodes...\n")
    print(f"{'Episode':>8} {'Reward':>10} {'Steps':>7} "
          f"{'Actor L':>9} {'Critic L':>9} {'Entropy':>9} {'Time':>7} {'Phase':>7}")
    print("-" * 73)

    for episode in range(1, N_EPISODES + 1):
        # Toggle puzzle per phase
        if episode <= _PHASE1_EPISODES:
            _env_module.PUZZLE_ENABLED = False
            phase = "no-pzl"
        else:
            _env_module.PUZZLE_ENABLED = True
            phase = "puzzle"

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
            "steps": ep_steps, "phase": phase,
            "actor_loss":  round(update_info.get("actor_loss",  0.0), 6),
            "critic_loss": round(update_info.get("critic_loss", 0.0), 6),
            "entropy":     round(update_info.get("entropy",     0.0), 6),
            "time_s":      round(elapsed, 2),
        }
        log.append(entry)

        if episode % 10 == 0 or episode == 1 or episode == _PHASE1_EPISODES + 1:
            print(f"{episode:>8d} {ep_reward:>10.4f} {ep_steps:>7d} "
                  f"{entry['actor_loss']:>9.6f} {entry['critic_loss']:>9.6f} "
                  f"{entry['entropy']:>9.6f} {elapsed:>6.1f}s {phase:>7}")

        if episode % CHECKPOINT_EVERY == 0:
            mappo.save(_S18_DIR)
            with open(_S18_DIR / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)
            print(f"  Checkpoint saved at episode {episode}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            mappo.save(_S18_DIR / "best")

    # Ensure puzzle is enabled for final save (evaluation expects it)
    _env_module.PUZZLE_ENABLED = True

    mappo.save(_S18_DIR)
    with open(_S18_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining complete. Best reward: {best_reward:.4f}")
    print(f"Models saved to: {_S18_DIR}")


if __name__ == "__main__":
    train()
