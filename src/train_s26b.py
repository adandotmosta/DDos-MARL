"""
train_s26b.py
MAPPO + tiered action profiles + Monday-only LightGBM (S26b).

Identical to S26 except uses lgbm_bundle_monday.pkl instead of the combined bundle.
Purpose: isolate the MARL contribution from the LightGBM detector quality.
  - If S26b ≈ S26 → MARL tiered profiles are genuinely contributing.
  - If S26b collapses → combined LightGBM was doing most of the work.

Actions:
  0 = passive    (forward all)
  1 = light      (T_block=0.85, T_puzzle=0.35)
  2 = moderate   (T_block=0.60, T_puzzle=0.20)
  3 = firm       (T_block=0.35, T_puzzle=0.10)
  4 = aggressive (T_block=0.20, T_puzzle=0.05)
  5 = lockdown   (T_block=0.10, T_puzzle=0.02)

State:     4-dim [p_attack, p_benign, p_suspicious, flow_count_norm]
Actions:   6
Reward:    local per-agent
Bundle:    lgbm_bundle_monday.pkl (Monday-only)
Saves to:  artifacts/marl/v1/ablation/S26b/

Usage:
    python main.py --stage s26b
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
    from src.environment_tiered import MultiAgentEnv, STATE_DIM, JOINT_DIM, ACTION_DIM
    from src.mappo import MAPPO, RolloutBuffer
except ImportError:
    from coopids.config import (
        CSV_MONDAY, LGBM_BUNDLE,
        MARL_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )
    from coopids.src.environment_tiered import MultiAgentEnv, STATE_DIM, JOINT_DIM, ACTION_DIM
    from coopids.src.mappo import MAPPO, RolloutBuffer

_S26B_DIR = MARL_DIR / "ablation" / "S26b"


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
    _S26B_DIR.mkdir(parents=True, exist_ok=True)

    if not LGBM_BUNDLE.exists():
        raise FileNotFoundError(
            f"Monday LightGBM bundle not found: {LGBM_BUNDLE}\n"
            "Run: python Risk_Model/train_eval_save_lgbm.py"
        )

    print("S26b config: TIERED ACTION PROFILES + MONDAY-ONLY LightGBM")
    print("  Purpose: isolate MARL contribution from LightGBM detector quality")
    print("  Actions: passive / light / moderate / firm / aggressive / lockdown")
    print("  Per-flow routing by p_attack → block / puzzle / forward")
    print("  No IP risk table, no throttle, no oracle routing")
    print("  LightGBM: trained on Monday ONLY (same as S9 baseline)")
    print("  MARL trained on: Monday data only")
    print(f"  State dim: {STATE_DIM}, Joint dim: {JOINT_DIM}, Action dim: {ACTION_DIM}")

    print(f"\nLoading Monday-only LGBM bundle from {LGBM_BUNDLE}...")
    bundle = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]
    print(f"Features: {len(features)}")

    print(f"\nLoading Monday data from {CSV_MONDAY}...")
    df_monday = load_monday(CSV_MONDAY, features)

    print("\nBuilding MultiAgentEnv (tiered profiles, Monday-only LightGBM)...")
    env = MultiAgentEnv(
        df=df_monday, bundle=bundle,
        n_agents=N_AGENTS, window_size=WINDOW_SIZE,
    )
    print(f"Approx windows per episode: {env.n_windows}")

    print(f"\nBuilding MAPPO (state_dim={STATE_DIM}, joint_dim={JOINT_DIM}, action_dim={ACTION_DIM})...")
    mappo  = MAPPO(state_dim=STATE_DIM, joint_dim=JOINT_DIM, action_dim=ACTION_DIM)
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
            mappo.save(_S26B_DIR)
            with open(_S26B_DIR / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)
            print(f"  Checkpoint saved at episode {episode}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            mappo.save(_S26B_DIR / "best")

    mappo.save(_S26B_DIR)
    with open(_S26B_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining complete. Best reward: {best_reward:.4f}")
    print(f"Models saved to: {_S26B_DIR}")


if __name__ == "__main__":
    train()
