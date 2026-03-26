"""
train_s37.py
MAPPO + p_risk tiered profiles + Monday-only LightGBM + 6 agents (S37).

Identical to S31 EXCEPT: MAPPO instead of IPPO.
Joint critic sees all 6 agents' 4-dim states → 24-dim joint observation.

Purpose: test whether MAPPO's joint critic helps when agent count increases.
Hypothesis: MAPPO scales better than IPPO with more agents because the joint
critic can learn cross-agent correlations that IPPO misses.

Comparison:
  S29  (3 agents, IPPO)  — baseline
  S31  (6 agents, IPPO)  — more agents, no CTDE
  S37  (6 agents, MAPPO) — more agents, with CTDE

  If S37 > S31: CTDE provides value at higher agent counts
  If S37 ≈ S31: CTDE remains irrelevant regardless of scale
  If S37 < S31: joint critic overfits the 24-dim joint state with 6 agents

Saves to: artifacts/marl/v1/ablation/S37/

Usage:
    python main.py --stage s37
    python src/train_s37.py
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
        CHECKPOINT_EVERY, WINDOW_SIZE,
    )
    from src.environment_tiered_v2 import MultiAgentEnv, STATE_DIM, ACTION_DIM
    from src.mappo import MAPPO, RolloutBuffer
except ImportError:
    from coopids.config import (
        CSV_MONDAY, LGBM_BUNDLE,
        MARL_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, WINDOW_SIZE,
    )
    from coopids.src.environment_tiered_v2 import MultiAgentEnv, STATE_DIM, ACTION_DIM
    from coopids.src.mappo import MAPPO, RolloutBuffer

N_AGENTS_S37  = 6
JOINT_DIM_S37 = STATE_DIM * N_AGENTS_S37   # 4 * 6 = 24
_S37_DIR      = MARL_DIR / "ablation" / "S37"


def load_bundle(bundle_path: Path) -> dict:
    with open(bundle_path, "rb") as f:
        return pickle.load(f)


def load_day(csv_path: Path, features: list) -> pd.DataFrame:
    needed = list(dict.fromkeys(
        features + ["label", "src_ip", "dst_ip", "timestamp", "bytes_rate"]
    ))
    all_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    needed = [c for c in needed if c in all_cols]
    df = pd.read_csv(csv_path, usecols=needed, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def train() -> None:
    _S37_DIR.mkdir(parents=True, exist_ok=True)

    if not LGBM_BUNDLE.exists():
        raise FileNotFoundError(f"Bundle not found: {LGBM_BUNDLE}")

    print("S37 config: MAPPO + p_risk TIERED PROFILES + MONDAY-ONLY LightGBM + 6 AGENTS")
    print("  Algorithm:  MAPPO — joint critic sees 24-dim joint state (4-dim × 6 agents)")
    print("  Agents:     6  (vs S29 baseline of 3, vs S31 IPPO 6-agent)")
    print("  Question:   Does CTDE help when agent count increases?")
    print(f"  State dim: {STATE_DIM}, Joint dim: {JOINT_DIM_S37}, Action dim: {ACTION_DIM}")

    bundle   = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]

    print(f"\nLoading Monday data...")
    df_monday = load_day(CSV_MONDAY, features)
    print(f"  Monday: {len(df_monday):,} flows")
    print(f"  Approx flows per agent per window: "
          f"~{len(df_monday) / N_AGENTS_S37 / (8*3600/WINDOW_SIZE):.0f}")

    env    = MultiAgentEnv(df=df_monday, bundle=bundle,
                           n_agents=N_AGENTS_S37, window_size=WINDOW_SIZE)
    mappo  = MAPPO(state_dim=STATE_DIM, joint_dim=JOINT_DIM_S37,
                   action_dim=ACTION_DIM, n_agents=N_AGENTS_S37)
    buffer = RolloutBuffer(n_agents=N_AGENTS_S37)
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
            print(f"{episode:>8d} {ep_reward:>10.4f} {ep_steps:>7d} "
                  f"{entry['actor_loss']:>9.6f} {entry['critic_loss']:>9.6f} "
                  f"{entry['entropy']:>9.6f} {elapsed:>6.1f}s")

        if episode % CHECKPOINT_EVERY == 0:
            mappo.save(_S37_DIR)
            with open(_S37_DIR / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)

        if ep_reward > best_reward:
            best_reward = ep_reward
            mappo.save(_S37_DIR / "best")

    mappo.save(_S37_DIR)
    with open(_S37_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining complete. Best reward: {best_reward:.4f}")
    print(f"Models saved to: {_S37_DIR}")


if __name__ == "__main__":
    train()
