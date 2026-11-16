"""
train_s42.py
MAPPO + SupCon encoder state + LightGBM p_risk enforcement (S42).

State:      7-dim  [mean(z_0..z_5), flow_count_norm]
            z = frozen SupCon embedding (contrastive pre-training on CICIDS)

Enforcement: unchanged — LightGBM p_risk vs (T_block, T_puzzle).

Question: does a contrastively-trained embedding produce a better state
than raw features (S41) or LightGBM probabilities (S27)?

Saves to: artifacts/marl/v1/ablation/S42/

Usage:
    python src/train_s42.py
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CSV_MONDAY, LGBM_BUNDLE,
    MARL_DIR, N_EPISODES, UPDATE_EVERY,
    CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
)
from src.environment_tiered_v6 import MultiAgentEnv, STATE_DIM, JOINT_DIM, ACTION_DIM
from src.mappo import MAPPO, RolloutBuffer

_S42_DIR = MARL_DIR / "ablation" / "S42"


def load_bundle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_monday(csv_path: Path, features: list) -> pd.DataFrame:
    needed = list(dict.fromkeys(
        features + ["label", "src_ip", "dst_ip", "timestamp"]
    ))
    df = pd.read_csv(csv_path, usecols=needed, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def train() -> None:
    _S42_DIR.mkdir(parents=True, exist_ok=True)

    print("S42: MAPPO + SUPCON ENCODER STATE + LightGBM enforcement")
    print("  State: 7-dim SupCon embedding (frozen, pre-trained on CICIDS)")
    print("  Enforcement: LightGBM p_risk — identical to S27/S41")
    print(f"  STATE_DIM={STATE_DIM}  JOINT_DIM={JOINT_DIM}  ACTION_DIM={ACTION_DIM}\n")

    bundle   = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]

    print("Loading Monday data …")
    df = load_monday(CSV_MONDAY, features)
    print(f"  {len(df):,} flows\n")

    env    = MultiAgentEnv(df=df, bundle=bundle, n_agents=N_AGENTS, window_size=WINDOW_SIZE)
    mappo  = MAPPO(state_dim=STATE_DIM, joint_dim=JOINT_DIM,
                   action_dim=ACTION_DIM, n_agents=N_AGENTS)
    buffer = RolloutBuffer(n_agents=N_AGENTS)

    log         = []
    best_reward = -np.inf

    print(f"Starting training — {N_EPISODES} episodes\n")
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

            r = float(np.mean(reward)) if isinstance(reward, list) else float(reward)
            buffer.store(
                local_obs=local_states, joint_obs=joint_obs,
                actions=actions, log_probs=log_probs,
                reward=r, done=done, value=value,
            )
            local_states = next_local
            joint_obs    = next_joint
            ep_reward   += r
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
            mappo.save(_S42_DIR)
            with open(_S42_DIR / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)

        if ep_reward > best_reward:
            best_reward = ep_reward
            mappo.save(_S42_DIR / "best")

    mappo.save(_S42_DIR)
    with open(_S42_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining complete. Best reward: {best_reward:.4f}")
    print(f"Models saved to: {_S42_DIR}")


if __name__ == "__main__":
    train()
