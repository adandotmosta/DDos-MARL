"""
train_s29.py
IPPO + p_risk tiered profiles + Monday-only LightGBM (S29).

Identical to S27 EXCEPT: replaces MAPPO (centralised critic) with IPPO
(independent local critics, no joint observation).

Purpose: isolate whether MAPPO's centralised critic / joint observation is
contributing to performance, or whether independent per-agent adaptation
alone explains S27's results.

Comparison:
  S27 (MAPPO): centralised critic sees joint obs (all agents concatenated)
               → cooperation between agents is possible
  S29 (IPPO):  each agent has its own local critic, sees only its own state
               → no cooperation, purely independent adaptation

If S29 ≈ S27:  joint observations add nothing — independent adaptation explains S27
If S29 < S27:  centralised critic / CTDE is genuinely contributing to performance
Also compare action distributions per router:
  If routers use equally diverse tiers in S29 as S27 → diversity is independent
  If S29 routers converge to the same tier   → diversity in S27 came from cooperation

Everything else is identical to S27:
  Routing:  p_risk = 1 - p_benign
  State:    4-dim [p_attack, p_benign, p_suspicious, flow_count_norm]
  Actions:  passive / light / moderate / firm / aggressive / lockdown
  Bundle:   lgbm_bundle_monday.pkl (Monday-only)
  Train:    Monday data only
  Saves to: artifacts/marl/v1/ablation/S29/

Usage:
    python main.py --stage s29
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
    from src.environment_tiered_v2 import MultiAgentEnv, STATE_DIM, ACTION_DIM
    from src.ippo import IPPO
except ImportError:
    from coopids.config import (
        CSV_MONDAY, LGBM_BUNDLE,
        MARL_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )
    from coopids.src.environment_tiered_v2 import MultiAgentEnv, STATE_DIM, ACTION_DIM
    from coopids.src.ippo import IPPO

_S29_DIR = MARL_DIR / "ablation" / "S29"


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
    _S29_DIR.mkdir(parents=True, exist_ok=True)

    if not LGBM_BUNDLE.exists():
        raise FileNotFoundError(
            f"Monday LightGBM bundle not found: {LGBM_BUNDLE}\n"
            "Run: python Risk_Model/train_eval_save_lgbm.py"
        )

    print("S29 config: IPPO (no cooperation) + p_risk TIERED PROFILES + MONDAY-ONLY LightGBM")
    print("  Algorithm:      IPPO — each agent has local critic only, NO joint observation")
    print("  Comparison:     S27 (MAPPO) vs S29 (IPPO) — isolates CTDE contribution")
    print("  Routing signal: p_risk = 1 - p_benign")
    print("  State:          4-dim [p_attack, p_benign, p_suspicious, flow_count_norm]")
    print("  LightGBM:       Monday-only (identical to S27)")
    print(f"  State dim: {STATE_DIM}, Action dim: {ACTION_DIM}")

    print(f"\nLoading Monday-only LGBM bundle from {LGBM_BUNDLE}...")
    bundle   = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]
    print(f"Features: {len(features)}")

    print(f"\nLoading Monday data...")
    df_monday = load_day(CSV_MONDAY, features)
    print(f"  Monday: {len(df_monday):,} flows")
    print(df_monday["label"].value_counts().to_string())

    print("\nBuilding MultiAgentEnv (p_risk routing, Monday-only LightGBM)...")
    env = MultiAgentEnv(
        df=df_monday, bundle=bundle,
        n_agents=N_AGENTS, window_size=WINDOW_SIZE,
    )
    print(f"Approx windows per episode: {env.n_windows}")

    print(f"\nBuilding IPPO (state_dim={STATE_DIM}, action_dim={ACTION_DIM}, n_agents={N_AGENTS})...")
    ippo = IPPO(state_dim=STATE_DIM, action_dim=ACTION_DIM, n_agents=N_AGENTS)

    log         = []
    best_reward = -np.inf

    print(f"\nStarting training for {N_EPISODES} episodes...\n")
    print(f"{'Episode':>8} {'Reward':>10} {'Steps':>7} "
          f"{'Actor L':>9} {'Critic L':>9} {'Entropy':>9} {'Time':>7}")
    print("-" * 65)

    for episode in range(1, N_EPISODES + 1):
        t0 = time.time()

        local_states, _ = env.reset()   # joint_obs unused — IPPO is fully local
        ep_reward = 0.0
        ep_steps  = 0
        done      = False

        while not done:
            actions, log_probs, values = ippo.act(local_states)
            next_local, _, rewards, done = env.step(actions)
            mean_reward = float(np.mean(rewards))

            ippo.store(
                local_obs=local_states,
                actions=actions,
                log_probs=log_probs,
                reward=mean_reward,
                done=done,
                values=values,
            )

            local_states = next_local
            ep_reward   += mean_reward
            ep_steps    += 1

        update_info = {}
        if episode % UPDATE_EVERY == 0 and len(ippo.buffers[0]) > 0:
            _, _, last_values = ippo.act(local_states)
            update_info = ippo.update(last_values=last_values)
            ippo.clear_buffers()

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
            ippo.save(_S29_DIR)
            with open(_S29_DIR / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)
            print(f"  Checkpoint saved at episode {episode}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            ippo.save(_S29_DIR / "best")

    ippo.save(_S29_DIR)
    with open(_S29_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining complete. Best reward: {best_reward:.4f}")
    print(f"Models saved to: {_S29_DIR}")


if __name__ == "__main__":
    train()
