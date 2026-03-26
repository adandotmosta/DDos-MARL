"""
train_s35.py
MAPPO + victim-based routing + shared IP blocklist (S35).

The key experiment: does CTDE provide genuine value when routing creates
asymmetric threat distributions across routers?

With victim routing, at any given 30s window:
  - One router may be under heavy attack (target victim)
  - Other routers see only benign background traffic
  - shared_threat_level lets idle routers observe the active attack
  - MAPPO's joint critic can learn: "R0 is flooding → R1/R2 should escalate"
  - IPPO (S36) sees same signal but cannot coordinate the joint response

This is the structural condition that was absent in all prior experiments.
Hash routing (S27-S34) always produced symmetric conditions → no coordination need.
Victim routing (S35/S36) produces asymmetric conditions → coordination could emerge.

Comparison:
  S33 MAPPO (hash routing + shared signal):   coordination signal, symmetric
  S34 IPPO  (hash routing + shared signal):   coordination signal, symmetric
  S35 MAPPO (victim routing + shared signal): coordination signal, ASYMMETRIC
  S36 IPPO  (victim routing + shared signal): coordination signal, ASYMMETRIC

  If S35 > S36: CTDE genuinely useful under asymmetric routing
  If S35 ≈ S36: independent adaptation sufficient even with asymmetry
  Both vs S33/S34: effect of routing structure on coordination need

Saves to: artifacts/marl/v1/ablation/S35/
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
    from src.environment_tiered_v4 import MultiAgentEnvV4, STATE_DIM, JOINT_DIM, ACTION_DIM
    from src.mappo import MAPPO, RolloutBuffer
except ImportError:
    from coopids.config import (
        CSV_MONDAY, LGBM_BUNDLE,
        MARL_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
    )
    from coopids.src.environment_tiered_v4 import MultiAgentEnvV4, STATE_DIM, JOINT_DIM, ACTION_DIM
    from coopids.src.mappo import MAPPO, RolloutBuffer

_S35_DIR = MARL_DIR / "ablation" / "S35"


def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_day(csv_path, features):
    needed = list(dict.fromkeys(
        features + ["label", "src_ip", "dst_ip", "timestamp", "bytes_rate"]
    ))
    all_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    needed = [c for c in needed if c in all_cols]
    df = pd.read_csv(csv_path, usecols=needed, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def train():
    _S35_DIR.mkdir(parents=True, exist_ok=True)

    if not LGBM_BUNDLE.exists():
        raise FileNotFoundError(f"Bundle not found: {LGBM_BUNDLE}")

    print("S35: MAPPO + VICTIM-BASED ROUTING + SHARED IP BLOCKLIST")
    print("  Routing:     dst_ip → victim → router (asymmetric threat distribution)")
    print("  State:       5-dim [p_atk, p_ben, p_sus, flow_count, shared_threat]")
    print("  Algorithm:   MAPPO — joint critic sees all 15-dim joint state")
    print("  Question:    Does CTDE beat IPPO under asymmetric routing conditions?")
    print(f"  State dim: {STATE_DIM}, Joint dim: {JOINT_DIM}, Action dim: {ACTION_DIM}")

    bundle   = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]

    print("\nLoading Monday data...")
    df = load_day(CSV_MONDAY, features)
    print(f"  {len(df):,} flows")

    # Show routing distribution
    from src.environment_tiered_v4 import assign_agents_victim
    df_routed = assign_agents_victim(df, N_AGENTS)
    for i in range(N_AGENTS):
        agent_df = df_routed[df_routed["agent"] == i]
        atk = (agent_df["label"] == "Attack").sum()
        ben = (agent_df["label"] == "Benign").sum()
        print(f"  Router {i}: {len(agent_df):,} flows "
              f"({atk:,} attack, {ben:,} benign)")

    env    = MultiAgentEnvV4(df=df, bundle=bundle, n_agents=N_AGENTS,
                              window_size=WINDOW_SIZE)
    mappo  = MAPPO(state_dim=STATE_DIM, joint_dim=JOINT_DIM, action_dim=ACTION_DIM)
    buffer = RolloutBuffer(n_agents=N_AGENTS)
    buffer.clear()

    log         = []
    best_reward = -np.inf

    print(f"\nTraining for {N_EPISODES} episodes...")
    print(f"{'Episode':>8} {'Reward':>10} {'Steps':>7} "
          f"{'Actor L':>9} {'Critic L':>9} {'Entropy':>9} {'Time':>7}")
    print("-" * 65)

    for episode in range(1, N_EPISODES + 1):
        t0 = time.time()
        local_states, joint_obs = env.reset()
        ep_reward = ep_steps = 0
        done = False

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
            mappo.save(_S35_DIR)
            with open(_S35_DIR / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)

        if ep_reward > best_reward:
            best_reward = ep_reward
            mappo.save(_S35_DIR / "best")

    mappo.save(_S35_DIR)
    with open(_S35_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone. Best reward: {best_reward:.4f}")
    print(f"Saved to: {_S35_DIR}")


if __name__ == "__main__":
    train()
