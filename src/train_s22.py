"""
train_s22.py
MAPPO + counterfactual reward (S22, COMA-style credit assignment).

Instead of shared global reward, each agent receives a counterfactual-based
reward that measures its MARGINAL CONTRIBUTION to the team:

    cf_reward_i = global_reward(all_agents) - global_reward(agent_i forwards everything)

If agent i puzzles/blocks effectively:
    cf_reward_i > 0  (team did better WITH agent i than without)
If agent i's action made no difference (e.g., all 3 agents already covered it):
    cf_reward_i ≈ 0  (marginal contribution is near zero)

This breaks the credit assignment problem: puzzle-happy agents that contribute
nothing marginal (because teammates cover the same flows) get near-zero reward,
forcing them to find actions with positive marginal impact.

State:     5-dim baseline (same as S9)
Actions:   6 (same as S9)
Reward:    counterfactual marginal contribution per agent
Saves to:  artifacts/marl/v1/ablation/S22/

Usage:
    python main.py --stage s22
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
        REWARD_LEGIT_W, REWARD_ATTACK_W,
    )
    import src.environment as _env_module
    from src.environment import MultiAgentEnv
    from src.mappo import MAPPO, RolloutBuffer
except ImportError:
    from coopids.config import (
        CSV_MONDAY, LGBM_BUNDLE,
        MARL_DIR, N_EPISODES, UPDATE_EVERY,
        CHECKPOINT_EVERY, N_AGENTS, WINDOW_SIZE,
        REWARD_LEGIT_W, REWARD_ATTACK_W,
    )
    import coopids.src.environment as _env_module
    from coopids.src.environment import MultiAgentEnv
    from coopids.src.mappo import MAPPO, RolloutBuffer

_S22_DIR = MARL_DIR / "ablation" / "S22"


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


def _compute_cf_rewards(info: dict, n_agents: int) -> list:
    """
    Compute counterfactual marginal reward for each agent.

    For agent i: CF = global_reward - reward_if_agent_i_forwarded_everything
    Positive CF means agent i's action improved global outcome.

    Uses per_agent breakdowns stored in info["per_agent"] by environment.
    Falls back to mean local reward if per_agent not available.
    """
    per_agent = info.get("per_agent")
    if not per_agent or len(per_agent) != n_agents:
        return [0.0] * n_agents

    global_benign_total  = info.get("benign_total",  1)
    global_attack_total  = info.get("attack_total",  1)
    global_benign_passed = info.get("benign_forwarded", 0)
    global_attack_passed = info.get("attack_forwarded", 0)

    if global_benign_total == 0 and global_attack_total == 0:
        return [0.0] * n_agents

    # Actual global reward
    actual_legit_ratio  = global_benign_passed / max(global_benign_total,  1)
    actual_attack_ratio = global_attack_passed / max(global_attack_total, 1)
    actual_reward = float(REWARD_LEGIT_W * actual_legit_ratio
                          - REWARD_ATTACK_W * actual_attack_ratio)

    cf_rewards = []
    for i in range(n_agents):
        pa = per_agent[i]
        # Counterfactual: agent i forwards everything — add back what it blocked
        benign_it_blocked  = pa["benign_total"]  - pa["benign_passed"]
        attack_it_blocked  = pa["attack_total"]  - pa["attack_passed"]

        cf_benign_passed = global_benign_passed + benign_it_blocked
        cf_attack_passed = global_attack_passed + attack_it_blocked

        cf_legit_ratio  = cf_benign_passed / max(global_benign_total,  1)
        cf_attack_ratio = cf_attack_passed / max(global_attack_total, 1)
        cf_reward = float(REWARD_LEGIT_W * cf_legit_ratio
                          - REWARD_ATTACK_W * cf_attack_ratio)

        # Marginal contribution: how much better did we do WITH agent i?
        cf_rewards.append(actual_reward - cf_reward)

    return cf_rewards


def train() -> None:
    _S22_DIR.mkdir(parents=True, exist_ok=True)

    # Patch environment: local reward (to trigger per_agent tracking)
    _env_module.LOCAL_REWARD    = True
    _env_module.EXTENDED_STATE  = False
    _env_module.PUZZLE_ENABLED  = True
    _env_module.PUZZLE_COST     = 0.0

    print("S22 config: COUNTERFACTUAL reward (COMA-style marginal contribution)")

    print(f"Loading LGBM bundle from {LGBM_BUNDLE}...")
    bundle = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]
    print(f"Features: {len(features)}")

    print(f"\nLoading Monday data from {CSV_MONDAY}...")
    df_monday = load_monday(CSV_MONDAY, features)

    print("\nBuilding MultiAgentEnv (counterfactual reward)...")
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
    print(f"{'Episode':>8} {'CF Reward':>10} {'Steps':>7} "
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
            next_local, next_joint, _, done = env.step(actions)

            # Replace local rewards with counterfactual marginal rewards
            info = env.last_step_info
            cf_rewards = _compute_cf_rewards(info, N_AGENTS)
            mean_cf = float(np.mean(cf_rewards))

            buffer.store(
                local_obs=local_states, joint_obs=joint_obs,
                actions=actions, log_probs=log_probs,
                reward=mean_cf, done=done, value=value,
            )
            local_states = next_local
            joint_obs    = next_joint
            ep_reward   += mean_cf
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
            mappo.save(_S22_DIR)
            with open(_S22_DIR / "training_log.json", "w") as f:
                json.dump(log, f, indent=2)
            print(f"  Checkpoint saved at episode {episode}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            mappo.save(_S22_DIR / "best")

    mappo.save(_S22_DIR)
    with open(_S22_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining complete. Best CF reward: {best_reward:.4f}")
    print(f"Models saved to: {_S22_DIR}")


if __name__ == "__main__":
    train()
