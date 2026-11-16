"""
eval_s41_all_days.py
Evaluate S41 (7-dim raw-feature state) on all four days including benign-only Thursday/Saturday.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CSV_MONDAY, CSV_TUESDAY, CSV_THURSDAY, CSV_SATURDAY, LGBM_BUNDLE,
    MARL_DIR, N_AGENTS, WINDOW_SIZE, EVAL_DIR,
)

# S41 was trained with the original top-6 features → STATE_DIM = 7
TOP_6 = [
    "handshake_duration",
    "packet_IAT_min",
    "packet_IAT_max",
    "packets_IAT_median",
    "median_bwd_packets_delta_time",
    "fwd_init_win_bytes",
]
STATE_DIM = len(TOP_6) + 1   # 7
JOINT_DIM = STATE_DIM * N_AGENTS  # 21
ACTION_DIM = 6

ACTION_LABELS = {0: "passive", 1: "light", 2: "moderate",
                 3: "firm",    4: "aggressive", 5: "lockdown"}

_S41_DIR  = MARL_DIR / "ablation" / "S41"
_EVAL_DIR = EVAL_DIR / "s41"


def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_csv(csv_path, lgbm_features):
    needed = list(dict.fromkeys(
        lgbm_features + TOP_6 + ["label", "src_ip", "dst_ip", "timestamp"]
    ))
    all_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    needed = [c for c in needed if c in all_cols]
    df = pd.read_csv(csv_path, usecols=needed, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def run_episode(env, mappo):
    local_states, joint_obs = env.reset()
    total_reward = 0.0
    total_steps  = 0
    benign_in = attack_in = benign_out = attack_blocked = 0
    action_counts = [defaultdict(int) for _ in range(N_AGENTS)]

    done = False
    while not done:
        with torch.no_grad():
            actions, _, _ = mappo.act(local_states, joint_obs)
        next_local, next_joint, reward, done = env.step(actions)

        info = env.last_step_info
        benign_in      += info.get("benign_total",     0)
        attack_in      += info.get("attack_total",     0)
        benign_out     += info.get("benign_forwarded", 0)
        attack_blocked += info.get("attack_blocked",   0)

        r = float(np.mean(reward)) if isinstance(reward, list) else float(reward)
        total_reward += r
        total_steps  += 1

        for i, a in enumerate(actions):
            action_counts[i][ACTION_LABELS.get(a, str(a))] += 1

        local_states = next_local
        joint_obs    = next_joint

    tpr = attack_blocked / max(attack_in, 1)
    fpr = (benign_in - benign_out) / max(benign_in, 1)
    return {
        "total_reward":         round(total_reward, 4),
        "total_steps":          total_steps,
        "total_benign_in":      benign_in,
        "total_attack_in":      attack_in,
        "total_benign_out":     benign_out,
        "total_attack_blocked": attack_blocked,
        "tpr":                  round(tpr, 4),
        "fpr":                  round(fpr, 4),
        "legit_ratio":          round(benign_out / max(benign_in, 1), 4),
        "action_distribution":  [dict(ac) for ac in action_counts],
    }


def main():
    # Import the v5 environment but patch TOP_FEATURES before use
    import src.environment_tiered_v5 as env_mod
    env_mod.TOP_FEATURES = TOP_6  # patch to match checkpoint

    from src.environment_tiered_v5 import MultiAgentEnv
    from src.mappo import MAPPO

    _EVAL_DIR.mkdir(parents=True, exist_ok=True)

    bundle   = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]

    mappo = MAPPO(state_dim=STATE_DIM, joint_dim=JOINT_DIM,
                  action_dim=ACTION_DIM, n_agents=N_AGENTS)
    mappo.load(_S41_DIR)
    for actor in mappo.actors:
        actor.eval()
    mappo.critic.eval()
    print(f"S41 checkpoint loaded from {_S41_DIR}")
    print(f"STATE_DIM={STATE_DIM}  JOINT_DIM={JOINT_DIM}  ACTION_DIM={ACTION_DIM}\n")

    days = [
        ("monday",   CSV_MONDAY),
        ("tuesday",  CSV_TUESDAY),
        ("thursday", CSV_THURSDAY),
        ("saturday", CSV_SATURDAY),
    ]

    print(f"{'Day':<12} {'TPR':>6} {'FPR':>6} {'Legit':>7} {'Reward':>10}")
    print("-" * 48)
    for day, csv_path in days:
        df  = load_csv(csv_path, features)
        env = MultiAgentEnv(df=df, bundle=bundle, n_agents=N_AGENTS,
                            window_size=WINDOW_SIZE)
        res = run_episode(env, mappo)

        out = _EVAL_DIR / f"s41_{day}_results.json"
        with open(out, "w") as f:
            json.dump(res, f, indent=2)

        print(f"{day:<12} {res['tpr']:>6.4f} {res['fpr']:>6.4f} "
              f"{res['legit_ratio']:>7.4f} {res['total_reward']:>10.1f}")
        for i, ac in enumerate(res["action_distribution"]):
            dominant = max(ac, key=ac.get) if ac else "—"
            print(f"  Agent {i+1}: dominant={dominant}  dist={ac}")

    print("\nDone. Results saved to", _EVAL_DIR)


if __name__ == "__main__":
    main()
