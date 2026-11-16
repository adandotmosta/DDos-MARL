"""
eval_s41.py
Evaluate S41 (raw top-6 features state) on Monday and Tuesday.

Outputs JSON results matching the format of S27/S29/S30 for direct comparison.

Usage:
    python scripts/eval_s41.py
    python scripts/eval_s41.py --checkpoint best   # use best checkpoint
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CSV_MONDAY, CSV_TUESDAY, CSV_THURSDAY, CSV_SATURDAY, LGBM_BUNDLE,
    MARL_DIR, N_AGENTS, WINDOW_SIZE, EVAL_DIR,
)
from src.environment_tiered_v5 import (
    MultiAgentEnv, STATE_DIM, JOINT_DIM, ACTION_DIM,
    ACTION_LABELS, TOP_FEATURES,
)
from src.mappo import MAPPO

_S41_DIR  = MARL_DIR / "ablation" / "S41"
_EVAL_DIR = EVAL_DIR / "s41"


def load_bundle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_csv(csv_path: Path, features: list) -> pd.DataFrame:
    needed = list(dict.fromkeys(
        features + TOP_FEATURES + ["label", "src_ip", "dst_ip", "timestamp"]
    ))
    all_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    needed = [c for c in needed if c in all_cols]
    df = pd.read_csv(csv_path, usecols=needed, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def run_episode(env: MultiAgentEnv, mappo: MAPPO) -> dict:
    import torch
    local_states, joint_obs = env.reset()
    total_reward  = 0.0
    total_steps   = 0
    benign_in     = attack_in     = 0
    benign_out    = attack_blocked = 0
    action_counts = [defaultdict(int) for _ in range(N_AGENTS)]

    done = False
    while not done:
        with torch.no_grad():
            actions, _, _ = mappo.act(local_states, joint_obs)

        next_local, next_joint, reward, done = env.step(actions)

        info = env.last_step_info
        benign_in     += info.get("benign_total",    0)
        attack_in     += info.get("attack_total",    0)
        benign_out    += info.get("benign_forwarded", 0)
        attack_blocked += info.get("attack_blocked",  0)

        r = float(np.mean(reward)) if isinstance(reward, list) else float(reward)
        total_reward += r
        total_steps  += 1

        for i, a in enumerate(actions):
            action_counts[i][ACTION_LABELS.get(a, str(a))] += 1

        local_states = next_local
        joint_obs    = next_joint

    tpr = attack_blocked / max(attack_in, 1)
    fpr = (benign_in - benign_out) / max(benign_in, 1)
    legit_ratio = benign_out / max(benign_in, 1)

    return {
        "total_reward":       round(total_reward, 4),
        "total_steps":        total_steps,
        "total_benign_in":    benign_in,
        "total_attack_in":    attack_in,
        "total_benign_out":   benign_out,
        "total_attack_blocked": attack_blocked,
        "tpr":                round(tpr, 4),
        "fpr":                round(fpr, 4),
        "legit_ratio":        round(legit_ratio, 4),
        "action_distribution": [dict(ac) for ac in action_counts],
    }


def evaluate(checkpoint: str = "final") -> None:
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_dir = _S41_DIR / "best" if checkpoint == "best" else _S41_DIR
    if not (ckpt_dir / "actor_r1.pt").exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_dir}. Run src/train_s41.py first."
        )

    bundle   = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]

    mappo = MAPPO(state_dim=STATE_DIM, joint_dim=JOINT_DIM,
                  action_dim=ACTION_DIM, n_agents=N_AGENTS)
    mappo.load(ckpt_dir)
    for actor in mappo.actors:
        actor.eval()
    mappo.critic.eval()
    print(f"Loaded checkpoint from: {ckpt_dir}")
    print(f"State: {STATE_DIM}-dim raw features  |  "
          f"Joint: {JOINT_DIM}-dim  |  Actions: {ACTION_DIM}\n")

    for day, csv_path in [("monday", CSV_MONDAY), ("tuesday", CSV_TUESDAY),
                          ("thursday", CSV_THURSDAY), ("saturday", CSV_SATURDAY)]:
        print(f"Evaluating on {day.capitalize()} …")
        df  = load_csv(csv_path, features)
        env = MultiAgentEnv(df=df, bundle=bundle, n_agents=N_AGENTS,
                            window_size=WINDOW_SIZE)
        res = run_episode(env, mappo)

        out = _EVAL_DIR / f"s41_{day}_results.json"

        with open(out, "w") as f:
            json.dump(res, f, indent=2)

        print(f"  TPR={res['tpr']:.4f}  FPR={res['fpr']:.4f}  "
              f"Legit={res['legit_ratio']:.4f}  Reward={res['total_reward']:.1f}")
        print(f"  Saved → {out}\n")

    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", choices=["final", "best"], default="final")
    args = p.parse_args()
    evaluate(args.checkpoint)
