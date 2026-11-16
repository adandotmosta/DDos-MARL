"""
eval_static_baseline.py
Static baseline — fixed threshold policy, NO training, NO learning.

Each run: all 3 routers pick the SAME fixed action for EVERY window.
6 actions × 4 days = 24 runs total.

Compare the best static row against S27/S29 MARL results.
If best static ≈ MARL → RL learned nothing a fixed rule couldn't do.

Usage:
    python eval_static_baseline.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CSV_MONDAY, CSV_TUESDAY, CSV_THURSDAY, CSV_SATURDAY,
    LGBM_BUNDLE, N_AGENTS, WINDOW_SIZE,
)
from src.environment_tiered_v2 import (
    MultiAgentEnv, ACTION_LABELS, ACTION_DIM, TIERED_PROFILES_V2,
)


def load_day(csv_path: Path, features: list) -> pd.DataFrame:
    needed = list(dict.fromkeys(
        features + ["label", "src_ip", "dst_ip", "timestamp", "bytes_rate"]
    ))
    all_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    needed = [c for c in needed if c in all_cols]
    df = pd.read_csv(csv_path, usecols=needed, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def run_fixed(df, bundle, action: int) -> dict:
    """Fresh env per run so CAPTCHA RNG is identical across actions."""
    env = MultiAgentEnv(df=df, bundle=bundle, n_agents=N_AGENTS,
                        window_size=WINDOW_SIZE, seed=42)
    env.reset()
    benign_in = attack_in = benign_out = attack_blocked = 0
    done = False
    while not done:
        _, _, _, done = env.step([action] * N_AGENTS)
        info = env.last_step_info
        benign_in      += info["benign_total"]
        attack_in      += info["attack_total"]
        benign_out     += info["benign_forwarded"]
        attack_blocked += info["attack_blocked"]
    tpr = attack_blocked / attack_in if attack_in > 0 else 0.0
    fpr = 1.0 - (benign_out / benign_in) if benign_in > 0 else 0.0
    return {"tpr": round(tpr, 4), "fpr": round(fpr, 4)}


def main():
    bundle = pickle.load(open(LGBM_BUNDLE, "rb"))
    features = bundle["features"]

    days = {}
    for name, path in [("Monday", CSV_MONDAY), ("Tuesday", CSV_TUESDAY)]:
        if path.exists():
            days[name] = load_day(path, features)
            print(f"Loaded {name}: {len(days[name]):,} flows")

    # Header
    print(f"\n{'Action':<12} {'T_blk':>6} {'T_pzl':>6}", end="")
    for d in days:
        print(f"  {d:>12}", end="")
    print()
    print(f"{'':12} {'':6} {'':6}", end="")
    for _ in days:
        print(f"  {'TPR':>5} {'FPR':>5}", end="")
    print()
    print("-" * (24 + 14 * len(days)))

    # Static runs
    results = {}
    for aid in range(ACTION_DIM):
        tb, tp = TIERED_PROFILES_V2[aid]
        print(f"{ACTION_LABELS[aid]:<12} {tb:>6.2f} {tp:>6.2f}", end="")
        results[aid] = {}
        for name, df in days.items():
            r = run_fixed(df, bundle, aid)
            results[aid][name] = r
            print(f"  {r['tpr']:>5.1%} {r['fpr']:>5.1%}", end="")
        print()

    # MARL reference rows
    print("-" * (24 + 14 * len(days)))
    for label, ref in [("S27 MAPPO", {"Monday": (0.9956, 0.0276), "Tuesday": (0.8716, 0.0668),
                                       "Thursday": (0.0, 0.5129), "Saturday": (0.0, 0.4881)}),
                        ("S29 IPPO",  {"Monday": (0.9903, 0.0235), "Tuesday": (0.8938, 0.0715)}),
                        ("S28 mixed", {"Monday": (0.2412, 0.0067), "Tuesday": (0.6362, 0.0277),
                                       "Saturday": (0.0, 0.3051)})]:
        print(f"{label:<12} {'adapt':>6} {'adapt':>6}", end="")
        for name in days:
            if name in ref:
                t, f = ref[name]
                print(f"  {t:>5.1%} {f:>5.1%}", end="")
            else:
                print(f"  {'—':>5} {'—':>5}", end="")
        print()


if __name__ == "__main__":
    main()
