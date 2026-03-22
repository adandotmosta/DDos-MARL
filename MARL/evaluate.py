"""
evaluate.py
Evaluation for CoopIDS.

Fixes applied:
  - Issue 1: metrics now tracked from all_forwarded returned by env.step()
  - Issue 2: Monday held-out = last 20% by timestamp (not full Monday)
  - Issue 3: ablation stubs clearly marked as not-yet-trained

Usage:
    python src/evaluate.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

try:
    from config import (
        CSV_MONDAY, CSV_TUESDAY, LGBM_BUNDLE,
        MARL_DIR, EVAL_DIR, N_AGENTS, WINDOW_SIZE,
        ABLATION_STAGES,
    )
    from src.environment import MultiAgentEnv
    from src.mappo import MAPPO
except ImportError:
    from coopids.config import (
        CSV_MONDAY, CSV_TUESDAY, LGBM_BUNDLE,
        MARL_DIR, EVAL_DIR, N_AGENTS, WINDOW_SIZE,
        ABLATION_STAGES,
    )
    from coopids.src.environment import MultiAgentEnv
    from coopids.src.mappo import MAPPO


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_bundle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_day(csv_path: Path, features: list) -> pd.DataFrame:
    needed = list(dict.fromkeys(
        features + ["label", "src_ip", "dst_ip",
                    "timestamp", "bytes_rate", "activity"]
    ))
    df = pd.read_csv(csv_path, usecols=needed, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def monday_heldout(df: pd.DataFrame, heldout_ratio: float = 0.2) -> pd.DataFrame:
    """
    Fix Issue 2: return last heldout_ratio fraction of Monday by timestamp.
    Time-ordered — no shuffling.
    """
    n     = len(df)
    split = int(n * (1.0 - heldout_ratio))
    return df.iloc[split:].reset_index(drop=True)


def greedy_actions(mappo: MAPPO, local_states: List[np.ndarray]) -> List[int]:
    """Greedy (argmax) actions — no exploration during evaluation."""
    actions = []
    for i, actor in enumerate(mappo.actors):
        obs_t  = torch.FloatTensor(local_states[i]).unsqueeze(0)
        with torch.no_grad():
            dist   = actor.get_dist(obs_t)
            action = int(dist.probs.argmax().item())
        actions.append(action)
    return actions


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(
    env:   MultiAgentEnv,
    mappo: MAPPO,
) -> Dict:
    """
    Run one greedy evaluation episode.

    Fix Issue 1: tracks benign/attack counts from all_forwarded
    returned by env.step() — no longer zero by design.
    """
    local_states, joint_obs = env.reset()

    total_reward         = 0.0
    total_steps          = 0

    # Counters — updated from all_forwarded each step
    total_benign_in      = 0   # total benign flows seen (before action)
    total_benign_out     = 0   # benign flows that passed through
    total_attack_in      = 0   # total attack flows seen
    total_attack_blocked = 0   # attack flows that were blocked

    # Per-action counter for analysis
    action_counts = {i: 0 for i in range(6)}

    done = False

    while not done:
        actions = greedy_actions(mappo, local_states)

        # env.step now returns all_forwarded (Fix Issue 1)
        next_local, next_joint, reward, done, all_forwarded = env.step(actions)

        # ── Count from raw windows (before action) and forwarded (after)
        for router, forwarded, action in zip(env.routers, all_forwarded, actions):
            window = router.get_window(env.t_current - env.window_size)

            if "label" in window.columns:
                total_benign_in  += int((window["label"] == "Benign").sum())
                total_attack_in  += int((window["label"] == "Attack").sum())

            if "label" in forwarded.columns:
                total_benign_out += int((forwarded["label"] == "Benign").sum())
                # Blocked attacks = attacks in window - attacks in forwarded
                attack_forwarded  = int((forwarded["label"] == "Attack").sum())
                attack_in_window  = int((window["label"] == "Attack").sum()) \
                                    if "label" in window.columns else 0
                total_attack_blocked += max(0, attack_in_window - attack_forwarded)

            action_counts[action] += 1

        total_reward += reward
        total_steps  += 1
        local_states  = next_local
        joint_obs     = next_joint

    # ── Compute final metrics
    tpr         = total_attack_blocked / total_attack_in \
                  if total_attack_in  > 0 else 0.0
    fpr         = (total_benign_in - total_benign_out) / total_benign_in \
                  if total_benign_in > 0 else 0.0
    legit_ratio = total_benign_out / total_benign_in \
                  if total_benign_in > 0 else 0.0

    # F1 (treating attack detection as positive class)
    precision = tpr / (tpr + fpr + 1e-8)
    f1        = 2 * precision * tpr / (precision + tpr + 1e-8)

    return {
        "total_reward":          round(total_reward,         4),
        "total_steps":           total_steps,
        "total_benign_in":       total_benign_in,
        "total_benign_out":      total_benign_out,
        "total_attack_in":       total_attack_in,
        "total_attack_blocked":  total_attack_blocked,
        "tpr":                   round(tpr,         4),
        "fpr":                   round(fpr,         4),
        "legit_ratio":           round(legit_ratio, 4),
        "f1":                    round(f1,          4),
        "action_distribution":   action_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation wrappers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_day(
    df:     pd.DataFrame,
    bundle: dict,
    mappo:  MAPPO,
    label:  str,
) -> Dict:
    """Evaluate trained MAPPO on one day's DataFrame."""
    print(f"\nEvaluating on {label}...")
    print(f"  Flows: {len(df)}")
    if "label" in df.columns:
        print(df["label"].value_counts().to_string())

    env     = MultiAgentEnv(df=df, bundle=bundle, n_agents=N_AGENTS)
    metrics = run_episode(env, mappo)

    print(f"  Reward:      {metrics['total_reward']}")
    print(f"  TPR:         {metrics['tpr']}")
    print(f"  FPR:         {metrics['fpr']}")
    print(f"  Legit ratio: {metrics['legit_ratio']}")
    print(f"  F1:          {metrics['f1']}")
    print(f"  Actions:     {metrics['action_distribution']}")
    return metrics


def ablation_study(
    df_monday: pd.DataFrame,
    bundle:    dict,
) -> Dict:
    """
    Run ablation stages S1-S8.
    Stages without saved models are skipped with a clear note.
    """
    results = {}

    print("\n" + "=" * 55)
    print("ABLATION STUDY")
    print("=" * 55)

    for stage, description in ABLATION_STAGES.items():
        print(f"\n{stage}: {description}")

        stage_dir = MARL_DIR / "ablation" / stage
        if not stage_dir.exists():
            print(f"  Skipped — no saved model at {stage_dir}")
            results[stage] = {
                "description": description,
                "metrics":     {"note": "not trained — run ablation training first"},
            }
            continue

        env   = MultiAgentEnv(df=df_monday, bundle=bundle, n_agents=N_AGENTS)
        mappo = MAPPO()
        mappo.load(stage_dir)
        metrics = run_episode(env, mappo)

        results[stage] = {"description": description, "metrics": metrics}
        print(f"  TPR={metrics['tpr']}  FPR={metrics['fpr']}  "
              f"F1={metrics['f1']}  Legit={metrics['legit_ratio']}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load bundle and MAPPO
    print(f"Loading LGBM bundle from {LGBM_BUNDLE}...")
    bundle   = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]

    print(f"Loading trained MAPPO from {MARL_DIR}...")
    mappo = MAPPO()
    mappo.load(MARL_DIR)

    # ── Fix Issue 2: Monday held-out = last 20% by timestamp
    print(f"\nLoading Monday from {CSV_MONDAY}...")
    df_monday_full   = load_day(CSV_MONDAY, features)
    df_monday_heldout = monday_heldout(df_monday_full, heldout_ratio=0.2)
    print(f"Full Monday: {len(df_monday_full)} flows")
    print(f"Held-out 20%: {len(df_monday_heldout)} flows "
          f"(from {df_monday_heldout['timestamp'].min()} "
          f"to {df_monday_heldout['timestamp'].max()})")

    monday_metrics = evaluate_day(
        df     = df_monday_heldout,
        bundle = bundle,
        mappo  = mappo,
        label  = "Monday held-out 20% (seen attack types)",
    )
    with open(EVAL_DIR / "monday_results.json", "w") as f:
        json.dump(monday_metrics, f, indent=2)

    # ── Tuesday: full day (unseen attacks)
    print(f"\nLoading Tuesday from {CSV_TUESDAY}...")
    df_tuesday = load_day(CSV_TUESDAY, features)

    tuesday_metrics = evaluate_day(
        df     = df_tuesday,
        bundle = bundle,
        mappo  = mappo,
        label  = "Tuesday (unseen attack types)",
    )
    with open(EVAL_DIR / "tuesday_results.json", "w") as f:
        json.dump(tuesday_metrics, f, indent=2)

    # ── Comparison table
    print("\n" + "=" * 60)
    print("MONDAY vs TUESDAY COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<22} {'Monday (seen)':>16} {'Tuesday (unseen)':>17} {'Delta':>8}")
    print("-" * 60)
    for key in ["total_reward", "tpr", "fpr", "legit_ratio", "f1"]:
        m     = monday_metrics.get(key,  0.0)
        t     = tuesday_metrics.get(key, 0.0)
        delta = t - m if isinstance(m, (int, float)) else 0.0
        print(f"{key:<22} {m:>16.4f} {t:>17.4f} {delta:>8.4f}")

    # ── Ablation
    ablation = ablation_study(df_monday_full, bundle)
    with open(EVAL_DIR / "ablation_results.json", "w") as f:
        json.dump(ablation, f, indent=2)

    print(f"\nAll results saved to: {EVAL_DIR}")


if __name__ == "__main__":
    main()