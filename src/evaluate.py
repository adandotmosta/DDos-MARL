"""
evaluate.py
Evaluation for CoopIDS.

Runs:
  1. Monday held-out evaluation
  2. Tuesday generalization evaluation (unseen attacks)
  3. Ablation study S1-S8

Usage:
    python src/evaluate.py
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch

try:
    from config import (
        CSV_MONDAY, CSV_TUESDAY, LGBM_BUNDLE,
        MARL_DIR, EVAL_DIR, N_AGENTS, WINDOW_SIZE,
        ABLATION_STAGES, ACTIONS, S11_DIR, S12_DIR,
    )
    from src.environment import MultiAgentEnv
    import src.environment as _env_module
    from src.mappo import MAPPO
    from src.iql import IQL
    from src.ippo import IPPO
except ImportError:
    from coopids.config import (
        CSV_MONDAY, CSV_TUESDAY, LGBM_BUNDLE,
        MARL_DIR, EVAL_DIR, N_AGENTS, WINDOW_SIZE,
        ABLATION_STAGES, ACTIONS, S11_DIR, S12_DIR,
    )
    from coopids.src.environment import MultiAgentEnv
    import coopids.src.environment as _env_module
    from coopids.src.mappo import MAPPO
    from coopids.src.iql import IQL
    from coopids.src.ippo import IPPO

# Evaluation always uses global scalar reward regardless of LOCAL_REWARD setting
_env_module.LOCAL_REWARD = False
# Baseline models (S6–S10) were trained with 5-dim state and 6-action space.
# Force 5-dim + original DROP_PROBS for baseline eval.
# S11 block will temporarily override to 10-action space.
_env_module.EXTENDED_STATE = False

# Original 6-action DROP_PROBS (used by S6–S10 models)
_DROP_PROBS_6 = {2: 0.3, 3: 0.6, 4: 0.9, 5: 1.0}
# New 10-action DROP_PROBS (used by S11 model)
_DROP_PROBS_10 = {2: 0.10, 3: 0.20, 4: 0.30, 5: 0.50,
                  6: 0.70, 7: 0.90, 8: 0.95, 9: 1.00}
# S12 9-action DROP_PROBS (no puzzle; IDs 1–8 = throttle_10…block)
_DROP_PROBS_s12 = {1: 0.10, 2: 0.20, 3: 0.30, 4: 0.50,
                   5: 0.70, 6: 0.90, 7: 0.95, 8: 1.00}

# Default: use 6-action space for all baseline models
_env_module.DROP_PROBS = _DROP_PROBS_6


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_bundle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_day(csv_path: Path, features: list) -> pd.DataFrame:
    needed = list(dict.fromkeys(
        features + ["label", "src_ip", "dst_ip", "timestamp", "bytes_rate"]
    ))
    # Only load columns that actually exist in this CSV
    all_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    needed   = [c for c in needed if c in all_cols]
    df = pd.read_csv(csv_path, usecols=needed, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def run_episode(
    env: MultiAgentEnv,
    mappo: MAPPO,
) -> Dict:
    """
    Run one greedy evaluation episode.
    Returns full metrics dict including TPR, FPR, and per-agent action distribution.
    """
    local_states, joint_obs = env.reset()

    total_reward         = 0.0
    total_steps          = 0
    total_benign_in      = 0
    total_benign_out     = 0
    total_attack_in      = 0
    total_attack_blocked = 0
    # Per-agent action counts: action_id → count
    action_counts = [defaultdict(int) for _ in range(mappo.n_agents)]
    done = False

    while not done:
        # Greedy action — argmax of policy probabilities (no exploration)
        actions = []
        for i, actor in enumerate(mappo.actors):
            obs_t  = torch.FloatTensor(local_states[i]).unsqueeze(0)
            dist   = actor.get_dist(obs_t)
            action = int(dist.probs.argmax().item())
            actions.append(action)
            action_counts[i][action] += 1

        next_local, next_joint, reward, done = env.step(actions)

        # Read exact forwarded/blocked counts from environment
        info = env.last_step_info
        total_benign_in      += info["benign_total"]
        total_attack_in      += info["attack_total"]
        total_benign_out     += info["benign_forwarded"]
        total_attack_blocked += info["attack_blocked"]

        total_reward += reward
        total_steps  += 1
        local_states  = next_local
        joint_obs     = next_joint

    # ── Metrics
    tpr         = total_attack_blocked / total_attack_in if total_attack_in > 0 else 0.0
    legit_ratio = total_benign_out     / total_benign_in if total_benign_in > 0 else 0.0
    fpr         = 1.0 - legit_ratio   # fraction of legit traffic incorrectly blocked

    # Convert action counts to labelled dicts
    action_dist = [
        {ACTIONS.get(k, str(k)): v for k, v in sorted(ac.items())}
        for ac in action_counts
    ]

    return {
        "total_reward":         round(total_reward, 4),
        "total_steps":          total_steps,
        "total_benign_in":      total_benign_in,
        "total_attack_in":      total_attack_in,
        "total_benign_out":     total_benign_out,
        "total_attack_blocked": total_attack_blocked,
        "tpr":                  round(tpr, 4),
        "fpr":                  round(fpr, 4),
        "legit_ratio":          round(legit_ratio, 4),
        "action_distribution":  action_dist,
    }


def run_episode_iql(
    env: MultiAgentEnv,
    iql: IQL,
) -> Dict:
    """
    Run one greedy evaluation episode using IQL agents.
    Identical metrics to run_episode() for fair comparison.
    """
    local_states, _ = env.reset()

    total_reward         = 0.0
    total_steps          = 0
    total_benign_in      = 0
    total_benign_out     = 0
    total_attack_in      = 0
    total_attack_blocked = 0
    action_counts = [defaultdict(int) for _ in range(iql.n_agents)]
    done = False

    while not done:
        # Greedy: argmax Q-value, no exploration
        actions = iql.act(local_states, greedy=True)
        for i, a in enumerate(actions):
            action_counts[i][a] += 1

        next_local, _, reward, done = env.step(actions)

        info = env.last_step_info
        total_benign_in      += info["benign_total"]
        total_attack_in      += info["attack_total"]
        total_benign_out     += info["benign_forwarded"]
        total_attack_blocked += info["attack_blocked"]

        total_reward += reward
        total_steps  += 1
        local_states  = next_local

    tpr         = total_attack_blocked / total_attack_in if total_attack_in > 0 else 0.0
    legit_ratio = total_benign_out     / total_benign_in if total_benign_in > 0 else 0.0
    fpr         = 1.0 - legit_ratio

    action_dist = [
        {ACTIONS.get(k, str(k)): v for k, v in sorted(ac.items())}
        for ac in action_counts
    ]

    return {
        "total_reward":         round(total_reward, 4),
        "total_steps":          total_steps,
        "total_benign_in":      total_benign_in,
        "total_attack_in":      total_attack_in,
        "total_benign_out":     total_benign_out,
        "total_attack_blocked": total_attack_blocked,
        "tpr":                  round(tpr, 4),
        "fpr":                  round(fpr, 4),
        "legit_ratio":          round(legit_ratio, 4),
        "action_distribution":  action_dist,
    }


def run_episode_ippo(
    env:  MultiAgentEnv,
    ippo: IPPO,
) -> Dict:
    """
    Run one greedy evaluation episode using IPPO agents.
    Greedy = argmax of actor's softmax probabilities (no sampling).
    Identical metrics to run_episode() for fair comparison.
    """
    local_states, _ = env.reset()

    total_reward         = 0.0
    total_steps          = 0
    total_benign_in      = 0
    total_benign_out     = 0
    total_attack_in      = 0
    total_attack_blocked = 0
    action_counts = [defaultdict(int) for _ in range(ippo.n_agents)]
    done = False

    while not done:
        # Greedy: argmax of actor logits, no sampling
        actions = []
        for i, actor in enumerate(ippo.actors):
            obs_t  = torch.FloatTensor(local_states[i]).unsqueeze(0)
            dist   = actor.get_dist(obs_t)
            action = int(dist.probs.argmax().item())
            actions.append(action)
            action_counts[i][action] += 1

        next_local, _, reward, done = env.step(actions)

        info = env.last_step_info
        total_benign_in      += info["benign_total"]
        total_attack_in      += info["attack_total"]
        total_benign_out     += info["benign_forwarded"]
        total_attack_blocked += info["attack_blocked"]

        total_reward += reward
        total_steps  += 1
        local_states  = next_local

    tpr         = total_attack_blocked / total_attack_in if total_attack_in > 0 else 0.0
    legit_ratio = total_benign_out     / total_benign_in if total_benign_in > 0 else 0.0
    fpr         = 1.0 - legit_ratio

    action_dist = [
        {ACTIONS.get(k, str(k)): v for k, v in sorted(ac.items())}
        for ac in action_counts
    ]

    return {
        "total_reward":         round(total_reward, 4),
        "total_steps":          total_steps,
        "total_benign_in":      total_benign_in,
        "total_attack_in":      total_attack_in,
        "total_benign_out":     total_benign_out,
        "total_attack_blocked": total_attack_blocked,
        "tpr":                  round(tpr, 4),
        "fpr":                  round(fpr, 4),
        "legit_ratio":          round(legit_ratio, 4),
        "action_distribution":  action_dist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation functions
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_day(
    csv_path: Path,
    bundle:   dict,
    mappo:    MAPPO,
    label:    str,
) -> Dict:
    """Evaluate trained MAPPO on one day."""
    features = bundle["features"]
    df       = load_day(csv_path, features)

    print(f"\nEvaluating on {label}...")
    print(df["label"].value_counts().to_string())

    env     = MultiAgentEnv(df=df, bundle=bundle, n_agents=N_AGENTS)
    metrics = run_episode(env, mappo)

    print(f"  Reward:        {metrics['total_reward']}")
    print(f"  TPR:           {metrics['tpr']:.4f}  "
          f"({metrics['total_attack_blocked']}/{metrics['total_attack_in']} attacks blocked)")
    print(f"  FPR:           {metrics['fpr']:.4f}  "
          f"({metrics['total_benign_in'] - metrics['total_benign_out']}/{metrics['total_benign_in']} legit blocked)")
    print(f"  Legit ratio:   {metrics['legit_ratio']:.4f}")
    print(f"  Action distribution per agent:")
    for i, dist in enumerate(metrics["action_distribution"]):
        print(f"    Router {i+1}: {dist}")

    return metrics


def evaluate_day_iql(
    csv_path: Path,
    bundle:   dict,
    iql:      IQL,
    label:    str,
) -> Dict:
    """Evaluate trained IQL on one day."""
    features = bundle["features"]
    df       = load_day(csv_path, features)

    print(f"\nEvaluating IQL on {label}...")
    print(df["label"].value_counts().to_string())

    env     = MultiAgentEnv(df=df, bundle=bundle, n_agents=N_AGENTS)
    metrics = run_episode_iql(env, iql)

    print(f"  Reward:        {metrics['total_reward']}")
    print(f"  TPR:           {metrics['tpr']:.4f}  "
          f"({metrics['total_attack_blocked']}/{metrics['total_attack_in']} attacks blocked)")
    print(f"  FPR:           {metrics['fpr']:.4f}  "
          f"({metrics['total_benign_in'] - metrics['total_benign_out']}/{metrics['total_benign_in']} legit blocked)")
    print(f"  Legit ratio:   {metrics['legit_ratio']:.4f}")
    print(f"  Action distribution per agent:")
    for i, dist in enumerate(metrics["action_distribution"]):
        print(f"    Router {i+1}: {dist}")

    return metrics


def evaluate_day_ippo(
    csv_path: Path,
    bundle:   dict,
    ippo:     IPPO,
    label:    str,
) -> Dict:
    """Evaluate trained IPPO on one day."""
    features = bundle["features"]
    df       = load_day(csv_path, features)

    print(f"\nEvaluating IPPO on {label}...")
    print(df["label"].value_counts().to_string())

    env     = MultiAgentEnv(df=df, bundle=bundle, n_agents=N_AGENTS)
    metrics = run_episode_ippo(env, ippo)

    print(f"  Reward:        {metrics['total_reward']}")
    print(f"  TPR:           {metrics['tpr']:.4f}  "
          f"({metrics['total_attack_blocked']}/{metrics['total_attack_in']} attacks blocked)")
    print(f"  FPR:           {metrics['fpr']:.4f}  "
          f"({metrics['total_benign_in'] - metrics['total_benign_out']}/{metrics['total_benign_in']} legit blocked)")
    print(f"  Legit ratio:   {metrics['legit_ratio']:.4f}")
    print(f"  Action distribution per agent:")
    for i, dist in enumerate(metrics["action_distribution"]):
        print(f"    Router {i+1}: {dist}")

    return metrics


def ablation_study(
    df_monday: pd.DataFrame,
    bundle:    dict,
) -> Dict:
    """
    Run ablation stages S1-S8.
    Each stage uses different config/policy.
    Returns dict of results per stage.
    """
    results = {}

    print("\n" + "="*55)
    print("ABLATION STUDY")
    print("="*55)

    for stage, description in ABLATION_STAGES.items():
        print(f"\n{stage}: {description}")

        stage_dir = MARL_DIR / "ablation" / stage
        env       = MultiAgentEnv(df=df_monday, bundle=bundle, n_agents=N_AGENTS)

        if stage == "S6":
            # S6 uses IQL (independent DQN agents, no centralized critic)
            agent = IQL(state_dim=5, action_dim=6)
            if stage_dir.exists():
                agent.load(stage_dir)
                metrics = run_episode_iql(env, agent)
            else:
                print(f"  No saved model for {stage} — skipping.")
                metrics = {"note": "not trained yet"}
        elif stage == "S7":
            # S7 uses IPPO (independent PPO, local critic, no centralized critic)
            agent = IPPO(state_dim=5, action_dim=6)
            if stage_dir.exists():
                agent.load(stage_dir)
                metrics = run_episode_ippo(env, agent)
            else:
                print(f"  No saved model for {stage} — skipping.")
                metrics = {"note": "not trained yet"}
        elif stage == "S10":
            # S10 uses MAPPO with 7-dim extended state, 6-action space
            if stage_dir.exists():
                _env_module.EXTENDED_STATE = True
                _env_module.STATE_DIM = 7
                env_e13 = MultiAgentEnv(df=df_monday, bundle=bundle, n_agents=N_AGENTS)
                agent = MAPPO(state_dim=7, joint_dim=21, action_dim=6)
                agent.load(stage_dir)
                metrics = run_episode(env_e13, agent)
                _env_module.EXTENDED_STATE = False
                _env_module.STATE_DIM = 5
            else:
                print(f"  No saved model for {stage} — skipping.")
                metrics = {"note": "not trained yet"}
        elif stage == "S11":
            # S11 was trained with 7-dim state + 10-action space
            if stage_dir.exists():
                _env_module.EXTENDED_STATE = True
                _env_module.STATE_DIM = 7
                _env_module.DROP_PROBS = _DROP_PROBS_10
                env_s11 = MultiAgentEnv(df=df_monday, bundle=bundle, n_agents=N_AGENTS)
                agent = MAPPO(state_dim=7, joint_dim=21, action_dim=10)
                agent.load(stage_dir)
                metrics = run_episode(env_s11, agent)
                _env_module.EXTENDED_STATE = False
                _env_module.STATE_DIM = 5
                _env_module.DROP_PROBS = _DROP_PROBS_6
            else:
                print(f"  No saved model for {stage} — skipping.")
                metrics = {"note": "not trained yet"}
        else:
            # All other stages use MAPPO with 5-dim state, 6-action space
            agent = MAPPO(state_dim=5, joint_dim=15, action_dim=6)
            if stage_dir.exists():
                agent.load(stage_dir)
                metrics = run_episode(env, agent)
            else:
                print(f"  No saved model for {stage} — skipping.")
                metrics = {"note": "not trained yet"}

        results[stage] = {
            "description": description,
            "metrics":     metrics,
        }
        print(f"  Result: {metrics}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(tag: str = "baseline", model_tag: str | None = None) -> None:
    out_dir  = EVAL_DIR / tag
    mappo_dir = MARL_DIR / "experiments" / model_tag if model_tag else MARL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {out_dir}")

    # Load bundle and models
    print(f"Loading LGBM bundle from {LGBM_BUNDLE}...")
    bundle   = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]

    print(f"Loading trained MAPPO from {mappo_dir}...")
    mappo = MAPPO(state_dim=5, joint_dim=15, action_dim=6)
    mappo.load(mappo_dir)

    # ── Monday evaluation (seen attacks — same day as training)
    monday_metrics = evaluate_day(
        csv_path = CSV_MONDAY,
        bundle   = bundle,
        mappo    = mappo,
        label    = "Monday (seen attacks)",
    )
    with open(out_dir / "monday_results.json", "w") as f:
        json.dump(monday_metrics, f, indent=2)

    # ── Tuesday evaluation (unseen attacks — generalization test)
    tuesday_metrics = evaluate_day(
        csv_path = CSV_TUESDAY,
        bundle   = bundle,
        mappo    = mappo,
        label    = "Tuesday (unseen attacks)",
    )
    with open(out_dir / "tuesday_results.json", "w") as f:
        json.dump(tuesday_metrics, f, indent=2)

    # ── Comparison table
    print("\n" + "="*60)
    print("MONDAY vs TUESDAY COMPARISON")
    print("="*60)
    print(f"{'Metric':<22} {'Monday':>12} {'Tuesday':>12} {'Delta':>10}")
    print("-"*60)
    for key in ["total_reward", "tpr", "fpr", "legit_ratio"]:
        m = monday_metrics.get(key, 0)
        t = tuesday_metrics.get(key, 0)
        delta = t - m if isinstance(m, (int, float)) else 0
        print(f"{key:<22} {m:>12} {t:>12} {delta:>10.4f}")

    # ── IQL evaluation (S6 ablation model)
    iql_dir = MARL_DIR / "ablation" / "S6"
    iql_monday_metrics  = {}
    iql_tuesday_metrics = {}

    if iql_dir.exists():
        print(f"\nLoading trained IQL from {iql_dir}...")
        iql = IQL(state_dim=5, action_dim=6)
        iql.load(iql_dir)

        iql_monday_metrics = evaluate_day_iql(
            csv_path = CSV_MONDAY,
            bundle   = bundle,
            iql      = iql,
            label    = "Monday — IQL (seen attacks)",
        )
        with open(out_dir / "iql_monday_results.json", "w") as f:
            json.dump(iql_monday_metrics, f, indent=2)

        iql_tuesday_metrics = evaluate_day_iql(
            csv_path = CSV_TUESDAY,
            bundle   = bundle,
            iql      = iql,
            label    = "Tuesday — IQL (unseen attacks)",
        )
        with open(out_dir / "iql_tuesday_results.json", "w") as f:
            json.dump(iql_tuesday_metrics, f, indent=2)
    else:
        print(f"\nNo IQL model found at {iql_dir} — skipping IQL evaluation.")
        print("Run: python main.py --stage iql")

    # ── IPPO evaluation (S7 ablation model)
    ippo_dir = MARL_DIR / "ablation" / "S7"
    ippo_monday_metrics  = {}
    ippo_tuesday_metrics = {}

    if ippo_dir.exists():
        print(f"\nLoading trained IPPO from {ippo_dir}...")
        ippo = IPPO(state_dim=5, action_dim=6)
        ippo.load(ippo_dir)

        ippo_monday_metrics = evaluate_day_ippo(
            csv_path = CSV_MONDAY,
            bundle   = bundle,
            ippo     = ippo,
            label    = "Monday — IPPO (seen attacks)",
        )
        with open(out_dir / "ippo_monday_results.json", "w") as f:
            json.dump(ippo_monday_metrics, f, indent=2)

        ippo_tuesday_metrics = evaluate_day_ippo(
            csv_path = CSV_TUESDAY,
            bundle   = bundle,
            ippo     = ippo,
            label    = "Tuesday — IPPO (unseen attacks)",
        )
        with open(out_dir / "ippo_tuesday_results.json", "w") as f:
            json.dump(ippo_tuesday_metrics, f, indent=2)
    else:
        print(f"\nNo IPPO model found at {ippo_dir} — skipping IPPO evaluation.")
        print("Run: python main.py --stage ippo")

    # ── MAPL evaluation (S9 — MAPPO + local reward)
    from src.mappo import MAPPO as _MAPPO
    mapl_dir = MARL_DIR / "ablation" / "S9"
    mapl_monday_metrics  = {}
    mapl_tuesday_metrics = {}

    if mapl_dir.exists():
        print(f"\nLoading trained MAPL (S9) from {mapl_dir}...")
        mapl = _MAPPO(state_dim=5, joint_dim=15, action_dim=6)
        mapl.load(mapl_dir)

        mapl_monday_metrics = evaluate_day(
            csv_path = CSV_MONDAY,
            bundle   = bundle,
            mappo    = mapl,
            label    = "Monday — MAPL/S9 (local reward, seen attacks)",
        )
        with open(out_dir / "mapl_monday_results.json", "w") as f:
            json.dump(mapl_monday_metrics, f, indent=2)

        mapl_tuesday_metrics = evaluate_day(
            csv_path = CSV_TUESDAY,
            bundle   = bundle,
            mappo    = mapl,
            label    = "Tuesday — MAPL/S9 (local reward, unseen attacks)",
        )
        with open(out_dir / "mapl_tuesday_results.json", "w") as f:
            json.dump(mapl_tuesday_metrics, f, indent=2)
    else:
        print(f"\nNo MAPL model found at {mapl_dir} — skipping.")
        print("Run: python main.py --stage mapl")

    # ── E13 evaluation (S10 — MAPPO + extended 7-dim state)
    e13_dir = MARL_DIR / "ablation" / "S10"
    e13_monday_metrics  = {}
    e13_tuesday_metrics = {}

    if e13_dir.exists():
        print(f"\nLoading trained E13 (S10) from {e13_dir}...")
        # E13 uses 7-dim state / 21-dim joint — instantiate explicitly
        # and temporarily enable EXTENDED_STATE + STATE_DIM in the environment module.
        _env_module.EXTENDED_STATE = True
        _env_module.STATE_DIM = 7
        e13_mappo = MAPPO(state_dim=7, joint_dim=21, action_dim=6)
        e13_mappo.load(e13_dir)

        e13_monday_metrics = evaluate_day(
            csv_path = CSV_MONDAY,
            bundle   = bundle,
            mappo    = e13_mappo,
            label    = "Monday — E13/S10 (extended state, seen attacks)",
        )
        with open(out_dir / "e13_monday_results.json", "w") as f:
            json.dump(e13_monday_metrics, f, indent=2)

        e13_tuesday_metrics = evaluate_day(
            csv_path = CSV_TUESDAY,
            bundle   = bundle,
            mappo    = e13_mappo,
            label    = "Tuesday — E13/S10 (extended state, unseen attacks)",
        )
        with open(out_dir / "e13_tuesday_results.json", "w") as f:
            json.dump(e13_tuesday_metrics, f, indent=2)

        # Restore 5-dim for ablation study (which loads S8 baseline)
        _env_module.EXTENDED_STATE = False
        _env_module.STATE_DIM = 5
    else:
        print(f"\nNo E13 model found at {e13_dir} — skipping.")
        print("Run: python main.py --stage e13")

    # ── S11 evaluation (MAPPO + 10-action space)
    s11_monday_metrics  = {}
    s11_tuesday_metrics = {}

    if S11_DIR.exists():
        print(f"\nLoading trained S11 from {S11_DIR}...")
        # S11 was trained with EXTENDED_STATE=True (7-dim) + 10-action space.
        _env_module.EXTENDED_STATE = True
        _env_module.STATE_DIM = 7
        _env_module.DROP_PROBS = _DROP_PROBS_10
        s11_mappo = MAPPO(state_dim=7, joint_dim=21, action_dim=10)
        s11_mappo.load(S11_DIR)

        s11_monday_metrics = evaluate_day(
            csv_path = CSV_MONDAY,
            bundle   = bundle,
            mappo    = s11_mappo,
            label    = "Monday — S11 (10-action, seen attacks)",
        )
        with open(out_dir / "s11_monday_results.json", "w") as f:
            json.dump(s11_monday_metrics, f, indent=2)

        s11_tuesday_metrics = evaluate_day(
            csv_path = CSV_TUESDAY,
            bundle   = bundle,
            mappo    = s11_mappo,
            label    = "Tuesday — S11 (10-action, unseen attacks)",
        )
        with open(out_dir / "s11_tuesday_results.json", "w") as f:
            json.dump(s11_tuesday_metrics, f, indent=2)

        # Restore 5-dim state + 6-action DROP_PROBS for ablation study
        _env_module.EXTENDED_STATE = False
        _env_module.STATE_DIM = 5
        _env_module.DROP_PROBS = _DROP_PROBS_6
    else:
        print(f"\nNo S11 model found at {S11_DIR} — skipping.")
        print("Run: python main.py --stage s11")

    # ── S12 evaluation (MAPPO + no puzzle, 9-action space)
    s12_monday_metrics  = {}
    s12_tuesday_metrics = {}

    if S12_DIR.exists():
        print(f"\nLoading trained S12 from {S12_DIR}...")
        # S12: 5-dim state, 9-action (no puzzle). Disable puzzle in env module.
        _env_module.EXTENDED_STATE  = False
        _env_module.STATE_DIM       = 5
        _env_module.PUZZLE_ENABLED  = False
        _env_module.DROP_PROBS      = _DROP_PROBS_s12
        s12_mappo = MAPPO(state_dim=5, joint_dim=15, action_dim=9)
        s12_mappo.load(S12_DIR)

        s12_monday_metrics = evaluate_day(
            csv_path = CSV_MONDAY,
            bundle   = bundle,
            mappo    = s12_mappo,
            label    = "Monday — S12 (no-puzzle, seen attacks)",
        )
        with open(out_dir / "s12_monday_results.json", "w") as f:
            json.dump(s12_monday_metrics, f, indent=2)

        s12_tuesday_metrics = evaluate_day(
            csv_path = CSV_TUESDAY,
            bundle   = bundle,
            mappo    = s12_mappo,
            label    = "Tuesday — S12 (no-puzzle, unseen attacks)",
        )
        with open(out_dir / "s12_tuesday_results.json", "w") as f:
            json.dump(s12_tuesday_metrics, f, indent=2)

        # Restore puzzle + 6-action DROP_PROBS for ablation study
        _env_module.PUZZLE_ENABLED = True
        _env_module.DROP_PROBS     = _DROP_PROBS_6
    else:
        print(f"\nNo S12 model found at {S12_DIR} — skipping.")
        print("Run: python main.py --stage s12")

    # ── Full 7×2 comparison table
    print("\n" + "="*150)
    print("FULL COMPARISON: MAPPO (S8) vs IPPO (S7) vs MAPL (S9) vs E13 (S10) vs S11 (10-act) vs S12 (no-puzzle) vs IQL (S6)")
    print("="*150)
    print(f"{'Metric':<18} {'MAPPO Mon':>11} {'IPPO Mon':>10} {'MAPL Mon':>10} {'E13 Mon':>9} "
          f"{'S11 Mon':>9} {'S12 Mon':>9} {'IQL Mon':>10} "
          f"{'MAPPO Tue':>11} {'IPPO Tue':>10} {'MAPL Tue':>10} {'E13 Tue':>9} "
          f"{'S11 Tue':>9} {'S12 Tue':>9} {'IQL Tue':>10}")
    print("-"*150)
    for key in ["total_reward", "tpr", "fpr", "legit_ratio"]:
        mm  = monday_metrics.get(key, "-")
        pm  = ippo_monday_metrics.get(key, "-")
        lm  = mapl_monday_metrics.get(key, "-")
        em  = e13_monday_metrics.get(key, "-")
        sm  = s11_monday_metrics.get(key, "-")
        xm  = s12_monday_metrics.get(key, "-")
        im  = iql_monday_metrics.get(key, "-")
        mt  = tuesday_metrics.get(key, "-")
        pt  = ippo_tuesday_metrics.get(key, "-")
        lt  = mapl_tuesday_metrics.get(key, "-")
        et  = e13_tuesday_metrics.get(key, "-")
        st  = s11_tuesday_metrics.get(key, "-")
        xt  = s12_tuesday_metrics.get(key, "-")
        it  = iql_tuesday_metrics.get(key, "-")
        print(f"{key:<18} {str(mm):>11} {str(pm):>10} {str(lm):>10} {str(em):>9} "
              f"{str(sm):>9} {str(xm):>9} {str(im):>10} "
              f"{str(mt):>11} {str(pt):>10} {str(lt):>10} {str(et):>9} "
              f"{str(st):>9} {str(xt):>9} {str(it):>10}")

    # ── Ablation study
    df_monday = load_day(CSV_MONDAY, features)
    ablation  = ablation_study(df_monday, bundle)
    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(ablation, f, indent=2)

    print(f"\nAll results saved to: {out_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="baseline")
    p.add_argument("--model_tag", type=str, default=None)
    args = p.parse_args()
    main(tag=args.tag, model_tag=args.model_tag)
