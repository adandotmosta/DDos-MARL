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
        CSV_MONDAY, CSV_TUESDAY, CSV_THURSDAY, CSV_SATURDAY, LGBM_BUNDLE,
        MARL_DIR, EVAL_DIR, N_AGENTS, WINDOW_SIZE,
        ABLATION_STAGES, ACTIONS, S11_DIR, S12_DIR, S13_DIR,
        S14_DIR, S15_DIR, S16_DIR, S17_DIR, S18_DIR,
        S19_DIR, S21_DIR, S22_DIR, S23_DIR, S24_DIR, S25_DIR, S26_DIR, S26B_DIR, S27_DIR, S28_DIR, S29_DIR,
        LGBM_BUNDLE_COMBINED,
    )
    from src.environment import MultiAgentEnv
    import src.environment as _env_module
    from src.mappo import MAPPO
    from src.iql import IQL
    from src.ippo import IPPO
except ImportError:
    from coopids.config import (
        CSV_MONDAY, CSV_TUESDAY, CSV_THURSDAY, CSV_SATURDAY, LGBM_BUNDLE,
        MARL_DIR, EVAL_DIR, N_AGENTS, WINDOW_SIZE,
        ABLATION_STAGES, ACTIONS, S11_DIR, S12_DIR,
        S13_DIR, S14_DIR, S15_DIR, S16_DIR, S17_DIR, S18_DIR,
        S19_DIR, S21_DIR, S22_DIR, S23_DIR, S24_DIR, S25_DIR, S26B_DIR, S27_DIR, S28_DIR, S29_DIR,
        LGBM_BUNDLE_COMBINED,
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
    action_labels: dict = None,
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

        total_reward += float(np.mean(reward)) if isinstance(reward, list) else reward
        total_steps  += 1
        local_states  = next_local
        joint_obs     = next_joint

    # ── Metrics
    tpr         = total_attack_blocked / total_attack_in if total_attack_in > 0 else 0.0
    legit_ratio = total_benign_out     / total_benign_in if total_benign_in > 0 else 0.0
    fpr         = 1.0 - legit_ratio   # fraction of legit traffic incorrectly blocked

    # Convert action counts to labelled dicts
    labels = action_labels if action_labels is not None else ACTIONS
    action_dist = [
        {labels.get(k, str(k)): v for k, v in sorted(ac.items())}
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

        total_reward += float(np.mean(reward)) if isinstance(reward, list) else reward
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

        total_reward += float(np.mean(reward)) if isinstance(reward, list) else reward
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
        elif stage == "S12":
            # S12: 5-dim state, 9-action space (no puzzle)
            if stage_dir.exists():
                _env_module.PUZZLE_ENABLED = False
                _env_module.DROP_PROBS = _DROP_PROBS_s12
                env_s12 = MultiAgentEnv(df=df_monday, bundle=bundle, n_agents=N_AGENTS)
                agent = MAPPO(state_dim=5, joint_dim=15, action_dim=9)
                agent.load(stage_dir)
                metrics = run_episode(env_s12, agent)
                _env_module.PUZZLE_ENABLED = True
                _env_module.DROP_PROBS = _DROP_PROBS_6
            else:
                print(f"  No saved model for {stage} — skipping.")
                metrics = {"note": "not trained yet"}
        elif stage in ("S14", "S17"):
            # S14/S17: 6-dim state (puzzle_block_rate or entropy)
            if stage_dir.exists():
                if stage == "S14":
                    _env_module.PUZZLE_STATE = True
                else:
                    _env_module.ENTROPY_STATE = True
                agent = MAPPO(state_dim=6, joint_dim=18, action_dim=6)
                agent.load(stage_dir)
                env_6d = MultiAgentEnv(df=df_monday, bundle=bundle, n_agents=N_AGENTS)
                metrics = run_episode(env_6d, agent)
                _env_module.PUZZLE_STATE = False
                _env_module.ENTROPY_STATE = False
            else:
                print(f"  No saved model for {stage} — skipping.")
                metrics = {"note": "not trained yet"}
        elif stage == "S15":
            # S15: risk-aware throttle
            if stage_dir.exists():
                _env_module.SMART_THROTTLE = True
                agent = MAPPO(state_dim=5, joint_dim=15, action_dim=6)
                agent.load(stage_dir)
                metrics = run_episode(env, agent)
                _env_module.SMART_THROTTLE = False
            else:
                print(f"  No saved model for {stage} — skipping.")
                metrics = {"note": "not trained yet"}
        elif stage == "S16":
            # S16: dynamic puzzle
            if stage_dir.exists():
                _env_module.DYNAMIC_PUZZLE = True
                agent = MAPPO(state_dim=5, joint_dim=15, action_dim=6)
                agent.load(stage_dir)
                metrics = run_episode(env, agent)
                _env_module.DYNAMIC_PUZZLE = False
            else:
                print(f"  No saved model for {stage} — skipping.")
                metrics = {"note": "not trained yet"}
        elif stage == "S19":
            # S19: asymmetric reward (attack_w=3.0)
            if stage_dir.exists():
                _env_module.REWARD_ATTACK_W = 3.0
                agent = MAPPO(state_dim=5, joint_dim=15, action_dim=6)
                agent.load(stage_dir)
                metrics = run_episode(env, agent)
                _env_module.REWARD_ATTACK_W = 1.0
            else:
                print(f"  No saved model for {stage} — skipping.")
                metrics = {"note": "not trained yet"}
        elif stage == "S21":
            # S21: lower BASE_THRESHOLD (0.20)
            if stage_dir.exists():
                _env_module.BASE_THRESHOLD = 0.20
                env_s21 = MultiAgentEnv(df=df_monday, bundle=bundle, n_agents=N_AGENTS)
                agent = MAPPO(state_dim=5, joint_dim=15, action_dim=6)
                agent.load(stage_dir)
                metrics = run_episode(env_s21, agent)
                _env_module.BASE_THRESHOLD = 0.35
            else:
                print(f"  No saved model for {stage} — skipping.")
                metrics = {"note": "not trained yet"}
        elif stage in ("S23", "S24"):
            # S23/S24: selective puzzle, 10-action space
            if stage_dir.exists():
                _env_module.SELECTIVE_PUZZLE = True
                _env_module.LOCAL_REWARD     = True
                agent = MAPPO(state_dim=5, joint_dim=15, action_dim=10)
                agent.load(stage_dir)
                metrics = run_episode(env, agent)
                _env_module.SELECTIVE_PUZZLE = False
                _env_module.LOCAL_REWARD     = False
            else:
                print(f"  No saved model for {stage} — skipping.")
                metrics = {"note": "not trained yet"}
        elif stage == "S25":
            # S25: puzzle on flagged IPs only + p_attack pass rates + combined LightGBM
            if stage_dir.exists() and LGBM_BUNDLE_COMBINED.exists():
                with open(LGBM_BUNDLE_COMBINED, "rb") as _f:
                    _bundle_combined = pickle.load(_f)
                _env_module.PUZZLE_FLAGGED_ONLY = True
                _env_module.LOCAL_REWARD        = True
                env_s25 = MultiAgentEnv(df=df_monday, bundle=_bundle_combined, n_agents=N_AGENTS)
                agent = MAPPO(state_dim=5, joint_dim=15, action_dim=10)
                agent.load(stage_dir)
                metrics = run_episode(env_s25, agent)
                _env_module.PUZZLE_FLAGGED_ONLY = False
                _env_module.LOCAL_REWARD        = False
            else:
                print(f"  No saved model or combined bundle for {stage} — skipping.")
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

def main(tag: str = "baseline", model_tag: str | None = None, models: set | None = None) -> None:
    """
    models: optional set of lowercase model names to evaluate, e.g. {"s23", "s24", "ippo"}.
            If None, all models are evaluated.
    Valid names: mappo, iql, ippo, mapl, e13, s11, s12, s13, s14, s15, s16,
                 s17, s18, s19, s21, s22, s23, s24, s25, s26, s26b, s27, s28, s29
    """
    def _should_run(name: str) -> bool:
        return models is None or name.lower() in models

    out_dir  = EVAL_DIR / tag
    mappo_dir = MARL_DIR / "experiments" / model_tag if model_tag else MARL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {out_dir}")

    # Load bundle and models
    print(f"Loading LGBM bundle from {LGBM_BUNDLE}...")
    bundle   = load_bundle(LGBM_BUNDLE)
    features = bundle["features"]

    monday_metrics  = {}
    tuesday_metrics = {}
    if _should_run("mappo"):
        print(f"Loading trained MAPPO from {mappo_dir}...")
        mappo = MAPPO(state_dim=5, joint_dim=15, action_dim=6)
        mappo.load(mappo_dir)

        monday_metrics = evaluate_day(
            csv_path = CSV_MONDAY,
            bundle   = bundle,
            mappo    = mappo,
            label    = "Monday (seen attacks)",
        )
        with open(out_dir / "monday_results.json", "w") as f:
            json.dump(monday_metrics, f, indent=2)

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

    if _should_run("iql") and iql_dir.exists():
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

    if _should_run("ippo") and ippo_dir.exists():
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

    if _should_run("mapl") and mapl_dir.exists():
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

    if _should_run("e13") and e13_dir.exists():
        print(f"\nLoading trained E13 (S10) from {e13_dir}...")
        # E13 uses 7-dim state / 21-dim joint — instantiate explicitly
        # and temporarily enable EXTENDED_STATE + STATE_DIM in the environment module.
        # Explicitly reset all other state-altering flags to prevent leakage from
        # the ablation study section (S14 sets PUZZLE_STATE, S17 sets ENTROPY_STATE).
        _env_module.PUZZLE_STATE    = False
        _env_module.ENTROPY_STATE   = False
        _env_module.EXTENDED_STATE  = True
        _env_module.STATE_DIM       = 5   # env adds 2 for extended → 7 total
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

    if _should_run("s11") and S11_DIR.exists():
        print(f"\nLoading trained S11 from {S11_DIR}...")
        _env_module.PUZZLE_STATE    = False
        _env_module.ENTROPY_STATE   = False
        _env_module.EXTENDED_STATE  = True
        _env_module.STATE_DIM       = 7
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

    if _should_run("s12") and S12_DIR.exists():
        print(f"\nLoading trained S12 from {S12_DIR}...")
        # S12: 5-dim state, 9-action (no puzzle). Disable puzzle in env module.
        _env_module.PUZZLE_STATE    = False
        _env_module.ENTROPY_STATE   = False
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

    # ── S13 evaluation (MAPPO + local reward + puzzle cost 0.05)
    s13_monday_metrics  = {}
    s13_tuesday_metrics = {}

    if _should_run("s13") and S13_DIR.exists():
        print(f"\nLoading trained S13 from {S13_DIR}...")
        _env_module.EXTENDED_STATE = False
        _env_module.STATE_DIM      = 5
        _env_module.PUZZLE_ENABLED = True
        _env_module.DROP_PROBS     = _DROP_PROBS_6
        s13_mappo = MAPPO(state_dim=5, joint_dim=15, action_dim=6)
        s13_mappo.load(S13_DIR)

        s13_monday_metrics = evaluate_day(
            csv_path = CSV_MONDAY,
            bundle   = bundle,
            mappo    = s13_mappo,
            label    = "Monday — S13 (MAPL + puzzle cost 0.05, seen attacks)",
        )
        with open(out_dir / "s13_monday_results.json", "w") as f:
            json.dump(s13_monday_metrics, f, indent=2)

        s13_tuesday_metrics = evaluate_day(
            csv_path = CSV_TUESDAY,
            bundle   = bundle,
            mappo    = s13_mappo,
            label    = "Tuesday — S13 (MAPL + puzzle cost 0.05, unseen attacks)",
        )
        with open(out_dir / "s13_tuesday_results.json", "w") as f:
            json.dump(s13_tuesday_metrics, f, indent=2)
    else:
        print(f"\nNo S13 model found at {S13_DIR} — skipping.")
        print("Run: python main.py --stage s13")

    # ── S14 evaluation (MAPPO + local reward + puzzle_block_rate state)
    s14_monday_metrics  = {}
    s14_tuesday_metrics = {}

    if _should_run("s14") and S14_DIR.exists():
        print(f"\nLoading trained S14 from {S14_DIR}...")
        _env_module.PUZZLE_STATE = True
        s14_mappo = MAPPO(state_dim=6, joint_dim=18, action_dim=6)
        s14_mappo.load(S14_DIR)

        s14_monday_metrics = evaluate_day(
            csv_path = CSV_MONDAY,
            bundle   = bundle,
            mappo    = s14_mappo,
            label    = "Monday — S14 (puzzle block rate state, seen attacks)",
        )
        with open(out_dir / "s14_monday_results.json", "w") as f:
            json.dump(s14_monday_metrics, f, indent=2)

        s14_tuesday_metrics = evaluate_day(
            csv_path = CSV_TUESDAY,
            bundle   = bundle,
            mappo    = s14_mappo,
            label    = "Tuesday — S14 (puzzle block rate state, unseen attacks)",
        )
        with open(out_dir / "s14_tuesday_results.json", "w") as f:
            json.dump(s14_tuesday_metrics, f, indent=2)

        _env_module.PUZZLE_STATE = False
    else:
        print(f"\nNo S14 model found at {S14_DIR} — skipping.")
        print("Run: python main.py --stage s14")

    # ── S15 evaluation (MAPPO + local reward + risk-aware throttle)
    s15_monday_metrics  = {}
    s15_tuesday_metrics = {}

    if _should_run("s15") and S15_DIR.exists():
        print(f"\nLoading trained S15 from {S15_DIR}...")
        _env_module.SMART_THROTTLE = True
        s15_mappo = MAPPO(state_dim=5, joint_dim=15, action_dim=6)
        s15_mappo.load(S15_DIR)

        s15_monday_metrics = evaluate_day(
            csv_path = CSV_MONDAY,
            bundle   = bundle,
            mappo    = s15_mappo,
            label    = "Monday — S15 (risk-aware throttle, seen attacks)",
        )
        with open(out_dir / "s15_monday_results.json", "w") as f:
            json.dump(s15_monday_metrics, f, indent=2)

        s15_tuesday_metrics = evaluate_day(
            csv_path = CSV_TUESDAY,
            bundle   = bundle,
            mappo    = s15_mappo,
            label    = "Tuesday — S15 (risk-aware throttle, unseen attacks)",
        )
        with open(out_dir / "s15_tuesday_results.json", "w") as f:
            json.dump(s15_tuesday_metrics, f, indent=2)

        _env_module.SMART_THROTTLE = False
    else:
        print(f"\nNo S15 model found at {S15_DIR} — skipping.")
        print("Run: python main.py --stage s15")

    # ── S16 evaluation (MAPPO + local reward + dynamic puzzle)
    s16_monday_metrics  = {}
    s16_tuesday_metrics = {}

    if _should_run("s16") and S16_DIR.exists():
        print(f"\nLoading trained S16 from {S16_DIR}...")
        _env_module.DYNAMIC_PUZZLE = True
        s16_mappo = MAPPO(state_dim=5, joint_dim=15, action_dim=6)
        s16_mappo.load(S16_DIR)

        s16_monday_metrics = evaluate_day(
            csv_path = CSV_MONDAY,
            bundle   = bundle,
            mappo    = s16_mappo,
            label    = "Monday — S16 (dynamic puzzle, seen attacks)",
        )
        with open(out_dir / "s16_monday_results.json", "w") as f:
            json.dump(s16_monday_metrics, f, indent=2)

        s16_tuesday_metrics = evaluate_day(
            csv_path = CSV_TUESDAY,
            bundle   = bundle,
            mappo    = s16_mappo,
            label    = "Tuesday — S16 (dynamic puzzle, unseen attacks)",
        )
        with open(out_dir / "s16_tuesday_results.json", "w") as f:
            json.dump(s16_tuesday_metrics, f, indent=2)

        _env_module.DYNAMIC_PUZZLE = False
    else:
        print(f"\nNo S16 model found at {S16_DIR} — skipping.")
        print("Run: python main.py --stage s16")

    # ── S17 evaluation (MAPPO + local reward + entropy state)
    s17_monday_metrics  = {}
    s17_tuesday_metrics = {}

    if _should_run("s17") and S17_DIR.exists():
        print(f"\nLoading trained S17 from {S17_DIR}...")
        _env_module.ENTROPY_STATE = True
        s17_mappo = MAPPO(state_dim=6, joint_dim=18, action_dim=6)
        s17_mappo.load(S17_DIR)

        s17_monday_metrics = evaluate_day(
            csv_path = CSV_MONDAY,
            bundle   = bundle,
            mappo    = s17_mappo,
            label    = "Monday — S17 (entropy state, seen attacks)",
        )
        with open(out_dir / "s17_monday_results.json", "w") as f:
            json.dump(s17_monday_metrics, f, indent=2)

        s17_tuesday_metrics = evaluate_day(
            csv_path = CSV_TUESDAY,
            bundle   = bundle,
            mappo    = s17_mappo,
            label    = "Tuesday — S17 (entropy state, unseen attacks)",
        )
        with open(out_dir / "s17_tuesday_results.json", "w") as f:
            json.dump(s17_tuesday_metrics, f, indent=2)

        _env_module.ENTROPY_STATE = False
    else:
        print(f"\nNo S17 model found at {S17_DIR} — skipping.")
        print("Run: python main.py --stage s17")

    # ── S18 evaluation (MAPPO + local reward + curriculum learning)
    s18_monday_metrics  = {}
    s18_tuesday_metrics = {}

    if _should_run("s18") and S18_DIR.exists():
        print(f"\nLoading trained S18 from {S18_DIR}...")
        # Eval with puzzle enabled (Phase 2 policy)
        _env_module.PUZZLE_ENABLED = True
        s18_mappo = MAPPO(state_dim=5, joint_dim=15, action_dim=6)
        s18_mappo.load(S18_DIR)

        s18_monday_metrics = evaluate_day(
            csv_path = CSV_MONDAY,
            bundle   = bundle,
            mappo    = s18_mappo,
            label    = "Monday — S18 (curriculum learning, seen attacks)",
        )
        with open(out_dir / "s18_monday_results.json", "w") as f:
            json.dump(s18_monday_metrics, f, indent=2)

        s18_tuesday_metrics = evaluate_day(
            csv_path = CSV_TUESDAY,
            bundle   = bundle,
            mappo    = s18_mappo,
            label    = "Tuesday — S18 (curriculum learning, unseen attacks)",
        )
        with open(out_dir / "s18_tuesday_results.json", "w") as f:
            json.dump(s18_tuesday_metrics, f, indent=2)
    else:
        print(f"\nNo S18 model found at {S18_DIR} — skipping.")
        print("Run: python main.py --stage s18")

    # ── S19 evaluation (MAPPO + local reward + asymmetric reward, attack_w=3.0)
    s19_monday_metrics  = {}
    s19_tuesday_metrics = {}

    if _should_run("s19") and S19_DIR.exists():
        print(f"\nLoading trained S19 from {S19_DIR}...")
        _env_module.REWARD_ATTACK_W = 3.0
        s19_mappo = MAPPO(state_dim=5, joint_dim=15, action_dim=6)
        s19_mappo.load(S19_DIR)

        s19_monday_metrics = evaluate_day(
            csv_path = CSV_MONDAY,
            bundle   = bundle,
            mappo    = s19_mappo,
            label    = "Monday — S19 (asymmetric reward attack_w=3, seen attacks)",
        )
        with open(out_dir / "s19_monday_results.json", "w") as f:
            json.dump(s19_monday_metrics, f, indent=2)

        s19_tuesday_metrics = evaluate_day(
            csv_path = CSV_TUESDAY,
            bundle   = bundle,
            mappo    = s19_mappo,
            label    = "Tuesday — S19 (asymmetric reward attack_w=3, unseen attacks)",
        )
        with open(out_dir / "s19_tuesday_results.json", "w") as f:
            json.dump(s19_tuesday_metrics, f, indent=2)

        _env_module.REWARD_ATTACK_W = 1.0
    else:
        print(f"\nNo S19 model found at {S19_DIR} — skipping.")
        print("Run: python main.py --stage s19")

    # ── S21 evaluation (MAPPO + local reward + lower threshold 0.20)
    s21_monday_metrics  = {}
    s21_tuesday_metrics = {}

    if _should_run("s21") and S21_DIR.exists():
        print(f"\nLoading trained S21 from {S21_DIR}...")
        _env_module.BASE_THRESHOLD = 0.20
        s21_mappo = MAPPO(state_dim=5, joint_dim=15, action_dim=6)
        s21_mappo.load(S21_DIR)

        s21_monday_metrics = evaluate_day(
            csv_path = CSV_MONDAY,
            bundle   = bundle,
            mappo    = s21_mappo,
            label    = "Monday — S21 (lower threshold 0.20, seen attacks)",
        )
        with open(out_dir / "s21_monday_results.json", "w") as f:
            json.dump(s21_monday_metrics, f, indent=2)

        s21_tuesday_metrics = evaluate_day(
            csv_path = CSV_TUESDAY,
            bundle   = bundle,
            mappo    = s21_mappo,
            label    = "Tuesday — S21 (lower threshold 0.20, unseen attacks)",
        )
        with open(out_dir / "s21_tuesday_results.json", "w") as f:
            json.dump(s21_tuesday_metrics, f, indent=2)

        _env_module.BASE_THRESHOLD = 0.35
    else:
        print(f"\nNo S21 model found at {S21_DIR} — skipping.")
        print("Run: python main.py --stage s21")

    # ── S22 evaluation (MAPPO + counterfactual reward)
    s22_monday_metrics  = {}
    s22_tuesday_metrics = {}

    if _should_run("s22") and S22_DIR.exists():
        print(f"\nLoading trained S22 from {S22_DIR}...")
        # S22 trains with LOCAL_REWARD=True (to trigger per_agent tracking),
        # but eval uses standard global reward for fair comparison
        s22_mappo = MAPPO(state_dim=5, joint_dim=15, action_dim=6)
        s22_mappo.load(S22_DIR)

        s22_monday_metrics = evaluate_day(
            csv_path = CSV_MONDAY,
            bundle   = bundle,
            mappo    = s22_mappo,
            label    = "Monday — S22 (counterfactual reward, seen attacks)",
        )
        with open(out_dir / "s22_monday_results.json", "w") as f:
            json.dump(s22_monday_metrics, f, indent=2)

        s22_tuesday_metrics = evaluate_day(
            csv_path = CSV_TUESDAY,
            bundle   = bundle,
            mappo    = s22_mappo,
            label    = "Tuesday — S22 (counterfactual reward, unseen attacks)",
        )
        with open(out_dir / "s22_tuesday_results.json", "w") as f:
            json.dump(s22_tuesday_metrics, f, indent=2)
    else:
        print(f"\nNo S22 model found at {S22_DIR} — skipping.")
        print("Run: python main.py --stage s22")

    # ── S23 evaluation (MAPPO + selective puzzle — suspicious-only)
    s23_monday_metrics  = {}
    s23_tuesday_metrics = {}

    if _should_run("s23") and S23_DIR.exists():
        print(f"\nLoading trained S23 from {S23_DIR}...")
        # S23 trains with SELECTIVE_PUZZLE=True — must patch during eval so puzzle
        # behaves the same way the agent was trained under.
        _env_module.SELECTIVE_PUZZLE = True
        _env_module.LOCAL_REWARD     = True

        s23_mappo = MAPPO(state_dim=5, joint_dim=15, action_dim=10)
        s23_mappo.load(S23_DIR)

        s23_monday_metrics = evaluate_day(
            csv_path = CSV_MONDAY,
            bundle   = bundle,
            mappo    = s23_mappo,
            label    = "Monday — S23 (selective puzzle, seen attacks)",
        )
        with open(out_dir / "s23_monday_results.json", "w") as f:
            json.dump(s23_monday_metrics, f, indent=2)

        s23_tuesday_metrics = evaluate_day(
            csv_path = CSV_TUESDAY,
            bundle   = bundle,
            mappo    = s23_mappo,
            label    = "Tuesday — S23 (selective puzzle, unseen attacks)",
        )
        with open(out_dir / "s23_tuesday_results.json", "w") as f:
            json.dump(s23_tuesday_metrics, f, indent=2)

        # Restore defaults
        _env_module.SELECTIVE_PUZZLE = False
        _env_module.LOCAL_REWARD     = False
    else:
        print(f"\nNo S23 model found at {S23_DIR} — skipping.")
        print("Run: python main.py --stage s23")

    # ── S24 evaluation (MAPPO + selective puzzle + combined LightGBM)
    # Isolates MARL throttling quality: LightGBM trained on Mon+Tue → accurate on Tuesday.
    # Compare S24 vs S23 Tuesday TPR to measure how much of the gap is LightGBM's fault.
    s24_monday_metrics  = {}
    s24_tuesday_metrics = {}

    if _should_run("s24") and S24_DIR.exists() and LGBM_BUNDLE_COMBINED.exists():
        print(f"\nLoading trained S24 from {S24_DIR}...")
        print("  (Uses combined Monday+Tuesday LightGBM bundle)")

        # Load the combined bundle for inference
        with open(LGBM_BUNDLE_COMBINED, "rb") as _f:
            bundle_combined = pickle.load(_f)

        _env_module.SELECTIVE_PUZZLE = True
        _env_module.LOCAL_REWARD     = True

        s24_mappo = MAPPO(state_dim=5, joint_dim=15, action_dim=10)
        s24_mappo.load(S24_DIR)

        s24_monday_metrics = evaluate_day(
            csv_path = CSV_MONDAY,
            bundle   = bundle_combined,
            mappo    = s24_mappo,
            label    = "Monday — S24 (selective puzzle + combined LightGBM)",
        )
        with open(out_dir / "s24_monday_results.json", "w") as f:
            json.dump(s24_monday_metrics, f, indent=2)

        s24_tuesday_metrics = evaluate_day(
            csv_path = CSV_TUESDAY,
            bundle   = bundle_combined,
            mappo    = s24_mappo,
            label    = "Tuesday — S24 (selective puzzle + combined LightGBM, MARL policy isolated)",
        )
        with open(out_dir / "s24_tuesday_results.json", "w") as f:
            json.dump(s24_tuesday_metrics, f, indent=2)

        _env_module.SELECTIVE_PUZZLE = False
        _env_module.LOCAL_REWARD     = False
    else:
        if not LGBM_BUNDLE_COMBINED.exists():
            print(f"\nNo combined LightGBM bundle found — skipping S24.")
            print("Run: python train_lgbm_combined.py")
        else:
            print(f"\nNo S24 model found at {S24_DIR} — skipping.")
            print("Run: python main.py --stage s24")

    # ── S25 evaluation (MAPPO + puzzle on flagged IPs + p_attack pass rates + combined LightGBM)
    s25_monday_metrics  = {}
    s25_tuesday_metrics = {}

    if _should_run("s25") and S25_DIR.exists() and LGBM_BUNDLE_COMBINED.exists():
        print(f"\nLoading trained S25 from {S25_DIR}...")
        print("  (Puzzle on flagged IPs only, pass_rate = 1 - p_attack, combined LightGBM)")

        with open(LGBM_BUNDLE_COMBINED, "rb") as _f:
            bundle_combined = pickle.load(_f)

        _env_module.PUZZLE_FLAGGED_ONLY = True
        _env_module.LOCAL_REWARD        = True

        s25_mappo = MAPPO(state_dim=5, joint_dim=15, action_dim=10)
        s25_mappo.load(S25_DIR)

        s25_monday_metrics = evaluate_day(
            csv_path = CSV_MONDAY,
            bundle   = bundle_combined,
            mappo    = s25_mappo,
            label    = "Monday — S25 (flagged-only puzzle + p_attack pass rates)",
        )
        with open(out_dir / "s25_monday_results.json", "w") as f:
            json.dump(s25_monday_metrics, f, indent=2)

        s25_tuesday_metrics = evaluate_day(
            csv_path = CSV_TUESDAY,
            bundle   = bundle_combined,
            mappo    = s25_mappo,
            label    = "Tuesday — S25 (flagged-only puzzle + p_attack pass rates, unseen attacks)",
        )
        with open(out_dir / "s25_tuesday_results.json", "w") as f:
            json.dump(s25_tuesday_metrics, f, indent=2)

        _env_module.PUZZLE_FLAGGED_ONLY = False
        _env_module.LOCAL_REWARD        = False
    else:
        if not LGBM_BUNDLE_COMBINED.exists():
            print(f"\nNo combined LightGBM bundle found — skipping S25.")
            print("Run: python train_lgbm_combined.py")
        else:
            print(f"\nNo S25 model found at {S25_DIR} — skipping.")
            print("Run: python main.py --stage s25")

    # ── S26 evaluation (MAPPO + tiered action profiles)
    # Uses its own environment (environment_tiered.py) — completely different action space.
    s26_monday_metrics  = {}
    s26_tuesday_metrics = {}

    if _should_run("s26") and S26_DIR.exists() and LGBM_BUNDLE_COMBINED.exists():
        print(f"\nLoading trained S26 from {S26_DIR}...")
        print("  (Tiered action profiles: block/puzzle/forward by p_attack, combined LightGBM)")

        from src.environment_tiered import (
            MultiAgentEnv as TieredMultiAgentEnv,
            STATE_DIM as T_STATE_DIM,
            JOINT_DIM as T_JOINT_DIM,
            ACTION_DIM as T_ACTION_DIM,
            ACTION_LABELS as T_ACTION_LABELS,
        )

        with open(LGBM_BUNDLE_COMBINED, "rb") as _f:
            bundle_combined = pickle.load(_f)

        s26_mappo = MAPPO(state_dim=T_STATE_DIM, joint_dim=T_JOINT_DIM, action_dim=T_ACTION_DIM)
        s26_mappo.load(S26_DIR)
        features_combined = bundle_combined["features"]

        # Monday
        df_mon = load_day(CSV_MONDAY, features_combined)
        env_s26_mon = TieredMultiAgentEnv(df=df_mon, bundle=bundle_combined, n_agents=N_AGENTS)
        print(f"\nEvaluating S26 on Monday (seen attacks)...")
        print(df_mon["label"].value_counts().to_string())
        s26_monday_metrics = run_episode(env_s26_mon, s26_mappo, action_labels=T_ACTION_LABELS)
        print(f"  TPR: {s26_monday_metrics['tpr']:.4f}  FPR: {s26_monday_metrics['fpr']:.4f}")
        print(f"  Action distribution per agent:")
        for i, dist in enumerate(s26_monday_metrics["action_distribution"]):
            print(f"    Router {i+1}: {dist}")
        with open(out_dir / "s26_monday_results.json", "w") as f:
            json.dump(s26_monday_metrics, f, indent=2)

        # Tuesday
        df_tue = load_day(CSV_TUESDAY, features_combined)
        env_s26_tue = TieredMultiAgentEnv(df=df_tue, bundle=bundle_combined, n_agents=N_AGENTS)
        print(f"\nEvaluating S26 on Tuesday (unseen attacks)...")
        print(df_tue["label"].value_counts().to_string())
        s26_tuesday_metrics = run_episode(env_s26_tue, s26_mappo, action_labels=T_ACTION_LABELS)
        print(f"  TPR: {s26_tuesday_metrics['tpr']:.4f}  FPR: {s26_tuesday_metrics['fpr']:.4f}")
        print(f"  Action distribution per agent:")
        for i, dist in enumerate(s26_tuesday_metrics["action_distribution"]):
            print(f"    Router {i+1}: {dist}")
        with open(out_dir / "s26_tuesday_results.json", "w") as f:
            json.dump(s26_tuesday_metrics, f, indent=2)

        # Thursday
        if CSV_THURSDAY.exists():
            df_thu = load_day(CSV_THURSDAY, features_combined)
            env_s26_thu = TieredMultiAgentEnv(df=df_thu, bundle=bundle_combined, n_agents=N_AGENTS)
            print(f"\nEvaluating S26 on Thursday (unseen)...")
            print(df_thu["label"].value_counts().to_string())
            s26_thu = run_episode(env_s26_thu, s26_mappo, action_labels=T_ACTION_LABELS)
            print(f"  TPR: {s26_thu['tpr']:.4f}  FPR: {s26_thu['fpr']:.4f}")
            for i, dist in enumerate(s26_thu["action_distribution"]):
                print(f"    Router {i+1}: {dist}")
            with open(out_dir / "s26_thursday_results.json", "w") as f:
                json.dump(s26_thu, f, indent=2)

        # Saturday
        if CSV_SATURDAY.exists():
            df_sat = load_day(CSV_SATURDAY, features_combined)
            env_s26_sat = TieredMultiAgentEnv(df=df_sat, bundle=bundle_combined, n_agents=N_AGENTS)
            print(f"\nEvaluating S26 on Saturday (unseen)...")
            print(df_sat["label"].value_counts().to_string())
            s26_sat = run_episode(env_s26_sat, s26_mappo, action_labels=T_ACTION_LABELS)
            print(f"  TPR: {s26_sat['tpr']:.4f}  FPR: {s26_sat['fpr']:.4f}")
            for i, dist in enumerate(s26_sat["action_distribution"]):
                print(f"    Router {i+1}: {dist}")
            with open(out_dir / "s26_saturday_results.json", "w") as f:
                json.dump(s26_sat, f, indent=2)
    else:
        if not LGBM_BUNDLE_COMBINED.exists():
            print(f"\nNo combined LightGBM bundle found — skipping S26.")
            print("Run: python train_lgbm_combined.py")
        else:
            print(f"\nNo S26 model found at {S26_DIR} — skipping.")
            print("Run: python main.py --stage s26")

    # ── S26b evaluation (S26 + Monday-only LightGBM — isolates MARL contribution)
    s26b_monday_metrics  = {}
    s26b_tuesday_metrics = {}

    if _should_run("s26b") and S26B_DIR.exists():
        print(f"\nLoading trained S26b from {S26B_DIR}...")
        print("  (Tiered action profiles + Monday-only LightGBM — MARL contribution isolation)")

        from src.environment_tiered import (
            MultiAgentEnv as TieredMultiAgentEnv,
            STATE_DIM as T_STATE_DIM,
            JOINT_DIM as T_JOINT_DIM,
            ACTION_DIM as T_ACTION_DIM,
            ACTION_LABELS as T_ACTION_LABELS,
        )

        s26b_mappo = MAPPO(state_dim=T_STATE_DIM, joint_dim=T_JOINT_DIM, action_dim=T_ACTION_DIM)
        s26b_mappo.load(S26B_DIR)
        features_monday = bundle["features"]

        # Monday
        df_mon_b = load_day(CSV_MONDAY, features_monday)
        env_s26b_mon = TieredMultiAgentEnv(df=df_mon_b, bundle=bundle, n_agents=N_AGENTS)
        print(f"\nEvaluating S26b on Monday (seen attacks)...")
        print(df_mon_b["label"].value_counts().to_string())
        s26b_monday_metrics = run_episode(env_s26b_mon, s26b_mappo, action_labels=T_ACTION_LABELS)
        print(f"  TPR: {s26b_monday_metrics['tpr']:.4f}  FPR: {s26b_monday_metrics['fpr']:.4f}")
        print(f"  Action distribution per agent:")
        for i, dist in enumerate(s26b_monday_metrics["action_distribution"]):
            print(f"    Router {i+1}: {dist}")
        with open(out_dir / "s26b_monday_results.json", "w") as f:
            json.dump(s26b_monday_metrics, f, indent=2)

        # Tuesday
        df_tue_b = load_day(CSV_TUESDAY, features_monday)
        env_s26b_tue = TieredMultiAgentEnv(df=df_tue_b, bundle=bundle, n_agents=N_AGENTS)
        print(f"\nEvaluating S26b on Tuesday (unseen attacks)...")
        print(df_tue_b["label"].value_counts().to_string())
        s26b_tuesday_metrics = run_episode(env_s26b_tue, s26b_mappo, action_labels=T_ACTION_LABELS)
        print(f"  TPR: {s26b_tuesday_metrics['tpr']:.4f}  FPR: {s26b_tuesday_metrics['fpr']:.4f}")
        print(f"  Action distribution per agent:")
        for i, dist in enumerate(s26b_tuesday_metrics["action_distribution"]):
            print(f"    Router {i+1}: {dist}")
        with open(out_dir / "s26b_tuesday_results.json", "w") as f:
            json.dump(s26b_tuesday_metrics, f, indent=2)

        # Thursday
        if CSV_THURSDAY.exists():
            df_thu = load_day(CSV_THURSDAY, features_monday)
            env_s26b_thu = TieredMultiAgentEnv(df=df_thu, bundle=bundle, n_agents=N_AGENTS)
            print(f"\nEvaluating S26b on Thursday (unseen)...")
            print(df_thu["label"].value_counts().to_string())
            s26b_thu = run_episode(env_s26b_thu, s26b_mappo, action_labels=T_ACTION_LABELS)
            print(f"  TPR: {s26b_thu['tpr']:.4f}  FPR: {s26b_thu['fpr']:.4f}")
            for i, dist in enumerate(s26b_thu["action_distribution"]):
                print(f"    Router {i+1}: {dist}")
            with open(out_dir / "s26b_thursday_results.json", "w") as f:
                json.dump(s26b_thu, f, indent=2)

        # Saturday
        if CSV_SATURDAY.exists():
            df_sat = load_day(CSV_SATURDAY, features_monday)
            env_s26b_sat = TieredMultiAgentEnv(df=df_sat, bundle=bundle, n_agents=N_AGENTS)
            print(f"\nEvaluating S26b on Saturday (unseen)...")
            print(df_sat["label"].value_counts().to_string())
            s26b_sat = run_episode(env_s26b_sat, s26b_mappo, action_labels=T_ACTION_LABELS)
            print(f"  TPR: {s26b_sat['tpr']:.4f}  FPR: {s26b_sat['fpr']:.4f}")
            for i, dist in enumerate(s26b_sat["action_distribution"]):
                print(f"    Router {i+1}: {dist}")
            with open(out_dir / "s26b_saturday_results.json", "w") as f:
                json.dump(s26b_sat, f, indent=2)
    else:
        print(f"\nNo S26b model found at {S26B_DIR} — skipping.")
        print("Run: python main.py --stage s26b")

    # ── S27 evaluation (p_risk tiered profiles + Monday-only LightGBM)
    # Routing signal: p_risk = 1 - p_benign. Compare vs S26b to isolate routing score.
    s27_monday_metrics  = {}
    s27_tuesday_metrics = {}

    if _should_run("s27") and S27_DIR.exists():
        print(f"\nLoading trained S27 from {S27_DIR}...")
        print("  (p_risk = 1-p_benign routing + Monday-only LightGBM)")

        from src.environment_tiered_v2 import (
            MultiAgentEnv as TieredEnvV2,
            STATE_DIM as T2_STATE_DIM,
            JOINT_DIM as T2_JOINT_DIM,
            ACTION_DIM as T2_ACTION_DIM,
            ACTION_LABELS as T2_ACTION_LABELS,
        )

        s27_mappo = MAPPO(state_dim=T2_STATE_DIM, joint_dim=T2_JOINT_DIM, action_dim=T2_ACTION_DIM)
        s27_mappo.load(S27_DIR)
        features_monday = bundle["features"]

        # Monday
        df_mon_27 = load_day(CSV_MONDAY, features_monday)
        env_s27_mon = TieredEnvV2(df=df_mon_27, bundle=bundle, n_agents=N_AGENTS)
        print(f"\nEvaluating S27 on Monday (seen attacks)...")
        print(df_mon_27["label"].value_counts().to_string())
        s27_monday_metrics = run_episode(env_s27_mon, s27_mappo, action_labels=T2_ACTION_LABELS)
        print(f"  TPR: {s27_monday_metrics['tpr']:.4f}  FPR: {s27_monday_metrics['fpr']:.4f}")
        print(f"  Action distribution per agent:")
        for i, dist in enumerate(s27_monday_metrics["action_distribution"]):
            print(f"    Router {i+1}: {dist}")
        with open(out_dir / "s27_monday_results.json", "w") as f:
            json.dump(s27_monday_metrics, f, indent=2)

        # Tuesday
        df_tue_27 = load_day(CSV_TUESDAY, features_monday)
        env_s27_tue = TieredEnvV2(df=df_tue_27, bundle=bundle, n_agents=N_AGENTS)
        print(f"\nEvaluating S27 on Tuesday (unseen attacks)...")
        print(df_tue_27["label"].value_counts().to_string())
        s27_tuesday_metrics = run_episode(env_s27_tue, s27_mappo, action_labels=T2_ACTION_LABELS)
        print(f"  TPR: {s27_tuesday_metrics['tpr']:.4f}  FPR: {s27_tuesday_metrics['fpr']:.4f}")
        print(f"  Action distribution per agent:")
        for i, dist in enumerate(s27_tuesday_metrics["action_distribution"]):
            print(f"    Router {i+1}: {dist}")
        with open(out_dir / "s27_tuesday_results.json", "w") as f:
            json.dump(s27_tuesday_metrics, f, indent=2)

        # Thursday
        if CSV_THURSDAY.exists():
            df_thu = load_day(CSV_THURSDAY, features_monday)
            env_s27_thu = TieredEnvV2(df=df_thu, bundle=bundle, n_agents=N_AGENTS)
            print(f"\nEvaluating S27 on Thursday (unseen)...")
            print(df_thu["label"].value_counts().to_string())
            s27_thu = run_episode(env_s27_thu, s27_mappo, action_labels=T2_ACTION_LABELS)
            print(f"  TPR: {s27_thu['tpr']:.4f}  FPR: {s27_thu['fpr']:.4f}")
            for i, dist in enumerate(s27_thu["action_distribution"]):
                print(f"    Router {i+1}: {dist}")
            with open(out_dir / "s27_thursday_results.json", "w") as f:
                json.dump(s27_thu, f, indent=2)

        # Saturday
        if CSV_SATURDAY.exists():
            df_sat = load_day(CSV_SATURDAY, features_monday)
            env_s27_sat = TieredEnvV2(df=df_sat, bundle=bundle, n_agents=N_AGENTS)
            print(f"\nEvaluating S27 on Saturday (unseen)...")
            print(df_sat["label"].value_counts().to_string())
            s27_sat = run_episode(env_s27_sat, s27_mappo, action_labels=T2_ACTION_LABELS)
            print(f"  TPR: {s27_sat['tpr']:.4f}  FPR: {s27_sat['fpr']:.4f}")
            for i, dist in enumerate(s27_sat["action_distribution"]):
                print(f"    Router {i+1}: {dist}")
            with open(out_dir / "s27_saturday_results.json", "w") as f:
                json.dump(s27_sat, f, indent=2)
    else:
        print(f"\nNo S27 model found at {S27_DIR} — skipping.")
        print("Run: python main.py --stage s27")

    # ── S28 evaluation (p_risk tiered profiles + Monday+Thursday mixed training)
    # Key test: does Saturday FPR drop vs S27?  Tuesday TPR must stay high.
    # Thursday is EXCLUDED — it was seen during training.
    s28_monday_metrics  = {}
    s28_tuesday_metrics = {}
    if _should_run("s28") and S28_DIR.exists():
        print(f"\nLoading trained S28 from {S28_DIR}...")
        print("  (p_risk routing + mixed Monday+Thursday training — benign-day FPR fix)")

        from src.environment_tiered_v2 import (
            MultiAgentEnv as TieredEnvV2,
            STATE_DIM as T2_STATE_DIM,
            JOINT_DIM as T2_JOINT_DIM,
            ACTION_DIM as T2_ACTION_DIM,
            ACTION_LABELS as T2_ACTION_LABELS,
        )

        s28_mappo = MAPPO(state_dim=T2_STATE_DIM, joint_dim=T2_JOINT_DIM, action_dim=T2_ACTION_DIM)
        s28_mappo.load(S28_DIR)
        features_s28 = bundle["features"]

        # Monday — sanity check (seen attacks, seen benign)
        df_mon_28 = load_day(CSV_MONDAY, features_s28)
        env_s28_mon = TieredEnvV2(df=df_mon_28, bundle=bundle, n_agents=N_AGENTS)
        print(f"\nEvaluating S28 on Monday (sanity — seen attacks)...")
        print(df_mon_28["label"].value_counts().to_string())
        s28_monday_metrics = run_episode(env_s28_mon, s28_mappo, action_labels=T2_ACTION_LABELS)
        print(f"  TPR: {s28_monday_metrics['tpr']:.4f}  FPR: {s28_monday_metrics['fpr']:.4f}")
        for i, dist in enumerate(s28_monday_metrics["action_distribution"]):
            print(f"    Router {i+1}: {dist}")
        with open(out_dir / "s28_monday_results.json", "w") as f:
            json.dump(s28_monday_metrics, f, indent=2)

        # Tuesday — primary test (unseen attack types)
        df_tue_28 = load_day(CSV_TUESDAY, features_s28)
        env_s28_tue = TieredEnvV2(df=df_tue_28, bundle=bundle, n_agents=N_AGENTS)
        print(f"\nEvaluating S28 on Tuesday (unseen attacks)...")
        print(df_tue_28["label"].value_counts().to_string())
        s28_tuesday_metrics = run_episode(env_s28_tue, s28_mappo, action_labels=T2_ACTION_LABELS)
        print(f"  TPR: {s28_tuesday_metrics['tpr']:.4f}  FPR: {s28_tuesday_metrics['fpr']:.4f}")
        for i, dist in enumerate(s28_tuesday_metrics["action_distribution"]):
            print(f"    Router {i+1}: {dist}")
        with open(out_dir / "s28_tuesday_results.json", "w") as f:
            json.dump(s28_tuesday_metrics, f, indent=2)

        # Saturday — KEY test (unseen benign-only day — did stand-down generalise?)
        # Thursday is intentionally skipped — it was in the training set.
        if CSV_SATURDAY.exists():
            df_sat_28 = load_day(CSV_SATURDAY, features_s28)
            env_s28_sat = TieredEnvV2(df=df_sat_28, bundle=bundle, n_agents=N_AGENTS)
            print(f"\nEvaluating S28 on Saturday (KEY: unseen benign-only day)...")
            print(df_sat_28["label"].value_counts().to_string())
            s28_sat_metrics = run_episode(env_s28_sat, s28_mappo, action_labels=T2_ACTION_LABELS)
            print(f"  TPR: {s28_sat_metrics['tpr']:.4f}  FPR: {s28_sat_metrics['fpr']:.4f}")
            print(f"  (S27 Saturday FPR was 0.4881 — did it drop?)")
            for i, dist in enumerate(s28_sat_metrics["action_distribution"]):
                print(f"    Router {i+1}: {dist}")
            with open(out_dir / "s28_saturday_results.json", "w") as f:
                json.dump(s28_sat_metrics, f, indent=2)
    else:
        print(f"\nNo S28 model found at {S28_DIR} — skipping.")
        print("Run: python main.py --stage s28")

    # ── S29 evaluation (IPPO + p_risk tiered profiles — no cooperation)
    # Cooperation isolation: S27 (MAPPO, joint obs) vs S29 (IPPO, local obs only)
    # run_episode() works directly — IPPO has same .actors and .n_agents as MAPPO
    s29_monday_metrics  = {}
    s29_tuesday_metrics = {}
    if _should_run("s29") and S29_DIR.exists():
        print(f"\nLoading trained S29 from {S29_DIR}...")
        print("  (IPPO — no joint observation, no centralised critic)")
        print("  Comparing vs S27 (MAPPO) to isolate cooperation contribution")

        from src.environment_tiered_v2 import (
            MultiAgentEnv as TieredEnvV2,
            STATE_DIM as T2_STATE_DIM,
            ACTION_DIM as T2_ACTION_DIM,
            ACTION_LABELS as T2_ACTION_LABELS,
        )
        s29_ippo = IPPO(state_dim=T2_STATE_DIM, action_dim=T2_ACTION_DIM, n_agents=N_AGENTS)
        s29_ippo.load(S29_DIR)
        features_s29 = bundle["features"]

        # Monday
        df_mon_29 = load_day(CSV_MONDAY, features_s29)
        env_s29_mon = TieredEnvV2(df=df_mon_29, bundle=bundle, n_agents=N_AGENTS)
        print(f"\nEvaluating S29 on Monday (seen attacks)...")
        print(df_mon_29["label"].value_counts().to_string())
        s29_monday_metrics = run_episode(env_s29_mon, s29_ippo, action_labels=T2_ACTION_LABELS)
        print(f"  TPR: {s29_monday_metrics['tpr']:.4f}  FPR: {s29_monday_metrics['fpr']:.4f}")
        print(f"  (S27 MAPPO Monday: TPR=0.9956, FPR=0.0276)")
        for i, dist in enumerate(s29_monday_metrics["action_distribution"]):
            print(f"    Router {i+1}: {dist}")
        with open(out_dir / "s29_monday_results.json", "w") as f:
            json.dump(s29_monday_metrics, f, indent=2)

        # Tuesday
        df_tue_29 = load_day(CSV_TUESDAY, features_s29)
        env_s29_tue = TieredEnvV2(df=df_tue_29, bundle=bundle, n_agents=N_AGENTS)
        print(f"\nEvaluating S29 on Tuesday (unseen attacks)...")
        print(df_tue_29["label"].value_counts().to_string())
        s29_tuesday_metrics = run_episode(env_s29_tue, s29_ippo, action_labels=T2_ACTION_LABELS)
        print(f"  TPR: {s29_tuesday_metrics['tpr']:.4f}  FPR: {s29_tuesday_metrics['fpr']:.4f}")
        print(f"  (S27 MAPPO Tuesday: TPR=0.8716, FPR=0.0668)")
        print(f"  → If S29 ≈ S27: cooperation not needed. If S29 < S27: CTDE contributing.")
        for i, dist in enumerate(s29_tuesday_metrics["action_distribution"]):
            print(f"    Router {i+1}: {dist}")
        with open(out_dir / "s29_tuesday_results.json", "w") as f:
            json.dump(s29_tuesday_metrics, f, indent=2)
    else:
        print(f"\nNo S29 model found at {S29_DIR} — skipping.")
        print("Run: python main.py --stage s29")

    # ── Full comparison table (Monday)
    print("\n" + "="*150)
    print("FULL COMPARISON — MONDAY (seen attacks)")
    print("="*150)
    all_models_mon = [
        ("MAPPO",  monday_metrics),
        ("IQL",    iql_monday_metrics),
        ("IPPO",   ippo_monday_metrics),
        ("MAPL",   mapl_monday_metrics),
        ("E13",    e13_monday_metrics),
        ("S11",    s11_monday_metrics),
        ("S12",    s12_monday_metrics),
        ("S13",    s13_monday_metrics),
        ("S14",    s14_monday_metrics),
        ("S15",    s15_monday_metrics),
        ("S16",    s16_monday_metrics),
        ("S17",    s17_monday_metrics),
        ("S18",    s18_monday_metrics),
        ("S19",    s19_monday_metrics),
        ("S21",    s21_monday_metrics),
        ("S22",    s22_monday_metrics),
        ("S23",    s23_monday_metrics),
        ("S24",    s24_monday_metrics),
        ("S25",    s25_monday_metrics),
        ("S26",    s26_monday_metrics),
        ("S26b",   s26b_monday_metrics),
        ("S27",    s27_monday_metrics),
        ("S28",    s28_monday_metrics),
        ("S29",    s29_monday_metrics),
    ]
    hdr = f"{'Metric':<16}" + "".join(f"{n:>10}" for n, _ in all_models_mon)
    print(hdr)
    print("-" * len(hdr))
    for key in ["total_reward", "tpr", "fpr", "legit_ratio"]:
        vals = [str(m.get(key, "-")) for _, m in all_models_mon]
        print(f"{key:<16}" + "".join(f"{v:>10}" for v in vals))

    # ── Full comparison table (Tuesday)
    print("\n" + "="*150)
    print("FULL COMPARISON — TUESDAY (unseen attacks)")
    print("="*150)
    all_models_tue = [
        ("MAPPO",  tuesday_metrics),
        ("IQL",    iql_tuesday_metrics),
        ("IPPO",   ippo_tuesday_metrics),
        ("MAPL",   mapl_tuesday_metrics),
        ("E13",    e13_tuesday_metrics),
        ("S11",    s11_tuesday_metrics),
        ("S12",    s12_tuesday_metrics),
        ("S13",    s13_tuesday_metrics),
        ("S14",    s14_tuesday_metrics),
        ("S15",    s15_tuesday_metrics),
        ("S16",    s16_tuesday_metrics),
        ("S17",    s17_tuesday_metrics),
        ("S18",    s18_tuesday_metrics),
        ("S19",    s19_tuesday_metrics),
        ("S21",    s21_tuesday_metrics),
        ("S22",    s22_tuesday_metrics),
        ("S23",    s23_tuesday_metrics),
        ("S24",    s24_tuesday_metrics),
        ("S25",    s25_tuesday_metrics),
        ("S26",    s26_tuesday_metrics),
        ("S26b",   s26b_tuesday_metrics),
        ("S27",    s27_tuesday_metrics),
        ("S28",    s28_tuesday_metrics),
        ("S29",    s29_tuesday_metrics),
    ]
    hdr = f"{'Metric':<16}" + "".join(f"{n:>10}" for n, _ in all_models_tue)
    print(hdr)
    print("-" * len(hdr))
    for key in ["total_reward", "tpr", "fpr", "legit_ratio"]:
        vals = [str(m.get(key, "-")) for _, m in all_models_tue]
        print(f"{key:<16}" + "".join(f"{v:>10}" for v in vals))

    # ── Ablation study
    df_monday = load_day(CSV_MONDAY, features)
    ablation  = ablation_study(df_monday, bundle)
    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(ablation, f, indent=2)

    print(f"\nAll results saved to: {out_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="CoopIDS evaluation. Runs all ablations by default.")
    p.add_argument("--tag", type=str, default="baseline",
                   help="Results saved to artifacts/evaluation/v1/<tag>/")
    p.add_argument("--model_tag", type=str, default=None,
                   help="Model tag for MAPPO checkpoint directory.")
    p.add_argument("--models", type=str, default=None,
                   help="Comma-separated models to evaluate "
                        "(e.g. s23,s24,s27,s29). If omitted, all are evaluated. "
                        "Valid names: mappo, iql, ippo, mapl, e13, s11, s12, s13, s14, "
                        "s15, s16, s17, s18, s19, s21, s22, s23, s24, s25, s26, s26b, "
                        "s27, s28, s29")
    args = p.parse_args()
    models_set = (
        {m.strip().lower() for m in args.models.split(",")}
        if args.models else None
    )
    main(tag=args.tag, model_tag=args.model_tag, models=models_set)
