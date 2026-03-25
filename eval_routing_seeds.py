"""
eval_routing_seeds.py
Test whether S30 agents are state-dependent or bucket-memorizing.

Runs S30 on Tuesday with 5 different routing seeds (0-4).
Each seed randomly re-assigns src_IPs to routers, so each router
sees a different traffic mix than in the default hash-based routing.

If agents are bucket-memorizing:
  → dominant action per router changes dramatically across seeds
  → TPR/FPR degrades significantly when a router gets "wrong" traffic

If agents are genuinely state-dependent:
  → dominant action shifts with the new traffic distribution
  → TPR/FPR stays stable because agents adapt to what they see

Usage:
    python eval_routing_seeds.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CSV_TUESDAY, LGBM_BUNDLE_COMBINED,
    S30_DIR, N_AGENTS, WINDOW_SIZE,
)
from src.environment_tiered_v2 import (
    MultiAgentEnv, STATE_DIM, JOINT_DIM, ACTION_DIM, ACTION_LABELS,
)
from src.mappo import MAPPO

T2_ACTION_LABELS = ACTION_LABELS


def load_bundle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_tuesday(csv_path: Path, features: list) -> pd.DataFrame:
    needed = list(dict.fromkeys(
        features + ["label", "src_ip", "dst_ip", "timestamp", "bytes_rate"]
    ))
    all_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    needed = [c for c in needed if c in all_cols]
    df = pd.read_csv(csv_path, usecols=needed, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def run_episode(env: MultiAgentEnv, mappo: MAPPO) -> dict:
    import torch
    from collections import defaultdict

    local_states, joint_obs = env.reset()
    done = False
    total_benign_in = total_benign_out = total_attack_in = total_attack_blocked = 0
    action_counts = [defaultdict(int) for _ in range(env.n_agents)]

    while not done:
        # Greedy — same as evaluate.py
        actions = []
        for i, actor in enumerate(mappo.actors):
            obs_t  = torch.FloatTensor(local_states[i]).unsqueeze(0)
            dist   = actor.get_dist(obs_t)
            action = int(dist.probs.argmax().item())
            actions.append(action)
            action_counts[i][action] += 1

        next_local, next_joint, rewards, done = env.step(actions)

        info = env.last_step_info
        total_benign_in      += info["benign_total"]
        total_attack_in      += info["attack_total"]
        total_benign_out     += info["benign_forwarded"]
        total_attack_blocked += info["attack_blocked"]

        local_states = next_local
        joint_obs    = next_joint

    tpr = total_attack_blocked / max(total_attack_in, 1)
    fpr = 1.0 - total_benign_out / max(total_benign_in, 1)

    action_dist = [
        {T2_ACTION_LABELS.get(k, str(k)): v for k, v in sorted(ac.items())}
        for ac in action_counts
    ]
    return {"tpr": round(tpr, 4), "fpr": round(fpr, 4), "action_distribution": action_dist}


def main() -> None:
    print("=" * 70)
    print("  S30 ROUTING SEED EXPERIMENT")
    print("  Question: are agents state-dependent or bucket-memorizing?")
    print("=" * 70)

    bundle = load_bundle(LGBM_BUNDLE_COMBINED)
    features = bundle["features"]
    df_tue = load_tuesday(CSV_TUESDAY, features)

    print(f"\nTuesday flows: {len(df_tue):,}")
    print(df_tue["label"].value_counts().to_string())

    mappo = MAPPO(state_dim=STATE_DIM, joint_dim=JOINT_DIM, action_dim=ACTION_DIM)
    mappo.load(S30_DIR)
    print(f"\nLoaded S30 model from {S30_DIR}")

    seeds = [None, 0, 1, 2, 3]  # None = default hash routing
    seed_labels = ["default (hash)", "seed=0", "seed=1", "seed=2", "seed=3"]

    results = []
    for seed, label in zip(seeds, seed_labels):
        env = MultiAgentEnv(
            df=df_tue, bundle=bundle,
            n_agents=N_AGENTS, window_size=WINDOW_SIZE,
            routing_seed=seed,
        )

        # Show how traffic is split across routers
        agent_counts = df_tue["src_ip"].apply(
            lambda ip: hash(str(ip)) % N_AGENTS if seed is None
            else int(np.random.RandomState(seed).randint(0, N_AGENTS))
        ).value_counts().sort_index()

        r = run_episode(env, mappo)
        results.append((label, r))

        print(f"\n  [{label}]  TPR={r['tpr']:.4f}  FPR={r['fpr']:.4f}")
        for i, dist in enumerate(r["action_distribution"]):
            total = sum(dist.values())
            dominant = max(dist, key=dist.get)
            dominant_pct = 100 * dist[dominant] / max(total, 1)
            print(f"    Router {i+1}: dominant={dominant} ({dominant_pct:.0f}%)  full={dist}")

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY — dominant action per router across routing seeds")
    print("=" * 70)
    print(f"\n  {'Routing':<20} {'TPR':>6} {'FPR':>6}  R1-dominant  R2-dominant  R3-dominant")
    print(f"  {'─'*70}")
    for label, r in results:
        dists = r["action_distribution"]
        doms = []
        for dist in dists:
            total = sum(dist.values())
            dom = max(dist, key=dist.get)
            pct = 100 * dist[dom] / max(total, 1)
            doms.append(f"{dom}({pct:.0f}%)")
        print(f"  {label:<20} {r['tpr']:>6.4f} {r['fpr']:>6.4f}  "
              + "  ".join(f"{d:<13}" for d in doms))

    print("\n  Interpretation:")
    print("  - If dominant action changes per router across seeds → state-dependent ✓")
    print("  - If dominant action is fixed regardless of seed    → bucket-memorizing ✗")
    print("  - If TPR/FPR degrades significantly with new seeds  → bucket-memorizing ✗")


if __name__ == "__main__":
    main()
