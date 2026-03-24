"""
analyse_action_space.py
Analyse p_attack distributions per true label to design tiered action profiles.

Outputs:
  1. Detailed percentiles of p_attack per label
  2. Histogram: what % of each label falls in each p_attack bucket
  3. For candidate threshold pairs (T_block, T_puzzle): impact on each label

Usage:
    python analyse_action_space.py
"""

from __future__ import annotations

import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

from config import (
    CSV_MONDAY, CSV_TUESDAY,
    LGBM_BUNDLE, LGBM_BUNDLE_COMBINED,
)


def load_bundle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_features(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    features   = bundle["features"]
    imputer    = bundle["imputer"]
    hs_mapping = bundle["handshake_mapping"]
    X = df[features].copy()
    if "handshake_state" in X.columns and hs_mapping:
        X["handshake_state"] = (
            X["handshake_state"].astype(str).map(hs_mapping).fillna(-1).astype(np.float32)
        )
    for col in X.columns:
        if col != "handshake_state":
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).astype("float32")
    return imputer.transform(X)


def get_p_attack(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    X     = prepare_features(df, bundle)
    model = bundle["model_3class"]
    cls   = bundle["classes_3class"]
    proba = model.predict_proba(X)
    return proba[:, cls.index("attack")]


def analyse(day_name: str, csv_path: Path, bundle: dict, bundle_name: str):
    features = bundle["features"]
    needed   = list(dict.fromkeys(features + ["label"]))
    df       = pd.read_csv(csv_path, usecols=needed, low_memory=False)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    p_attack = get_p_attack(df, bundle)

    print(f"\n{'#'*80}")
    print(f"  {day_name}  |  Bundle: {bundle_name}")
    print(f"  Total flows: {len(df):,}")
    print(f"{'#'*80}")

    # ── 1. Percentiles per label
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  ── p_attack PERCENTILES per true label ──")
    print(f"  {'label':<12} {'count':>8}  " + "  ".join(f"{'p'+str(p):>6}" for p in pcts))
    print(f"  {'-'*100}")
    for label in ("benign", "suspicious", "attack"):
        mask = df["label"] == label
        n    = mask.sum()
        if n == 0:
            continue
        vals = p_attack[mask]
        row  = "  ".join(f"{np.percentile(vals, p):>6.3f}" for p in pcts)
        print(f"  {label:<12} {n:>8,}  {row}")

    # ── 2. Histogram buckets
    edges = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 0.95, 1.01]
    labels_str = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(edges)-1)]

    print(f"\n  ── p_attack HISTOGRAM (% of each label in each bucket) ──")
    print(f"  {'label':<12} " + " ".join(f"{b:>11}" for b in labels_str))
    print(f"  {'-'*115}")
    for label in ("benign", "suspicious", "attack"):
        mask = df["label"] == label
        n    = mask.sum()
        if n == 0:
            continue
        vals = p_attack[mask]
        counts, _ = np.histogram(vals, bins=edges)
        pcts_arr  = counts / n * 100
        row = " ".join(f"{p:>10.1f}%" for p in pcts_arr)
        print(f"  {label:<12} {row}")

    # ── 3. Candidate threshold profiles — impact analysis
    candidates = [
        ("passive",    1.00, 1.00),
        ("light",      0.85, 0.35),
        ("moderate",   0.60, 0.20),
        ("firm",       0.35, 0.10),
        ("aggressive", 0.20, 0.05),
        ("lockdown",   0.10, 0.02),
    ]

    print(f"\n  ── THRESHOLD PROFILE IMPACT (% of label blocked / puzzled / forwarded) ──")
    print(f"  {'profile':<12} {'T_blk':>5} {'T_pzl':>5}  |"
          f"  {'blk_atk':>8} {'pzl_atk':>8} {'fwd_atk':>8}  |"
          f"  {'blk_ben':>8} {'pzl_ben':>8} {'fwd_ben':>8}  |"
          f"  {'blk_sus':>8} {'pzl_sus':>8} {'fwd_sus':>8}")
    print(f"  {'-'*130}")

    for name, t_block, t_puzzle in candidates:
        row_parts = [f"  {name:<12} {t_block:>5.2f} {t_puzzle:>5.2f}  |"]
        for label in ("attack", "benign", "suspicious"):
            mask = df["label"] == label
            n    = mask.sum()
            if n == 0:
                row_parts.append(f"  {'n/a':>8} {'n/a':>8} {'n/a':>8}  |")
                continue
            vals = p_attack[mask]
            pct_blocked  = (vals >= t_block).mean() * 100
            pct_puzzled  = ((vals >= t_puzzle) & (vals < t_block)).mean() * 100
            pct_forward  = (vals < t_puzzle).mean() * 100
            row_parts.append(f"  {pct_blocked:>7.1f}% {pct_puzzled:>7.1f}% {pct_forward:>7.1f}%  |")
        print("".join(row_parts))

    # ── 4. Effective TPR and FPR for each profile
    # TPR = % of attack flows blocked + puzzled-and-caught (puzzle catch rate = 95%)
    # FPR = % of benign flows blocked + puzzled-and-caught (puzzle false-catch = 5%)
    print(f"\n  ── EFFECTIVE TPR / FPR per profile ──")
    print(f"  (puzzle catch: attack=95%, benign=5%)")
    print(f"  {'profile':<12} {'T_blk':>5} {'T_pzl':>5}  |  {'TPR':>7}  {'FPR':>7}  {'atk_leak':>9}")
    print(f"  {'-'*60}")

    for name, t_block, t_puzzle in candidates:
        # Attack
        mask_atk = df["label"] == "attack"
        n_atk = mask_atk.sum()
        if n_atk == 0:
            continue
        v_atk = p_attack[mask_atk]
        atk_blocked = (v_atk >= t_block).mean()
        atk_puzzled = ((v_atk >= t_puzzle) & (v_atk < t_block)).mean()
        atk_forward = (v_atk < t_puzzle).mean()
        # puzzled attacks: 95% caught by CAPTCHA
        tpr = atk_blocked + atk_puzzled * 0.95

        # Benign
        mask_ben = df["label"] == "benign"
        n_ben = mask_ben.sum()
        v_ben = p_attack[mask_ben]
        ben_blocked = (v_ben >= t_block).mean()
        ben_puzzled = ((v_ben >= t_puzzle) & (v_ben < t_block)).mean()
        # puzzled benign: 5% incorrectly caught
        fpr = ben_blocked + ben_puzzled * 0.05

        atk_leak = 1.0 - tpr  # % of attacks that get through

        print(f"  {name:<12} {t_block:>5.2f} {t_puzzle:>5.2f}  |  {tpr:>6.1f}%  {fpr:>6.1f}%  {atk_leak:>8.1f}%")


def main():
    print("Loading bundles...")
    bundle_monday   = load_bundle(LGBM_BUNDLE)
    bundle_combined = load_bundle(LGBM_BUNDLE_COMBINED)

    analyse("MONDAY",  CSV_MONDAY,  bundle_monday,   "monday-only")
    analyse("TUESDAY", CSV_TUESDAY, bundle_monday,   "monday-only")
    analyse("TUESDAY", CSV_TUESDAY, bundle_combined, "combined (Mon+Tue)")


if __name__ == "__main__":
    main()
