"""
inspect_lgbm_probs.py
Inspect LightGBM predicted probability distributions per true label class.

Answers:
  - How well-separated are p_attack / p_benign / p_suspicious per true class?
  - Is 1 - p_attack a viable pass-rate function for puzzle?
  - How do Monday vs Tuesday distributions differ (LightGBM confidence drop)?

Usage:
    python inspect_lgbm_probs.py
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


# ── helpers ──────────────────────────────────────────────────────────────────

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
            X["handshake_state"]
            .astype(str)
            .map(hs_mapping)
            .fillna(-1)
            .astype(np.float32)
        )
    for col in X.columns:
        if col != "handshake_state":
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).astype("float32")
    return imputer.transform(X)


def load_csv(path: Path, features: list) -> pd.DataFrame:
    needed = list(dict.fromkeys(features + ["label"]))
    df = pd.read_csv(path, usecols=needed, low_memory=False)
    # Normalise label to lowercase for consistency
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df


def get_probs(df: pd.DataFrame, bundle: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return p_attack, p_benign, p_susp arrays aligned with df rows."""
    X     = prepare_features(df, bundle)
    model = bundle["model_3class"]
    cls   = bundle["classes_3class"]
    proba = model.predict_proba(X)
    p_attack = proba[:, cls.index("attack")]
    p_benign = proba[:, cls.index("benign")]
    p_susp   = proba[:, cls.index("suspicious")]
    return p_attack, p_benign, p_susp


def pct_stats(arr: np.ndarray) -> dict:
    return {
        "mean":  float(arr.mean()),
        "std":   float(arr.std()),
        "p5":    float(np.percentile(arr, 5)),
        "p25":   float(np.percentile(arr, 25)),
        "p50":   float(np.percentile(arr, 50)),
        "p75":   float(np.percentile(arr, 75)),
        "p95":   float(np.percentile(arr, 95)),
    }


def print_stats(label_name: str, p_attack: np.ndarray, p_benign: np.ndarray, p_susp: np.ndarray):
    n = len(p_attack)
    print(f"\n  True class: {label_name.upper()}  (n={n:,})")
    print(f"  {'metric':<8}  {'p_attack':>9}  {'p_benign':>9}  {'p_susp':>9}  {'1-p_atk':>9}")
    print(f"  {'-'*52}")
    for key in ("mean", "std", "p5", "p25", "p50", "p75", "p95"):
        a = pct_stats(p_attack)[key]
        b = pct_stats(p_benign)[key]
        s = pct_stats(p_susp)[key]
        r = 1.0 - a
        print(f"  {key:<8}  {a:>9.4f}  {b:>9.4f}  {s:>9.4f}  {r:>9.4f}")


def analyse_dataset(day_name: str, csv_path: Path, bundle: dict, bundle_name: str):
    features = bundle["features"]
    print(f"\n{'='*70}")
    print(f"  {day_name}  |  Bundle: {bundle_name}")
    print(f"{'='*70}")

    df = load_csv(csv_path, features)
    label_counts = df["label"].value_counts()
    print(f"\n  Label distribution:  {dict(label_counts)}")

    p_attack, p_benign, p_susp = get_probs(df, bundle)

    for true_class in ("benign", "suspicious", "attack"):
        mask = df["label"] == true_class
        if mask.sum() == 0:
            print(f"\n  (no {true_class} flows found)")
            continue
        print_stats(true_class, p_attack[mask], p_benign[mask], p_susp[mask])

    # Extra: puzzle pass-rate analysis — if we use 1-p_attack as pass rate
    print(f"\n  --- Puzzle pass-rate analysis (pass_rate = 1 - p_attack) ---")
    for true_class in ("benign", "suspicious", "attack"):
        mask = df["label"] == true_class
        if mask.sum() == 0:
            continue
        pass_rates = 1.0 - p_attack[mask]
        mean_pass  = pass_rates.mean()
        pct_blocked = (pass_rates < 0.5).mean() * 100  # flows that would be mostly blocked
        print(f"  {true_class:<12}  mean_pass_rate={mean_pass:.4f}  "
              f"frac_below_0.5={pct_blocked:.1f}%")

    # Extra: probability spacing — argmax confidence
    print(f"\n  --- Argmax confidence (max_prob - 2nd_max_prob) per true class ---")
    proba_stack = np.stack([p_attack, p_benign, p_susp], axis=1)
    sorted_p    = np.sort(proba_stack, axis=1)[:, ::-1]  # descending
    margin      = sorted_p[:, 0] - sorted_p[:, 1]         # top - 2nd

    for true_class in ("benign", "suspicious", "attack"):
        mask = df["label"] == true_class
        if mask.sum() == 0:
            continue
        m = margin[mask]
        low_conf = (m < 0.10).mean() * 100   # very close probs
        print(f"  {true_class:<12}  mean_margin={m.mean():.4f}  "
              f"std={m.std():.4f}  frac_margin<0.10={low_conf:.1f}%")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading bundles...")
    bundle_monday   = load_bundle(LGBM_BUNDLE)
    bundle_combined = load_bundle(LGBM_BUNDLE_COMBINED)

    # Monday data — Monday bundle (in-distribution)
    analyse_dataset("MONDAY",  CSV_MONDAY,  bundle_monday,   "monday-only")
    # Tuesday data — Monday bundle (out-of-distribution: what MARL sees during eval)
    analyse_dataset("TUESDAY", CSV_TUESDAY, bundle_monday,   "monday-only")
    # Tuesday data — Combined bundle (better calibration)
    analyse_dataset("TUESDAY", CSV_TUESDAY, bundle_combined, "combined (Mon+Tue)")


if __name__ == "__main__":
    main()
