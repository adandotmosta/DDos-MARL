"""
analyse_prob_distributions.py
Full analysis of p_attack, p_suspicious, p_benign distributions per true label,
on Monday and Tuesday, with both the Monday-only and combined LightGBM bundles.

Core question:
  When p_attack collapses on Tuesday (Monday-only scorer), where does the mass go?
  If attacks score high p_suspicious, then composite scores like (1 - p_benign)
  or max(p_attack, p_suspicious) may generalise far better than p_attack alone.

Sections:
  1. Raw percentiles of all three probs per label — both days, both bundles
  2. Predicted label distribution (argmax): how often does scorer call an attack "benign"?
  3. Composite scores: 1-p_benign, max(p_atk, p_sus), p_atk+p_sus
     → percentiles per true label, both days
  4. Generalization gap table: Monday vs Tuesday p50 for each score × label
  5. Threshold impact for EACH composite score:
     For every candidate threshold T, show effective TPR/FPR on Monday AND Tuesday
     under Monday-only scorer — so you can see which score degrades least
  6. Design recommendations: which composite score has the smallest Mon→Tue gap
     for attacks, while keeping benign separation

Usage:
    python analyse_prob_distributions.py

Requires: config.py with CSV_MONDAY, CSV_TUESDAY, LGBM_BUNDLE, LGBM_BUNDLE_COMBINED
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

from config import (
    CSV_MONDAY, CSV_TUESDAY,
    LGBM_BUNDLE, LGBM_BUNDLE_COMBINED,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_bundle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_probs(csv_path: Path, bundle: dict) -> pd.DataFrame:
    """
    Load CSV, run LightGBM 3-class scorer, return DataFrame with columns:
        label (true), p_attack, p_suspicious, p_benign
    """
    features = bundle["features"]
    needed   = list(dict.fromkeys(features + ["label"]))
    df       = pd.read_csv(csv_path, usecols=needed, low_memory=False)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    X = df[features].copy()
    hs_map = bundle.get("handshake_mapping", {})
    if "handshake_state" in X.columns and hs_map:
        X["handshake_state"] = (
            X["handshake_state"].astype(str).map(hs_map).fillna(-1).astype(np.float32)
        )
    for col in X.columns:
        if col != "handshake_state":
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).astype("float32")
    X = bundle["imputer"].transform(X)

    model  = bundle["model_3class"]
    classes = bundle["classes_3class"]   # e.g. ['attack', 'benign', 'suspicious']
    proba  = model.predict_proba(X)

    idx_atk = classes.index("attack")
    idx_ben = classes.index("benign")
    idx_sus = classes.index("suspicious")

    out = pd.DataFrame({
        "label":       df["label"].values,
        "p_attack":    proba[:, idx_atk].astype(np.float32),
        "p_benign":    proba[:, idx_ben].astype(np.float32),
        "p_suspicious": proba[:, idx_sus].astype(np.float32),
    })
    # composite scores
    out["p_risk"]        = out["p_attack"] + out["p_suspicious"]   # = 1 - p_benign
    out["p_max_threat"]  = np.maximum(out["p_attack"], out["p_suspicious"])
    out["pred_label"]    = [classes[i] for i in proba.argmax(axis=1)]
    return out


def sep(width=110):
    print("-" * width)


def header(title, width=110):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Raw percentiles for all three probs
# ─────────────────────────────────────────────────────────────────────────────

PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
TRUE_LABELS = ["benign", "suspicious", "attack"]

def section_raw_percentiles(df: pd.DataFrame, title: str):
    header(f"RAW PERCENTILES — {title}")
    for score in ("p_attack", "p_suspicious", "p_benign"):
        print(f"\n  [{score}]")
        print(f"  {'label':<12} {'N':>8}  " + "  ".join(f"p{p:02d}" for p in PCTS))
        sep(80)
        for lbl in TRUE_LABELS:
            mask = df["label"] == lbl
            n = mask.sum()
            if n == 0:
                continue
            vals = df.loc[mask, score].values
            row  = "  ".join(f"{np.percentile(vals, p):>5.3f}" for p in PCTS)
            print(f"  {lbl:<12} {n:>8,}  {row}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Predicted label distribution
# ─────────────────────────────────────────────────────────────────────────────

def section_predicted_labels(df: pd.DataFrame, title: str):
    header(f"PREDICTED LABEL DISTRIBUTION — {title}")
    print(f"  (Where does the scorer put each true class?)")
    print(f"\n  {'true_label':<14} {'N':>8}  {'→ benign':>10}  {'→ suspicious':>14}  {'→ attack':>10}")
    sep(70)
    for lbl in TRUE_LABELS:
        mask = df["label"] == lbl
        n = mask.sum()
        if n == 0:
            continue
        sub = df.loc[mask, "pred_label"]
        for pred in ("benign", "suspicious", "attack"):
            pct = (sub == pred).mean() * 100
        rows = {p: (sub == p).mean() * 100 for p in ("benign", "suspicious", "attack")}
        print(f"  {lbl:<14} {n:>8,}  {rows['benign']:>9.1f}%  {rows['suspicious']:>13.1f}%  {rows['attack']:>9.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Composite score percentiles
# ─────────────────────────────────────────────────────────────────────────────

def section_composite_percentiles(df: pd.DataFrame, title: str):
    header(f"COMPOSITE SCORE PERCENTILES — {title}")
    composites = [
        ("p_risk      (p_atk+p_sus = 1-p_ben)", "p_risk"),
        ("p_max_threat (max(p_atk, p_sus))",     "p_max_threat"),
        ("p_attack    (baseline, repeated)",     "p_attack"),
    ]
    for label_str, col in composites:
        print(f"\n  [{label_str}]")
        print(f"  {'label':<12} {'N':>8}  " + "  ".join(f"p{p:02d}" for p in PCTS))
        sep(80)
        for lbl in TRUE_LABELS:
            mask = df["label"] == lbl
            n = mask.sum()
            if n == 0:
                continue
            vals = df.loc[mask, col].values
            row  = "  ".join(f"{np.percentile(vals, p):>5.3f}" for p in PCTS)
            print(f"  {lbl:<12} {n:>8,}  {row}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Generalization gap table
# ─────────────────────────────────────────────────────────────────────────────

def section_generalization_gap(mon: pd.DataFrame, tue: pd.DataFrame,
                                bundle_name: str):
    header(f"GENERALIZATION GAP — {bundle_name}  (Monday → Tuesday p50 shift)")
    scores = [
        ("p_attack",     "p_attack"),
        ("p_suspicious", "p_suspicious"),
        ("p_benign",     "p_benign"),
        ("p_risk",       "1-p_benign"),
        ("p_max_threat", "max(atk,sus)"),
    ]
    print(f"\n  {'score':<16}  {'true_lbl':<12} {'Mon p50':>9} {'Tue p50':>9} {'gap (Tue-Mon)':>14}  note")
    sep(80)
    for col, col_name in scores:
        for lbl in TRUE_LABELS:
            mask_mon = mon["label"] == lbl
            mask_tue = tue["label"] == lbl
            if mask_mon.sum() == 0 or mask_tue.sum() == 0:
                continue
            p50_mon = np.percentile(mon.loc[mask_mon, col].values, 50)
            p50_tue = np.percentile(tue.loc[mask_tue, col].values, 50)
            gap     = p50_tue - p50_mon
            # Flag: if attack score drops a lot (gap << 0), scorer degrades
            # If benign score stays low and attack stays high → good generalization
            if lbl == "attack":
                note = "⚠ DEGRADES" if gap < -0.15 else ("OK" if gap > -0.05 else "mild drop")
            elif lbl == "benign":
                note = "⚠ FP RISK" if gap > 0.10 else "OK"
            else:
                note = ""
            print(f"  {col_name:<16}  {lbl:<12} {p50_mon:>9.3f} {p50_tue:>9.3f} {gap:>+14.3f}  {note}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Threshold impact per score — Monday AND Tuesday
# ─────────────────────────────────────────────────────────────────────────────

def _threshold_tpr_fpr(df: pd.DataFrame, score_col: str,
                        t_block: float, t_puzzle: float) -> tuple[float, float]:
    """
    Block flows with score >= t_block.
    Puzzle flows with score in [t_puzzle, t_block).
    Return (TPR, FPR) where:
      TPR = fraction of attack flows removed (blocked + puzzle-caught with 95% rate)
      FPR = fraction of benign flows wrongly removed (blocked + puzzle-caught with 5% rate)
    """
    mask_atk = df["label"] == "attack"
    mask_ben = df["label"] == "benign"
    if mask_atk.sum() == 0 or mask_ben.sum() == 0:
        return 0.0, 0.0

    v_atk = df.loc[mask_atk, score_col].values
    v_ben = df.loc[mask_ben, score_col].values

    atk_blocked = (v_atk >= t_block).mean()
    atk_puzzled = ((v_atk >= t_puzzle) & (v_atk < t_block)).mean()
    tpr = atk_blocked + atk_puzzled * 0.95

    ben_blocked = (v_ben >= t_block).mean()
    ben_puzzled = ((v_ben >= t_puzzle) & (v_ben < t_block)).mean()
    fpr = ben_blocked + ben_puzzled * 0.05

    return tpr, fpr


def section_threshold_impact(mon: pd.DataFrame, tue: pd.DataFrame,
                              bundle_name: str):
    header(f"THRESHOLD IMPACT (Mon vs Tue TPR/FPR) — {bundle_name}")
    print("  (puzzle CAPTCHA pass rates: attack=5%, benign=95%)")

    scores = [
        ("p_attack",     "p_attack (baseline)"),
        ("p_risk",       "p_risk = 1-p_benign"),
        ("p_max_threat", "p_max = max(atk,sus)"),
    ]

    # Candidate (T_block, T_puzzle) pairs — sweep a range useful for each score
    candidates = [
        (0.95, 0.70),
        (0.90, 0.50),
        (0.85, 0.35),
        (0.80, 0.30),
        (0.70, 0.25),
        (0.60, 0.20),
        (0.50, 0.15),
        (0.40, 0.10),
        (0.30, 0.08),
        (0.20, 0.05),
        (0.10, 0.02),
    ]

    for col, col_name in scores:
        print(f"\n  ── [{col_name}] ──")
        print(f"  {'T_block':>7} {'T_puzz':>7}  |"
              f"  {'Mon_TPR':>8} {'Mon_FPR':>8}  |"
              f"  {'Tue_TPR':>8} {'Tue_FPR':>8}  |"
              f"  {'TPR_gap':>8} {'FPR_gap':>8}")
        sep(90)
        for t_block, t_puzzle in candidates:
            tpr_mon, fpr_mon = _threshold_tpr_fpr(mon, col, t_block, t_puzzle)
            tpr_tue, fpr_tue = _threshold_tpr_fpr(tue, col, t_block, t_puzzle)
            tpr_gap = tpr_tue - tpr_mon
            fpr_gap = fpr_tue - fpr_mon
            flag = "  ← GOOD" if abs(tpr_gap) < 0.05 and tpr_mon > 0.80 else ""
            print(f"  {t_block:>7.2f} {t_puzzle:>7.2f}  |"
                  f"  {tpr_mon*100:>7.1f}% {fpr_mon*100:>7.1f}%  |"
                  f"  {tpr_tue*100:>7.1f}% {fpr_tue*100:>7.1f}%  |"
                  f"  {tpr_gap*100:>+7.1f}% {fpr_gap*100:>+7.1f}%{flag}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Score correlation for attacks — where p_attack collapses
# ─────────────────────────────────────────────────────────────────────────────

def section_attack_score_breakdown(mon: pd.DataFrame, tue: pd.DataFrame,
                                   bundle_name: str):
    header(f"ATTACK FLOW SCORE BREAKDOWN — {bundle_name}")
    print("  (For attack flows only: what does each score look like when p_attack is low?)")
    print("  Rows = p_attack bucket for attack flows; Cols = p50 of other scores in that bucket\n")

    for day_name, df in [("Monday", mon), ("Tuesday", tue)]:
        mask_atk = df["label"] == "attack"
        sub = df.loc[mask_atk].copy()
        if len(sub) == 0:
            continue
        print(f"  ── {day_name} (N_attack = {len(sub):,}) ──")
        print(f"  {'p_atk bucket':<20} {'N':>7}  "
              f"{'p_sus p50':>10} {'p_ben p50':>10} {'p_risk p50':>11} {'p_max p50':>10}")
        sep(80)

        edges = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.01]
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i+1]
            bucket = sub[(sub["p_attack"] >= lo) & (sub["p_attack"] < hi)]
            n = len(bucket)
            if n == 0:
                continue
            p_sus  = np.percentile(bucket["p_suspicious"].values, 50)
            p_ben  = np.percentile(bucket["p_benign"].values, 50)
            p_risk = np.percentile(bucket["p_risk"].values, 50)
            p_max  = np.percentile(bucket["p_max_threat"].values, 50)
            bname  = f"[{lo:.2f}, {hi:.2f})"
            print(f"  {bname:<20} {n:>7,}  "
                  f"{p_sus:>10.3f} {p_ben:>10.3f} {p_risk:>11.3f} {p_max:>10.3f}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Design recommendations
# ─────────────────────────────────────────────────────────────────────────────

def section_recommendations(mon: pd.DataFrame, tue: pd.DataFrame):
    header("DESIGN RECOMMENDATIONS")
    print("""
  Key question: can we replace pure p_attack routing with a composite score
  that is more stable when the scorer degrades on Tuesday?

  Scores to compare:
    A. p_attack            — baseline, known to collapse on Tuesday with Monday scorer
    B. p_risk = 1-p_benign  — captures all non-benign mass; may stay high even if
                              the model shifts probability from 'attack' → 'suspicious'
    C. p_max = max(p_atk, p_sus) — similar to p_risk but ignores their sum;
                              useful if one class dominates

  Reading the gap table (Section 4):
    - If p_risk for attacks drops less than p_attack → B is better routing signal
    - If p_max drops less → C is better
    - Look at benign rows: if p_risk / p_max for benign stays LOW on Tuesday → low FPR maintained

  Reading the threshold impact (Section 5):
    - Find rows where TPR_gap (Tue - Mon) is small (< 5%) AND Mon_TPR is high (> 85%)
    - The score × threshold pair with smallest TPR drop is the most generalizable design

  Implication for S26b / S27 action space:
    - If p_risk generalises better → use 1-p_benign as the routing signal in environment_tiered.py
      (replace p_attack threshold checks with (1 - p_benign) > threshold)
    - Thresholds must be recalibrated: p_risk is stochastically HIGHER than p_attack
      (it sums two probabilities), so Monday-optimised p_attack thresholds will
      be too aggressive under p_risk — re-read Section 5 to pick the right T_block, T_puzzle
    - If p_max generalises better and maintains benign separation → use it as the signal
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading bundles...")
    bundle_mon = load_bundle(LGBM_BUNDLE)
    bundle_com = load_bundle(LGBM_BUNDLE_COMBINED)

    print("Running scorer on Monday (Monday bundle)...")
    mon_mon = get_probs(CSV_MONDAY, bundle_mon)
    print(f"  Monday flows: {len(mon_mon):,}  | labels: {dict(mon_mon['label'].value_counts())}")

    print("Running scorer on Tuesday (Monday bundle)...")
    tue_mon = get_probs(CSV_TUESDAY, bundle_mon)
    print(f"  Tuesday flows: {len(tue_mon):,}  | labels: {dict(tue_mon['label'].value_counts())}")

    print("Running scorer on Tuesday (Combined bundle)...")
    tue_com = get_probs(CSV_TUESDAY, bundle_com)
    print(f"  Tuesday flows: {len(tue_com):,}  | labels: {dict(tue_com['label'].value_counts())}")

    # ── Sections ──────────────────────────────────────────────────────────────

    print("\n\n" + "=" * 110)
    print("  PART 1: MONDAY DATA — Monday-only scorer")
    print("=" * 110)
    section_raw_percentiles(mon_mon, "MONDAY / Monday-only scorer")
    section_composite_percentiles(mon_mon, "MONDAY / Monday-only scorer")
    section_predicted_labels(mon_mon, "MONDAY / Monday-only scorer")

    print("\n\n" + "=" * 110)
    print("  PART 2: TUESDAY DATA — Monday-only scorer  (the degradation case)")
    print("=" * 110)
    section_raw_percentiles(tue_mon, "TUESDAY / Monday-only scorer")
    section_composite_percentiles(tue_mon, "TUESDAY / Monday-only scorer")
    section_predicted_labels(tue_mon, "TUESDAY / Monday-only scorer")

    print("\n\n" + "=" * 110)
    print("  PART 3: GENERALIZATION GAP  (Monday-only scorer: Mon → Tue)")
    print("=" * 110)
    section_generalization_gap(mon_mon, tue_mon, "Monday-only scorer")

    print("\n\n" + "=" * 110)
    print("  PART 4: ATTACK SCORE BREAKDOWN — where does p_attack mass go on Tuesday?")
    print("=" * 110)
    section_attack_score_breakdown(mon_mon, tue_mon, "Monday-only scorer")

    print("\n\n" + "=" * 110)
    print("  PART 5: THRESHOLD IMPACT — Monday vs Tuesday (Monday-only scorer)")
    print("=" * 110)
    section_threshold_impact(mon_mon, tue_mon, "Monday-only scorer")

    print("\n\n" + "=" * 110)
    print("  PART 6: REFERENCE — TUESDAY with Combined scorer (upper bound)")
    print("=" * 110)
    section_raw_percentiles(tue_com, "TUESDAY / Combined scorer")
    section_composite_percentiles(tue_com, "TUESDAY / Combined scorer")
    section_generalization_gap(mon_mon, tue_com, "Combined scorer (Mon Mon-bundle vs Tue Combined-bundle)")

    print("\n\n" + "=" * 110)
    print("  PART 7: DESIGN RECOMMENDATIONS")
    print("=" * 110)
    section_recommendations(mon_mon, tue_mon)

    print("\nDone.")


if __name__ == "__main__":
    main()
