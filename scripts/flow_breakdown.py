"""
flow_breakdown.py
Computes per-flow routing breakdown for S27 tiered profiles on Monday and Tuesday.

For each tiered profile, classifies every flow as:
  - Forwarded:  p_risk < T_puzzle  (pass through directly)
  - Puzzle:     T_puzzle <= p_risk < T_block  (sent to challenge; stochastic outcome)
  - Blocked:    p_risk >= T_block  (deterministic block)

Puzzle oracle pass rates (from environment_tiered_v2.py):
  Benign: 95%, Suspicious: 50%, Attack: 5%
"""
import pickle
import numpy as np
import pandas as pd
import json

# ── Config ────────────────────────────────────────────────────────────────────
BASE = "/Users/abdenourbelarbi/Documents/MultiAgents/MARL_Cybersecurity/Research_project"
BUNDLE = f"{BASE}/Risk_Model/artifacts/vae/v1/models/lgbm_bundle_monday.pkl"
MON_CSV = f"{BASE}/BCC-2024/Monday_18_Dec_2023.csv"
TUE_CSV = f"{BASE}/BCC-2024/Tuesday_19_Dec_2023.csv"

TIERED_PROFILES = {
    0: (1.00, 1.00),   # passive     — all forwarded
    1: (0.90, 0.60),   # light
    2: (0.80, 0.40),   # moderate
    3: (0.70, 0.25),   # firm
    4: (0.60, 0.15),   # aggressive
    5: (0.50, 0.10),   # lockdown
}
PROFILE_NAMES = ["Passive", "Light", "Moderate", "Firm", "Aggressive", "Lockdown"]

# Puzzle oracle pass rates
PUZZLE_PASS = {"Benign": 0.95, "Suspicious": 0.50, "Attack": 0.05}


def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_risk_scores(df, bundle):
    """Return (p_risk, label_category) arrays for all flows."""
    features = bundle["features"]
    hs_map = bundle["handshake_mapping"]
    imputer = bundle["imputer"]
    model = bundle["model_3class"]
    classes = bundle["classes_3class"]

    X = df[features].copy()
    if "handshake_state" in X.columns and hs_map:
        X["handshake_state"] = (
            X["handshake_state"].astype(str).map(hs_map).fillna(-1).astype(np.float32)
        )
    for col in X.columns:
        if col != "handshake_state":
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).astype("float32")
    X_imp = imputer.transform(X)

    proba = model.predict_proba(X_imp)   # (N, 3)
    # classes: typically ['Attack', 'Benign', 'Suspicious']
    cls_list = list(classes)
    idx_b = cls_list.index("benign")
    p_risk = 1.0 - proba[:, idx_b]      # p_risk = 1 - p_benign

    # Label category (CSV labels are e.g. "Benign" or attack names)
    label_col = "label" if "label" in df.columns else "Label"
    label = df[label_col].astype(str).str.lower().values
    cat = np.where(label == "benign", "Benign",
           np.where(label == "suspicious", "Suspicious", "Attack"))
    return p_risk, cat


def breakdown(p_risk, cat, t_block, t_puzzle):
    """Return dict with forwarded/puzzle/blocked counts, and effective pass-through."""
    n = len(p_risk)
    fwd_mask  = p_risk < t_puzzle
    blk_mask  = p_risk >= t_block
    puzz_mask = ~fwd_mask & ~blk_mask

    result = {}
    for label in ["Benign", "Suspicious", "Attack", "Total"]:
        if label == "Total":
            m = np.ones(n, dtype=bool)
        else:
            m = cat == label
        total = m.sum()
        fwd   = (fwd_mask  & m).sum()
        puzz  = (puzz_mask & m).sum()
        blk   = (blk_mask  & m).sum()

        # Expected flows passing puzzle
        pass_rate = PUZZLE_PASS.get(label, 0.50)  # Total gets 0.5 as placeholder
        puzz_pass = int(puzz * pass_rate)
        effective_out = fwd + puzz_pass

        result[label] = {
            "total": int(total),
            "forwarded": int(fwd),
            "puzzle": int(puzz),
            "blocked": int(blk),
            "puzz_pass_est": int(puzz_pass),
            "effective_out": int(effective_out),
            "pct_forwarded": 100 * fwd / max(total, 1),
            "pct_puzzle": 100 * puzz / max(total, 1),
            "pct_blocked": 100 * blk / max(total, 1),
        }
    return result


def print_day(day_name, p_risk, cat):
    print(f"\n{'='*70}")
    print(f"  {day_name}")
    print(f"  Total flows: {len(p_risk):,}  |  "
          f"Benign: {(cat=='Benign').sum():,}  |  "
          f"Attack: {(cat=='Attack').sum():,}  |  "
          f"Suspicious: {(cat=='Suspicious').sum():,}")
    print(f"{'='*70}")

    # Focus on lockdown (S27 dominant action) and passive for contrast
    for pid, (t_block, t_puzzle) in TIERED_PROFILES.items():
        name = PROFILE_NAMES[pid]
        bd = breakdown(p_risk, cat, t_block, t_puzzle)

        tot   = bd["Total"]
        benign = bd["Benign"]
        attack = bd["Attack"]

        print(f"\n  Profile {pid}: {name}  (T_block={t_block}, T_puzzle={t_puzzle})")
        print(f"    {'Label':<12} {'Total':>8} {'Fwd%':>7} {'Puzz%':>7} {'Blk%':>7}  {'Eff.Out':>8}")
        print(f"    {'-'*55}")
        for label in ["Benign", "Suspicious", "Attack", "Total"]:
            b = bd[label]
            print(f"    {label:<12} {b['total']:>8,} "
                  f"{b['pct_forwarded']:>6.1f}% "
                  f"{b['pct_puzzle']:>6.1f}% "
                  f"{b['pct_blocked']:>6.1f}%  "
                  f"{b['effective_out']:>8,}")


def main():
    print("Loading LightGBM bundle...")
    bundle = load_bundle(BUNDLE)

    print("Loading Monday CSV...")
    mon = pd.read_csv(MON_CSV, low_memory=False)
    print(f"  Monday: {len(mon):,} flows")

    print("Loading Tuesday CSV...")
    tue = pd.read_csv(TUE_CSV, low_memory=False)
    print(f"  Tuesday: {len(tue):,} flows")

    print("Computing risk scores (Monday model on Monday data)...")
    p_risk_mon, cat_mon = get_risk_scores(mon, bundle)

    print("Computing risk scores (Monday model on Tuesday data)...")
    p_risk_tue, cat_tue = get_risk_scores(tue, bundle)

    print_day("MONDAY  (Monday LightGBM → Monday data)", p_risk_mon, cat_mon)
    print_day("TUESDAY  (Monday LightGBM → Tuesday data)", p_risk_tue, cat_tue)

    # Summary for presentation: lockdown profile breakdown
    print("\n\n" + "="*70)
    print("  SUMMARY: Lockdown profile (dominant S27 action)")
    print("="*70)
    for day_name, p_risk, cat in [("Monday", p_risk_mon, cat_mon), ("Tuesday", p_risk_tue, cat_tue)]:
        t_block, t_puzzle = 0.50, 0.10
        bd = breakdown(p_risk, cat, t_block, t_puzzle)
        total = len(p_risk)
        print(f"\n  {day_name}:")
        for label in ["Benign", "Attack"]:
            b = bd[label]
            print(f"    {label}: {b['forwarded']:,} fwd ({b['pct_forwarded']:.1f}%)  "
                  f"{b['puzzle']:,} puzzle ({b['pct_puzzle']:.1f}%)  "
                  f"{b['blocked']:,} blocked ({b['pct_blocked']:.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
