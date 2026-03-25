"""
analyse_dataset.py
Comprehensive dataset structure analysis for BCC-2024.

Answers:
  1. Label distribution per day (counts + %)
  2. Temporal flow of labels — how traffic evolves over time
  3. Attack windows vs rest periods (matches paper schedule)
  4. Flow volume time series per label
  5. Class imbalance across days
  6. Attack type breakdown (if activity column exists)
  7. Benign activity breakdown
  8. Key feature statistics per label per day

Usage:
    python analyse_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CSV_MONDAY, CSV_TUESDAY, CSV_THURSDAY, CSV_SATURDAY,
)

# ── Attack schedule from paper (Table A1) ────────────────────────────────────
MONDAY_ATTACKS = [
    ("09:00", "09:20", "TCP-SYN-Valid",   "Windows-machine-1"),
    ("09:30", "09:50", "TCP-BYPass-V1",   "Linux-admin-1"),
    ("10:00", "10:20", "Killall-v2",      "Linux-webserver"),
    ("10:32", "10:52", "TCP-IGMP",        "Windows-machine-2"),
    ("11:00", "11:20", "TCP-SYN",         "Linux-admin-1"),
    ("11:30", "11:50", "Killer-TCP",      "Windows-machine-4"),
    ("13:00", "13:20", "TCP-Control",     "Linux-webserver"),
    ("13:30", "13:50", "TCP-MIX",         "Linux-admin-1"),
    ("14:00", "14:20", "TCP-Flag-SYN",    "Windows-machine-2"),
    ("14:30", "14:50", "TCP-ACK",         "Windows-machine-3"),
    ("15:00", "15:20", "TCP-SYN-ACK",     "Linux-webserver"),
    ("15:30", "15:50", "TCP-ACK-PSH",     "Windows-machine-4"),
]

TUESDAY_ATTACKS = [
    ("09:00", "09:20", "TCP-RST-ACK",  "Linux-webserver"),
    ("09:30", "09:50", "TCP-SYN-TFO",  "Windows-machine-3"),
    ("10:00", "10:20", "TCP-SYN-TIME", "Linux-admin-1"),
    ("10:50", "11:10", "TCP-OSYN",     "Windows-machine-1"),
    ("11:20", "11:40", "TCP-OSYNP",    "Linux-webserver"),
]


def load_day(path: Path, name: str) -> pd.DataFrame:
    print(f"  Loading {name} from {path.name}...")
    df = pd.read_csv(path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"    {len(df):,} flows  |  columns: {len(df.columns)}")
    return df


def section(title: str) -> None:
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")


def subsection(title: str) -> None:
    print(f"\n  {'─'*80}")
    print(f"  {title}")
    print(f"  {'─'*80}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Label distribution
# ─────────────────────────────────────────────────────────────────────────────

def label_distribution(days: dict[str, pd.DataFrame]) -> None:
    section("1. LABEL DISTRIBUTION PER DAY")

    all_labels = ["Benign", "Suspicious", "Attack"]

    # Header
    print(f"\n  {'Label':<14}", end="")
    for name in days:
        print(f"  {name:>20}", end="")
    print()
    print(f"  {'─'*14}", end="")
    for _ in days:
        print(f"  {'─'*20}", end="")
    print()

    for label in all_labels:
        print(f"  {label:<14}", end="")
        for df in days.values():
            n = (df["label"] == label).sum()
            pct = 100 * n / len(df)
            print(f"  {n:>10,} ({pct:>5.1f}%)", end="")
        print()

    print(f"  {'─'*14}", end="")
    for _ in days:
        print(f"  {'─'*20}", end="")
    print()
    print(f"  {'TOTAL':<14}", end="")
    for df in days.values():
        print(f"  {len(df):>20,}", end="")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Temporal label flow — flows per 30-min bucket
# ─────────────────────────────────────────────────────────────────────────────

def temporal_flow(name: str, df: pd.DataFrame, window_min: int = 30) -> None:
    subsection(f"Temporal label flow — {name}  (buckets: {window_min} min)")

    df = df.copy()
    df["bucket"] = df["timestamp"].dt.floor(f"{window_min}min")
    grouped = df.groupby(["bucket", "label"]).size().unstack(fill_value=0)

    for col in ["Benign", "Suspicious", "Attack"]:
        if col not in grouped.columns:
            grouped[col] = 0

    print(f"\n  {'Time':<8} {'Benign':>8} {'Suspicious':>10} {'Attack':>8}  visual")
    print(f"  {'─'*60}")

    max_total = grouped.sum(axis=1).max()
    for ts, row in grouped.iterrows():
        b = row.get("Benign", 0)
        s = row.get("Suspicious", 0)
        a = row.get("Attack", 0)
        total = b + s + a
        bar_len = int(40 * total / max(max_total, 1))
        a_len = int(40 * a / max(max_total, 1))
        s_len = int(40 * s / max(max_total, 1))
        b_len = bar_len - a_len - s_len

        bar = "A" * a_len + "S" * s_len + "." * b_len
        print(f"  {ts.strftime('%H:%M'):<8} {b:>8,} {s:>10,} {a:>8,}  |{bar}|")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Attack window verification (match paper schedule)
# ─────────────────────────────────────────────────────────────────────────────

def attack_window_check(name: str, df: pd.DataFrame, schedule: list) -> None:
    subsection(f"Attack window verification — {name}")

    if "Attack" not in df["label"].values:
        print(f"  No attack flows in {name} (expected for benign-only day)")
        return

    attacks = df[df["label"] == "Attack"].copy()
    date_str = attacks["timestamp"].dt.date.iloc[0]

    print(f"\n  {'Attack type':<22} {'Scheduled':>12} {'Actual start':>14} {'Actual end':>12} {'Flows':>8}")
    print(f"  {'─'*72}")

    for start_t, end_t, atk_name, target in schedule:
        window_start = pd.Timestamp(f"{date_str} {start_t}")
        window_end   = pd.Timestamp(f"{date_str} {end_t}")
        window_flows = attacks[
            (attacks["timestamp"] >= window_start) &
            (attacks["timestamp"] <= window_end)
        ]
        n = len(window_flows)
        if n > 0:
            actual_start = window_flows["timestamp"].min().strftime("%H:%M:%S")
            actual_end   = window_flows["timestamp"].max().strftime("%H:%M:%S")
        else:
            actual_start = "—"
            actual_end   = "—"
        sched = f"{start_t}–{end_t}"
        print(f"  {atk_name:<22} {sched:>12} {actual_start:>14} {actual_end:>12} {n:>8,}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Activity column breakdown (if exists)
# ─────────────────────────────────────────────────────────────────────────────

def activity_breakdown(name: str, df: pd.DataFrame) -> None:
    # Check if an activity/sub-label column exists
    activity_cols = [c for c in df.columns if "activity" in c.lower() or "class" in c.lower()]
    if not activity_cols:
        # Try to infer from label column variants
        print(f"\n  No activity column found in {name} — label column only has: "
              f"{df['label'].unique().tolist()}")
        return

    col = activity_cols[0]
    subsection(f"Activity breakdown — {name}  (column: {col})")
    counts = df[col].value_counts()
    for act, n in counts.items():
        pct = 100 * n / len(df)
        print(f"  {act:<40} {n:>8,}  ({pct:>5.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Key numeric feature stats per label
# ─────────────────────────────────────────────────────────────────────────────

def feature_stats(name: str, df: pd.DataFrame) -> None:
    subsection(f"Key feature stats per label — {name}")

    # Pick a few interpretable features if they exist
    candidate_features = [
        "bytes_rate", "duration", "pkts_rate",
        "fwd_pkts_rate", "bwd_pkts_rate",
        "total_hdr_byte", "mean_hdr_byte",
    ]
    available = [f for f in candidate_features if f in df.columns]

    if not available:
        print("  No standard numeric features found (may be in different column names)")
        # Show first 5 numeric columns instead
        available = df.select_dtypes(include=[np.number]).columns[:5].tolist()

    labels = ["Benign", "Suspicious", "Attack"]
    for feat in available:
        print(f"\n  [{feat}]")
        print(f"  {'Label':<14} {'mean':>10} {'median':>10} {'p10':>10} {'p90':>10} {'max':>12}")
        print(f"  {'─'*60}")
        for lbl in labels:
            sub = df[df["label"] == lbl][feat].dropna()
            if len(sub) == 0:
                continue
            print(f"  {lbl:<14} {sub.mean():>10.2f} {sub.median():>10.2f} "
                  f"{sub.quantile(0.1):>10.2f} {sub.quantile(0.9):>10.2f} {sub.max():>12.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Unique IPs per label (DDoS signature)
# ─────────────────────────────────────────────────────────────────────────────

def ip_analysis(name: str, df: pd.DataFrame) -> None:
    subsection(f"IP analysis per label — {name}")

    labels_present = df["label"].unique()
    print(f"\n  {'Label':<14} {'Unique src IPs':>16} {'Unique dst IPs':>16} "
          f"{'Flows/src IP (mean)':>20}")
    print(f"  {'─'*68}")

    for lbl in ["Benign", "Suspicious", "Attack"]:
        if lbl not in labels_present:
            continue
        sub = df[df["label"] == lbl]
        n_src = sub["src_ip"].nunique() if "src_ip" in sub.columns else "—"
        n_dst = sub["dst_ip"].nunique() if "dst_ip" in sub.columns else "—"
        if "src_ip" in sub.columns and isinstance(n_src, int) and n_src > 0:
            flows_per_src = len(sub) / n_src
        else:
            flows_per_src = 0
        print(f"  {lbl:<14} {str(n_src):>16} {str(n_dst):>16} {flows_per_src:>20.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Cross-day attack comparison
# ─────────────────────────────────────────────────────────────────────────────

def cross_day_summary(days: dict[str, pd.DataFrame]) -> None:
    section("7. CROSS-DAY SUMMARY — Attack type diversity")

    print("""
  Monday attacks  (12 types): TCP-SYN-Valid, TCP-BYPass-V1, Killall-v2,
    TCP-IGMP, TCP-SYN, Killer-TCP, TCP-Control, TCP-MIX,
    TCP-Flag-SYN, TCP-ACK, TCP-SYN-ACK, TCP-ACK-PSH

  Tuesday attacks (5 types):  TCP-RST-ACK, TCP-SYN-TFO, TCP-SYN-TIME,
    TCP-OSYN, TCP-OSYNP

  Thursday / Saturday: BENIGN ONLY — 0 attacks

  Key implication for MARL:
    - Agents train on 12 Monday attack types → learn Monday patterns
    - Tuesday has 5 DIFFERENT attack types → distribution shift
    - Thursday/Saturday: no attacks at all → FPR-only evaluation
    """)

    print(f"  {'Day':<12} {'Total flows':>12} {'Attack':>10} {'Benign':>10} "
          f"{'Suspicious':>12} {'Attack %':>10}")
    print(f"  {'─'*68}")
    for name, df in days.items():
        n = len(df)
        a = (df["label"] == "Attack").sum()
        b = (df["label"] == "Benign").sum()
        s = (df["label"] == "Suspicious").sum()
        print(f"  {name:<12} {n:>12,} {a:>10,} {b:>10,} {s:>12,} {100*a/n:>9.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 90)
    print("  BCC-2024 DATASET STRUCTURE ANALYSIS")
    print("  Understanding label distributions, temporal patterns, and attack diversity")
    print("=" * 90)

    # Load all days
    print("\nLoading datasets...")
    days: dict[str, pd.DataFrame] = {}

    for name, path in [
        ("Thursday",  CSV_THURSDAY),
        ("Saturday",  CSV_SATURDAY),
        ("Monday",    CSV_MONDAY),
        ("Tuesday",   CSV_TUESDAY),
    ]:
        if path.exists():
            days[name] = load_day(path, name)
        else:
            print(f"  {name}: NOT FOUND at {path}")

    # 1. Label distribution
    label_distribution(days)

    # 2. Temporal flow per day
    section("2. TEMPORAL LABEL FLOW (30-min buckets)  A=Attack  S=Suspicious  .=Benign")
    for name, df in days.items():
        temporal_flow(name, df, window_min=30)

    # 3. Attack window verification
    section("3. ATTACK WINDOW VERIFICATION (match paper Table A1)")
    if "Monday" in days:
        attack_window_check("Monday", days["Monday"], MONDAY_ATTACKS)
    if "Tuesday" in days:
        attack_window_check("Tuesday", days["Tuesday"], TUESDAY_ATTACKS)

    # 4. Activity column breakdown
    section("4. ACTIVITY / SUB-LABEL BREAKDOWN")
    for name, df in days.items():
        activity_breakdown(name, df)

    # 5. Feature stats
    section("5. KEY FEATURE STATISTICS PER LABEL")
    for name, df in days.items():
        feature_stats(name, df)

    # 6. IP analysis
    section("6. IP DIVERSITY PER LABEL (DDoS = many src IPs)")
    for name, df in days.items():
        ip_analysis(name, df)

    # 7. Cross-day summary
    cross_day_summary(days)

    print("\n" + "=" * 90)
    print("  DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
