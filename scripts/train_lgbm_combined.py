"""
train_lgbm_combined.py
Train LightGBM on Monday + Tuesday combined data.

PURPOSE: Isolate MARL throttling quality from LightGBM generalization quality.
  - Standard setup (S23): LightGBM trained on Monday only → degrades on Tuesday
  - Combined setup (S24): LightGBM trained on Monday + Tuesday → accurate on both days

By training MARL (S24) on Monday data with the combined LightGBM and evaluating
on Tuesday, we measure how well MARL's throttling POLICY works when the detector
is accurate — independent of LightGBM's cross-day generalization gap.

The gap between S23 (Monday LightGBM) and S24 (combined LightGBM) on Tuesday
tells you exactly how much of the performance drop is LightGBM's fault vs MARL's.

NOTE: This bundle should NOT be used for standard generalization experiments —
      training on Tuesday and evaluating on Tuesday is leakage for detection.
      It is only valid for isolating MARL's response policy quality.

Usage:
    python train_lgbm_combined.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    from config import CSV_MONDAY, CSV_TUESDAY, LGBM_BUNDLE_COMBINED, FEATURES_JSON
except ImportError:
    from coopids.config import CSV_MONDAY, CSV_TUESDAY, LGBM_BUNDLE_COMBINED, FEATURES_JSON

TRAIN_SCRIPT = Path(__file__).resolve().parent / "Risk_Model" / "train_eval_save_lgbm.py"


def main() -> None:
    LGBM_BUNDLE_COMBINED.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training LightGBM on Monday + Tuesday combined")
    print("=" * 60)
    print(f"  Train CSVs : {CSV_MONDAY.name}, {CSV_TUESDAY.name}")
    print(f"  Output     : {LGBM_BUNDLE_COMBINED}")
    print()

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--train-csvs", f"{CSV_MONDAY},{CSV_TUESDAY}",
        "--backend", "lightgbm",
        "--features-json", str(FEATURES_JSON),
        "--model-bundle-out", str(LGBM_BUNDLE_COMBINED),
    ]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] LightGBM training failed (exit code {result.returncode}).")
        sys.exit(1)

    print(f"\n[OK] Combined bundle saved to: {LGBM_BUNDLE_COMBINED}")


if __name__ == "__main__":
    main()
