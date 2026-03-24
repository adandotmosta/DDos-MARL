"""
wait_and_run.py
Watches for S18 training to complete, then automatically runs S19 → S21 → S22 → eval.

Usage (run in a new terminal while S18 is still training):
    python wait_and_run.py

You can also skip the wait and run immediately:
    python wait_and_run.py --skip-wait
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

S18_LOG    = Path("artifacts/marl/v1/ablation/S18/training_log.json")
N_EPISODES = 500
POLL_SECS  = 30
EVAL_TAG   = "s22"
STAGES     = ["s19", "s21", "s22"]


def s18_done() -> bool:
    if not S18_LOG.exists():
        return False
    try:
        with open(S18_LOG) as f:
            log = json.load(f)
        return len(log) >= N_EPISODES
    except Exception:
        return False


def run_stage(stage: str) -> None:
    print(f"\n{'='*55}")
    print(f"Starting stage: {stage.upper()}")
    print(f"{'='*55}")
    result = subprocess.run(
        [sys.executable, "main.py", "--stage", stage],
        check=False,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] Stage {stage} failed with exit code {result.returncode}. Stopping.")
        sys.exit(1)
    print(f"\n[OK] Stage {stage} complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-wait", action="store_true",
                        help="Skip waiting for S18 and run stages immediately")
    args = parser.parse_args()

    if not args.skip_wait:
        if s18_done():
            print("S18 already complete. Starting immediately...")
        else:
            print(f"Waiting for S18 to finish (polling every {POLL_SECS}s)...")
            while not s18_done():
                if S18_LOG.exists():
                    try:
                        with open(S18_LOG) as f:
                            log = json.load(f)
                        ep = log[-1]["episode"] if log else 0
                        print(f"  S18 progress: episode {ep}/{N_EPISODES}", end="\r")
                    except Exception:
                        pass
                time.sleep(POLL_SECS)
            print(f"\nS18 complete! Starting pipeline...")

    for stage in STAGES:
        run_stage(stage)

    print(f"\n{'='*55}")
    print(f"Running final evaluation (tag={EVAL_TAG})")
    print(f"{'='*55}")
    result = subprocess.run(
        [sys.executable, "main.py", "--stage", "eval", "--tag", EVAL_TAG],
        check=False,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] Eval failed.")
        sys.exit(1)

    print(f"\n{'='*55}")
    print("All done! Results saved to:")
    print(f"  artifacts/evaluation/v1/{EVAL_TAG}/")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
