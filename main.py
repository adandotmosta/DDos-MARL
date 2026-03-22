"""
main.py
Full CoopIDS pipeline entry point.

Run full pipeline:
    python main.py

Run individual stages:
    python main.py --stage lgbm
    python main.py --stage marl
    python main.py --stage eval
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoopIDS pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "lgbm", "marl", "iql", "ippo", "mapl", "e13", "s11", "s12", "s13", "s14", "eval"],
        help="Which stage to run",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="baseline",
        help="Experiment tag — results saved to artifacts/evaluation/v1/<tag>/",
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default=None,
        help="Model tag — MAPPO saved/loaded from artifacts/marl/v1/experiments/<model_tag>/. "
             "Defaults to MARL_DIR (root) when not set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.stage in ("all", "lgbm"):
        print("\n" + "=" * 55)
        print("STAGE 1: LightGBM Risk Model")
        print("=" * 55)
        print(
            "Run manually: python Risk_Model/train_eval_save_lgbm.py "
            "--split-mode day --backend lightgbm "
            "--model-bundle-out Risk_Model/artifacts/vae/v1/models/"
            "lgbm_bundle_monday.pkl"
        )
        print("(Already done - skipping)")

    if args.stage in ("all", "marl"):
        print("\n" + "=" * 55)
        print("STAGE 2: MARL Training")
        print("=" * 55)
        # Lazy import so --stage lgbm does not require torch.
        from src.train_marl import train

        train(model_tag=args.model_tag)

    if args.stage in ("all", "iql"):
        print("\n" + "=" * 55)
        print("STAGE 2b: IQL Training (ablation S6)")
        print("=" * 55)
        from src.train_iql import train as train_iql

        train_iql()

    if args.stage in ("all", "ippo"):
        print("\n" + "=" * 55)
        print("STAGE 2c: IPPO Training (ablation S7)")
        print("=" * 55)
        from src.train_ippo import train as train_ippo

        train_ippo()

    if args.stage in ("mapl",):
        print("\n" + "=" * 55)
        print("STAGE 2d: MAPPO + Local Reward (ablation S9)")
        print("=" * 55)
        from src.train_mappo_local import train as train_mapl

        train_mapl()

    if args.stage in ("e13",):
        print("\n" + "=" * 55)
        print("STAGE 2e: MAPPO + Extended State (E13 / S10)")
        print("=" * 55)
        from src.train_e13 import train as train_e13

        train_e13()

    if args.stage in ("s11",):
        print("\n" + "=" * 55)
        print("STAGE 2f: MAPPO + 10-Action Space (S11)")
        print("=" * 55)
        from src.train_s11 import train as train_s11

        train_s11()

    if args.stage in ("s12",):
        print("\n" + "=" * 55)
        print("STAGE 2g: MAPPO + No-Puzzle (S12)")
        print("=" * 55)
        from src.train_s12 import train as train_s12

        train_s12()

    if args.stage in ("s13",):
        print("\n" + "=" * 55)
        print("STAGE 2h: MAPPO + Local Reward + Puzzle Cost 0.05 (S13)")
        print("=" * 55)
        from src.train_s13 import train as train_s13

        train_s13()

    if args.stage in ("s14",):
        print("\n" + "=" * 55)
        print("STAGE 2i: MAPPO + Local Reward + Puzzle Block Rate State (S14)")
        print("=" * 55)
        from src.train_s14 import train as train_s14

        train_s14()

    if args.stage in ("all", "eval"):
        print("\n" + "=" * 55)
        print("STAGE 3: Evaluation")
        print("=" * 55)
        # Lazy import so --stage lgbm does not require torch.
        from src.evaluate import main as evaluate

        evaluate(tag=args.tag, model_tag=args.model_tag)


if __name__ == "__main__":
    main()
