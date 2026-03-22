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

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="CoopIDS pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "lgbm", "marl", "eval"],
        help="Which stage to run"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.stage in ("all", "lgbm"):
        print("\n" + "="*55)
        print("STAGE 1: LightGBM Risk Model")
        print("="*55)
        print("Run manually: python Risk_Model/train_eval_save_lgbm.py "
              "--split-mode day --backend lightgbm "
              "--model-bundle-out Risk_Model/artifacts/vae/v1/models/lgbm_bundle_monday.pkl")
        print("(Already done — skipping)")

    if args.stage in ("all", "marl"):
        print("\n" + "="*55)
        print("STAGE 2: MARL Training")
        print("="*55)
        from src.train_marl import train
        train()

    if args.stage in ("all", "eval"):
        print("\n" + "="*55)
        print("STAGE 3: Evaluation")
        print("="*55)
        from src.evaluate import main as evaluate
        evaluate()


if __name__ == "__main__":
    main()
