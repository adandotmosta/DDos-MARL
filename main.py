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
        choices=["all", "lgbm", "lgbm_combined", "marl", "iql", "ippo", "mapl", "e13", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s21", "s22", "s23", "s24", "s25", "s26", "s26b", "s27", "s28", "s29", "s30", "s31", "s32", "s33", "s34", "s35", "s36", "s37", "s38", "s39", "s40", "eval"],
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
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to evaluate (e.g. s23,s24,ippo). "
             "If omitted, all models are evaluated. Only used with --stage eval.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.stage in ("lgbm_combined",):
        print("\n" + "=" * 55)
        print("STAGE 1b: LightGBM — Combined (Monday + Tuesday)")
        print("=" * 55)
        import subprocess
        import sys as _sys
        result = subprocess.run(
            [_sys.executable, "train_lgbm_combined.py"], check=False
        )
        if result.returncode != 0:
            import sys; sys.exit(result.returncode)

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

    if args.stage in ("s15",):
        print("\n" + "=" * 55)
        print("STAGE 2j: MAPPO + Local Reward + Risk-Aware Throttle (S15)")
        print("=" * 55)
        from src.train_s15 import train as train_s15

        train_s15()

    if args.stage in ("s16",):
        print("\n" + "=" * 55)
        print("STAGE 2k: MAPPO + Local Reward + Dynamic Puzzle (S16)")
        print("=" * 55)
        from src.train_s16 import train as train_s16

        train_s16()

    if args.stage in ("s17",):
        print("\n" + "=" * 55)
        print("STAGE 2l: MAPPO + Local Reward + Entropy State (S17)")
        print("=" * 55)
        from src.train_s17 import train as train_s17

        train_s17()

    if args.stage in ("s19",):
        print("\n" + "=" * 55)
        print("STAGE 2n: MAPPO + Local Reward + Asymmetric Reward (S19)")
        print("=" * 55)
        from src.train_s19 import train as train_s19

        train_s19()

    if args.stage in ("s21",):
        print("\n" + "=" * 55)
        print("STAGE 2o: MAPPO + Local Reward + Lower Threshold (S21)")
        print("=" * 55)
        from src.train_s21 import train as train_s21

        train_s21()

    if args.stage in ("s22",):
        print("\n" + "=" * 55)
        print("STAGE 2p: MAPPO + Counterfactual Reward (S22)")
        print("=" * 55)
        from src.train_s22 import train as train_s22

        train_s22()

    if args.stage in ("s24",):
        print("\n" + "=" * 55)
        print("STAGE 2r: MAPPO + Selective Puzzle + Combined LightGBM (S24)")
        print("=" * 55)
        from src.train_s24 import train as train_s24

        train_s24()

    if args.stage in ("s23",):
        print("\n" + "=" * 55)
        print("STAGE 2q: MAPPO + Selective Puzzle (S23)")
        print("=" * 55)
        from src.train_s23 import train as train_s23

        train_s23()

    if args.stage in ("s25",):
        print("\n" + "=" * 55)
        print("STAGE 2s: MAPPO + Puzzle on Flagged IPs + p_attack Pass Rates (S25)")
        print("=" * 55)
        from src.train_s25 import train as train_s25

        train_s25()

    if args.stage in ("s26",):
        print("\n" + "=" * 55)
        print("STAGE 2t: MAPPO + Tiered Action Profiles (S26)")
        print("=" * 55)
        from src.train_s26 import train as train_s26

        train_s26()

    if args.stage in ("s26b",):
        print("\n" + "=" * 55)
        print("STAGE 2u: MAPPO + Tiered Profiles + Monday-only LightGBM (S26b)")
        print("=" * 55)
        from src.train_s26b import train as train_s26b

        train_s26b()

    if args.stage in ("s27",):
        print("\n" + "=" * 55)
        print("STAGE 2v: MAPPO + p_risk Tiered Profiles + Monday-only LightGBM (S27)")
        print("=" * 55)
        from src.train_s27 import train as train_s27

        train_s27()

    if args.stage in ("s28",):
        print("\n" + "=" * 55)
        print("STAGE 2w: MAPPO + p_risk Tiered Profiles + Mixed Training Monday+Thursday (S28)")
        print("=" * 55)
        from src.train_s28 import train as train_s28

        train_s28()

    if args.stage in ("s29",):
        print("\n" + "=" * 55)
        print("STAGE 2x: IPPO + p_risk Tiered Profiles + Monday-only LightGBM (S29)")
        print("=" * 55)
        from src.train_s29 import train as train_s29

        train_s29()

    if args.stage in ("s30",):
        print("\n" + "=" * 55)
        print("STAGE 2y: MAPPO + p_risk Tiered Profiles + Combined LightGBM (S30)")
        print("=" * 55)
        from src.train_s30 import train as train_s30

        train_s30()

    if args.stage in ("s31",):
        print("\n" + "=" * 55)
        print("STAGE S31: IPPO + p_risk Tiered Profiles + Monday-only LightGBM + 6 agents (S31)")
        print("=" * 55)
        from src.train_s31 import train as train_s31

        train_s31()

    if args.stage in ("s32",):
        print("\n" + "=" * 55)
        print("STAGE S32: IPPO + p_risk Tiered Profiles + 1-dim state [p_risk only] (S32)")
        print("=" * 55)
        from src.train_s32 import train as train_s32

        train_s32()

    if args.stage in ("s33",):
        print("\n" + "=" * 55)
        print("STAGE S33: MAPPO + Shared IP Blocklist + 5-dim state (S33)")
        print("=" * 55)
        from src.train_s33 import train as train_s33
        train_s33()

    if args.stage in ("s34",):
        print("\n" + "=" * 55)
        print("STAGE S34: IPPO + Shared IP Blocklist + 5-dim state (S34)")
        print("=" * 55)
        from src.train_s34 import train as train_s34
        train_s34()

    if args.stage in ("s35",):
        print("\n" + "=" * 55)
        print("STAGE S35: MAPPO + Victim Routing + Shared IP Blocklist (S35)")
        print("=" * 55)
        from src.train_s35 import train as train_s35
        train_s35()

    if args.stage in ("s36",):
        print("\n" + "=" * 55)
        print("STAGE S36: IPPO + Victim Routing + Shared IP Blocklist (S36)")
        print("=" * 55)
        from src.train_s36 import train as train_s36
        train_s36()

    if args.stage in ("s37",):
        print("\n" + "=" * 55)
        print("STAGE S37: MAPPO + p_risk Tiered Profiles + 6 agents (S37)")
        print("=" * 55)
        from src.train_s37 import train as train_s37
        train_s37()

    if args.stage in ("s38",):
        print("\n" + "=" * 55)
        print("STAGE S38: IPPO + p_risk Tiered Profiles + ECMP 5-tuple routing (S38)")
        print("=" * 55)
        from src.train_s38 import train as train_s38
        train_s38()

    if args.stage in ("s39",):
        print("\n" + "=" * 55)
        print("STAGE S39: MAPPO + End-to-End FlowEncoder (replaces LightGBM) (S39)")
        print("=" * 55)
        from src.train_s39 import train as train_s39
        train_s39()

    if args.stage in ("s40",):
        print("\n" + "=" * 55)
        print("STAGE S40: MAPPO + p_risk Tiered Profiles + GLOBAL Reward (S40)")
        print("=" * 55)
        from src.train_s40 import train as train_s40
        train_s40()

    if args.stage in ("s18",):
        print("\n" + "=" * 55)
        print("STAGE 2m: MAPPO + Local Reward + Curriculum Learning (S18)")
        print("=" * 55)
        from src.train_s18 import train as train_s18

        train_s18()

    if args.stage in ("all", "eval"):
        print("\n" + "=" * 55)
        print("STAGE 3: Evaluation")
        print("=" * 55)
        # Lazy import so --stage lgbm does not require torch.
        from src.evaluate import main as evaluate

        models_filter = (
            {m.strip().lower() for m in args.models.split(",")}
            if args.models else None
        )
        evaluate(tag=args.tag, model_tag=args.model_tag, models=models_filter)


if __name__ == "__main__":
    main()
