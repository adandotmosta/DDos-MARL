from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-flow LGBM probabilities and aggregate them by time windows."
        )
    )
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=root / "Risk_Model" / "artifacts" / "vae" / "v1" / "models" / "lgbm_bundle.pkl",
        help="Path to the saved LGBM bundle.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=root / "BCC-2024" / "Monday_18_Dec_2023.csv",
        help="Input CSV day file.",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="30s",
        help="Pandas floor window, e.g. 30s, 1min, 5min.",
    )
    parser.add_argument(
        "--timestamp-col",
        type=str,
        default="timestamp",
        help="Timestamp column name.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Ground-truth label column name.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output path for window stats CSV.",
    )
    return parser.parse_args()


def clean_features(
    df: pd.DataFrame, features: list[str], handshake_mapping: dict[str, int]
) -> pd.DataFrame:
    x = df[features].copy()
    for col in x.columns:
        if col != "handshake_state":
            x[col] = pd.to_numeric(x[col], errors="coerce")
    if "handshake_state" in x.columns:
        x["handshake_state"] = (
            x["handshake_state"]
            .astype(str)
            .map(handshake_mapping)
            .fillna(-1)
            .astype(np.float32)
        )
    x = x.replace([np.inf, -np.inf], np.nan)
    return x.astype("float32")


def normalize_label(raw: str) -> str:
    s = str(raw).strip().lower()
    if s == "attack":
        return "attack"
    if s == "benign":
        return "benign"
    if s == "suspicious":
        return "suspicious"
    return s


def main() -> None:
    args = parse_args()

    with open(args.bundle_path.resolve(), "rb") as f:
        bundle = pickle.load(f)

    features = [str(x) for x in bundle["features"]]
    handshake_mapping = bundle.get("handshake_mapping", {})
    imputer = bundle["imputer"]
    # Prefer 3-class model for attack/benign/suspicious probabilities.
    model = bundle.get("model_3class")
    if model is None:
        # Fallback for older bundles.
        model = bundle.get("model")
    if model is None:
        raise ValueError("Bundle does not contain model_3class or model.")

    need_cols = list(dict.fromkeys(features + [args.timestamp_col, args.label_col]))
    df = pd.read_csv(args.csv_path.resolve(), usecols=need_cols, low_memory=False)

    x_df = clean_features(df, features, handshake_mapping)
    x = imputer.transform(x_df)

    proba = model.predict_proba(x)
    classes = [str(c) for c in model.classes_]

    for i, cls_name in enumerate(classes):
        df[f"p_{cls_name}"] = proba[:, i]

    if "p_attack" not in df.columns:
        raise ValueError(
            f"Model classes are {classes}; expected an 'attack' class for p_attack."
        )
    if "p_benign" not in df.columns:
        df["p_benign"] = np.nan
    if "p_suspicious" not in df.columns:
        df["p_suspicious"] = np.nan

    df[args.timestamp_col] = pd.to_datetime(df[args.timestamp_col], errors="coerce")
    df = df.dropna(subset=[args.timestamp_col]).copy()
    df["window"] = df[args.timestamp_col].dt.floor(args.window)

    prob_cols = [f"p_{c}" for c in classes]
    agg_map: dict[str, str] = {c: "mean" for c in prob_cols}
    agg_map["p_attack"] = "mean"
    agg_map["count"] = "sum"

    df["count"] = 1
    window_stats = (
        df.groupby(["window", args.label_col], dropna=False)
        .agg(agg_map)
        .rename(columns={c: f"mean_{c}" for c in prob_cols})
        .rename(columns={"p_attack": "mean_p_attack"})
        .reset_index()
    )

    print(window_stats.head(20).to_string(index=False))

    # Separation summary.
    label_means = (
        df.assign(_norm_label=df[args.label_col].map(normalize_label))
        .groupby("_norm_label", dropna=False)["p_attack"]
        .mean()
        .to_dict()
    )
    attack_mean = float(label_means.get("attack", np.nan))
    benign_mean = float(label_means.get("benign", np.nan))
    suspicious_mean = float(label_means.get("suspicious", np.nan))
    gap = attack_mean - benign_mean if np.isfinite(attack_mean) and np.isfinite(benign_mean) else np.nan

    print("\nPer-label mean p_attack:")
    print(
        {
            "attack": attack_mean,
            "benign": benign_mean,
            "suspicious": suspicious_mean,
            "gap_attack_minus_benign": gap,
        }
    )

    if np.isfinite(gap):
        if gap >= 0.30:
            outcome = "A (good separation)"
        elif gap >= 0.10:
            outcome = "B (partial separation)"
        else:
            outcome = "C (weak separation)"
        print(f"Separation outcome: {outcome}")

    if args.output_csv is not None:
        out_path = args.output_csv.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        window_stats.to_csv(out_path, index=False)
        print(f"Saved window stats to: {out_path}")


if __name__ == "__main__":
    main()
