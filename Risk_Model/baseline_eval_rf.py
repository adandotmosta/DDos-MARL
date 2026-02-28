from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix


LABEL_MAP = {"Benign": "benign", "Attack": "attack", "Suspicious": "suspicious"}


def parse_args() -> argparse.Namespace:
    risk_dir = Path(__file__).resolve().parent
    data_dir = risk_dir.parent / "BCC-2024"
    parser = argparse.ArgumentParser(
        description="Leak-free day-based RF evaluation (train: Thu/Sat/Mon, test: Tue)."
    )
    parser.add_argument("--base-dir", type=Path, default=data_dir)
    parser.add_argument(
        "--features-json",
        type=Path,
        default=risk_dir / "artifacts" / "vae" / "v1" / "selected_features.json",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--target-attack-recall", type=float, default=0.80)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    return parser.parse_args()


def load_features(path: Path) -> list[str]:
    with open(path) as f:
        features = json.load(f)
    if not isinstance(features, list) or not features:
        raise ValueError(f"Invalid features list in {path}")
    return [str(x) for x in features]


def read_day(csv_path: Path, features: list[str]) -> pd.DataFrame:
    usecols = features + ["label"]
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    return df


def fit_handshake_encoder(train_series: pd.Series) -> dict[str, int]:
    categories = pd.Categorical(train_series.astype(str)).categories.tolist()
    return {cat: idx for idx, cat in enumerate(categories)}


def transform_handshake(series: pd.Series, mapping: dict[str, int]) -> np.ndarray:
    return series.astype(str).map(mapping).fillna(-1).astype(np.float32).to_numpy()


def clean_features(
    df: pd.DataFrame, features: list[str], handshake_mapping: dict[str, int]
) -> pd.DataFrame:
    x = df[features].copy()
    for col in x.columns:
        if col != "handshake_state":
            x[col] = pd.to_numeric(x[col], errors="coerce")

    if "handshake_state" in x.columns:
        x["handshake_state"] = transform_handshake(x["handshake_state"], handshake_mapping)

    x = x.replace([np.inf, -np.inf], np.nan)
    return x.astype("float32")


def print_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"\n{title}")
    print("labels:", labels)
    print(cm)


def threshold_metrics_binary(
    y_true: np.ndarray, attack_proba: np.ndarray, threshold: float
) -> dict[str, float]:
    y_pred = np.where(attack_proba >= threshold, "attack", "benign")
    tp = int(np.sum((y_true == "attack") & (y_pred == "attack")))
    fn = int(np.sum((y_true == "attack") & (y_pred == "benign")))
    fp = int(np.sum((y_true == "benign") & (y_pred == "attack")))
    tn = int(np.sum((y_true == "benign") & (y_pred == "benign")))
    n = int(y_true.shape[0])

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (tp + tn) / n if n else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision_attack": float(precision),
        "recall_attack": float(recall),
        "f1_attack": float(f1),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    features = load_features(args.features_json.resolve())

    train_csvs = [
        base_dir / "Thursday_14_Dec_2023.csv",
        base_dir / "Saturday_16_Dec_2023.csv",
        base_dir / "Monday_18_Dec_2023.csv",
    ]
    test_csv = base_dir / "Tuesday_19_Dec_2023.csv"

    print("Loading train days...")
    train_df = pd.concat([read_day(p, features) for p in train_csvs], ignore_index=True)
    print("Loading test day...")
    test_df = read_day(test_csv, features)

    y_train = train_df["label"].map(LABEL_MAP).to_numpy(dtype=object)
    y_test = test_df["label"].map(LABEL_MAP).to_numpy(dtype=object)

    handshake_mapping = (
        fit_handshake_encoder(train_df["handshake_state"])
        if "handshake_state" in features
        else {}
    )

    x_train_df = clean_features(train_df, features, handshake_mapping)
    x_test_df = clean_features(test_df, features, handshake_mapping)

    # Fit imputer on train only (no leakage), apply to test.
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_train_df)
    x_test = imputer.transform(x_test_df)

    print(f"\nTrain shape: {x_train.shape}, Test shape: {x_test.shape}")
    print(
        "Train labels:",
        {k: int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
    )
    print(
        "Test labels:",
        {k: int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
    )

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        class_weight="balanced_subsample",
    )
    clf.fit(x_train, y_train)

    print("\n=== Day Split 3-Class (Tue held out) ===")
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred, digits=4))
    print_confusion(y_test, y_pred, "Confusion Matrix (3-class)")

    # Binary eval (drop suspicious from both train and test)
    keep_train = np.isin(y_train, ["benign", "attack"])
    keep_test = np.isin(y_test, ["benign", "attack"])
    x_train_bin, y_train_bin = x_train[keep_train], y_train[keep_train]
    x_test_bin, y_test_bin = x_test[keep_test], y_test[keep_test]

    clf_bin = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        class_weight="balanced_subsample",
    )
    clf_bin.fit(x_train_bin, y_train_bin)
    y_pred_bin = clf_bin.predict(x_test_bin)

    print("\n=== Day Split Binary (Tue held out, suspicious dropped) ===")
    print(classification_report(y_test_bin, y_pred_bin, digits=4))
    print_confusion(y_test_bin, y_pred_bin, "Confusion Matrix (binary)")

    # Threshold sweep for attack recall/precision trade-off.
    attack_idx = int(np.where(clf_bin.classes_ == "attack")[0][0])
    attack_proba = clf_bin.predict_proba(x_test_bin)[:, attack_idx]

    thresholds = np.arange(
        args.threshold_min, args.threshold_max + 1e-12, args.threshold_step
    )
    sweep = [threshold_metrics_binary(y_test_bin, attack_proba, float(t)) for t in thresholds]
    best_f1 = max(sweep, key=lambda d: d["f1_attack"])

    feasible = [d for d in sweep if d["recall_attack"] >= args.target_attack_recall]
    if feasible:
        best_target = max(feasible, key=lambda d: d["precision_attack"])
    else:
        best_target = None

    print("\n=== Binary Threshold Sweep (attack probability) ===")
    print("thr   acc     prec_a  rec_a   f1_a")
    for d in sweep:
        print(
            f"{d['threshold']:.2f}  {d['accuracy']:.4f}  {d['precision_attack']:.4f}  "
            f"{d['recall_attack']:.4f}  {d['f1_attack']:.4f}"
        )

    print("\nBest by F1(attack):", best_f1)
    if best_target is None:
        print(
            f"No threshold reached target attack recall >= {args.target_attack_recall:.2f}."
        )
    else:
        print(
            f"Best precision with attack recall >= {args.target_attack_recall:.2f}:",
            best_target,
        )


if __name__ == "__main__":
    main()
