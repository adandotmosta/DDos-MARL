from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


LABEL_MAP = {"Benign": "benign", "Attack": "attack", "Suspicious": "suspicious"}


def parse_args() -> argparse.Namespace:
    risk_dir = Path(__file__).resolve().parent
    data_dir = risk_dir.parent / "BCC-2024"
    parser = argparse.ArgumentParser(
        description=(
            "Leak-free day-split boosted-tree eval "
            "(train: Thu/Sat/Mon, test: Tue)."
        ),
        epilog=(
            "Save bundle example:\n"
            "  python Risk_Model/train_eval_save_lgbm.py --split-mode random --backend lightgbm "
            "--model-bundle-out Risk_Model/artifacts/vae/v1/models/lgbm_bundle.pkl "
            "--metrics-json-out Risk_Model/artifacts/vae/v1/models/lgbm_metrics.json"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--base-dir", type=Path, default=data_dir)
    parser.add_argument(
        "--features-json",
        type=Path,
        default=risk_dir / "artifacts" / "vae" / "v1" / "selected_features.json",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="random",
        choices=["random", "day"],
        help="random: stratified random split (paper-style). day: Tue held out.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--train-csvs",
        type=str,
        default=None,
        help=(
            "Optional comma-separated CSV list for custom training set "
            "(relative to --base-dir unless absolute paths)."
        ),
    )
    parser.add_argument(
        "--test-csvs",
        type=str,
        default=None,
        help=(
            "Optional comma-separated CSV list for custom test set. "
            "If omitted with --train-csvs, the training set is also used for evaluation."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "lightgbm", "xgboost", "histgb"],
        help="Model backend. auto tries lightgbm, then xgboost, then histgb.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target-attack-recall", type=float, default=0.80)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument(
        "--model-bundle-out",
        type=Path,
        default=None,
        help="Optional pickle path to save models + preprocessors for later inference.",
    )
    parser.add_argument(
        "--metrics-json-out",
        type=Path,
        default=None,
        help="Optional JSON path to save evaluation summary and threshold sweep.",
    )
    return parser.parse_args()


def load_features(path: Path) -> list[str]:
    with open(path) as f:
        features = json.load(f)
    if not isinstance(features, list) or not features:
        raise ValueError(f"Invalid features list in {path}")
    return [str(x) for x in features]


def parse_csv_list(csv_list: str, base_dir: Path) -> list[Path]:
    names = [x.strip() for x in csv_list.split(",") if x.strip()]
    if not names:
        raise ValueError("CSV list is empty.")

    paths: list[Path] = []
    for name in names:
        p = Path(name)
        if not p.is_absolute():
            p = base_dir / p
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        paths.append(p)
    return paths


def read_day(csv_path: Path, features: list[str]) -> pd.DataFrame:
    return pd.read_csv(csv_path, usecols=features + ["label"], low_memory=False)


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


def make_model(backend: str, random_state: int, binary: bool):
    if backend in ("auto", "lightgbm"):
        try:
            from lightgbm import LGBMClassifier

            objective = "binary" if binary else "multiclass"
            num_class = None if binary else 3
            model = LGBMClassifier(
                objective=objective,
                num_class=num_class,
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
            )
            return model, "lightgbm"
        except Exception:
            if backend == "lightgbm":
                raise

    if backend in ("auto", "xgboost"):
        try:
            from xgboost import XGBClassifier

            if binary:
                model = XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=8,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=random_state,
                    n_jobs=-1,
                )
            else:
                model = XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=8,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    tree_method="hist",
                    random_state=random_state,
                    n_jobs=-1,
                )
            return model, "xgboost"
        except Exception:
            if backend == "xgboost":
                raise

    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        l2_regularization=1e-3,
        random_state=random_state,
    )
    return model, "histgb"


def make_sample_weight(y: np.ndarray, focus_attack: bool) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    total = y.shape[0]
    class_weight = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    if focus_attack and "attack" in class_weight:
        class_weight["attack"] *= 1.6
    return np.array([class_weight[c] for c in y], dtype=np.float64)


def save_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    features = load_features(args.features_json.resolve())
    split_mode_used = args.split_mode

    if args.train_csvs is not None:
        split_mode_used = "custom"
        train_csvs = parse_csv_list(args.train_csvs, base_dir)
        print("Loading custom train CSVs...")
        train_df = pd.concat([read_day(p, features) for p in train_csvs], ignore_index=True)

        if args.test_csvs is not None:
            test_csvs = parse_csv_list(args.test_csvs, base_dir)
            print("Loading custom test CSVs...")
            test_df = pd.concat([read_day(p, features) for p in test_csvs], ignore_index=True)
        else:
            print("No --test-csvs provided; using training CSVs for evaluation.")
            test_df = train_df
    else:
        all_csvs = [
            base_dir / "Thursday_14_Dec_2023.csv",
            base_dir / "Saturday_16_Dec_2023.csv",
            base_dir / "Monday_18_Dec_2023.csv",
            base_dir / "Tuesday_19_Dec_2023.csv",
        ]
        if args.split_mode == "random":
            print("Loading all days for random split...")
            all_df = pd.concat([read_day(p, features) for p in all_csvs], ignore_index=True)
            y_all = all_df["label"].map(LABEL_MAP).to_numpy(dtype=object)
            train_idx, test_idx = train_test_split(
                np.arange(len(all_df)),
                test_size=args.test_size,
                random_state=args.random_state,
                stratify=y_all,
            )
            train_df = all_df.iloc[train_idx].reset_index(drop=True)
            test_df = all_df.iloc[test_idx].reset_index(drop=True)
        else:
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

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_train_df)
    x_test = imputer.transform(x_test_df)

    print(f"\nSplit mode: {split_mode_used}")
    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")
    print("Train labels:", {k: int(v) for k, v in zip(*np.unique(y_train, return_counts=True))})
    print("Test labels:", {k: int(v) for k, v in zip(*np.unique(y_test, return_counts=True))})

    clf, used_backend = make_model(args.backend, args.random_state, binary=False)
    w_train = make_sample_weight(y_train, focus_attack=True)
    clf.fit(x_train, y_train, sample_weight=w_train)

    print(f"\nBackend used (3-class): {used_backend}")
    print("\n=== 3-Class Results ===")
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred, digits=4))
    print_confusion(y_test, y_pred, "Confusion Matrix (3-class)")

    keep_train = np.isin(y_train, ["benign", "attack"])
    keep_test = np.isin(y_test, ["benign", "attack"])
    x_train_bin, y_train_bin = x_train[keep_train], y_train[keep_train]
    x_test_bin, y_test_bin = x_test[keep_test], y_test[keep_test]

    clf_bin, used_backend_bin = make_model(args.backend, args.random_state, binary=True)
    w_train_bin = make_sample_weight(y_train_bin, focus_attack=True)
    clf_bin.fit(x_train_bin, y_train_bin, sample_weight=w_train_bin)

    print(f"\nBackend used (binary): {used_backend_bin}")
    y_pred_bin = clf_bin.predict(x_test_bin)
    print("\n=== Binary Results (suspicious dropped) ===")
    print(classification_report(y_test_bin, y_pred_bin, digits=4))
    print_confusion(y_test_bin, y_pred_bin, "Confusion Matrix (binary)")

    attack_idx = int(np.where(clf_bin.classes_ == "attack")[0][0])
    attack_proba = clf_bin.predict_proba(x_test_bin)[:, attack_idx]
    thresholds = np.arange(
        args.threshold_min, args.threshold_max + 1e-12, args.threshold_step
    )
    sweep = [threshold_metrics_binary(y_test_bin, attack_proba, float(t)) for t in thresholds]
    best_f1 = max(sweep, key=lambda d: d["f1_attack"])
    feasible = [d for d in sweep if d["recall_attack"] >= args.target_attack_recall]
    best_target = max(feasible, key=lambda d: d["precision_attack"]) if feasible else None

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

    summary = {
        "split_mode": split_mode_used,
        "backend_requested": args.backend,
        "backend_used_3class": used_backend,
        "backend_used_binary": used_backend_bin,
        "train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
        "test_shape": [int(x_test.shape[0]), int(x_test.shape[1])],
        "train_labels": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "test_labels": {str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        "best_f1_attack": best_f1,
        "best_precision_with_target_recall": best_target,
        "threshold_sweep": sweep,
    }

    if args.metrics_json_out is not None:
        out_json = args.metrics_json_out.resolve()
        save_json(out_json, summary)
        print(f"\nSaved metrics JSON to: {out_json}")

    if args.model_bundle_out is not None:
        out_bundle = args.model_bundle_out.resolve()
        payload = {
            "features": features,
            "handshake_mapping": handshake_mapping,
            "imputer": imputer,
            "model_3class": clf,
            "model_binary": clf_bin,
            "classes_3class": list(map(str, clf.classes_)),
            "classes_binary": list(map(str, clf_bin.classes_)),
            "recommended_threshold_f1": float(best_f1["threshold"]),
            "recommended_threshold_target_recall": (
                None if best_target is None else float(best_target["threshold"])
            ),
            "summary": summary,
        }
        save_pickle(out_bundle, payload)
        print(f"Saved model bundle to: {out_bundle}")


if __name__ == "__main__":
    main()
