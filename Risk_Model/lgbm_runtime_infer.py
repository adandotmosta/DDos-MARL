from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class LGBMBundleInfer:
    def __init__(self, bundle_path: Path, threshold: float | None = None) -> None:
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)

        self.features: list[str] = [str(x) for x in bundle["features"]]
        self.handshake_mapping: dict[str, int] = bundle["handshake_mapping"]
        self.imputer = bundle["imputer"]
        self.model_binary = bundle["model_binary"]
        self.model_3class = bundle.get("model_3class")

        if threshold is None:
            self.threshold = float(bundle.get("recommended_threshold_f1", 0.5))
        else:
            self.threshold = float(threshold)

        self.attack_idx = int(np.where(self.model_binary.classes_ == "attack")[0][0])
        self.binary_classes = [str(c) for c in self.model_binary.classes_]
        self.classes_3 = (
            [str(c) for c in self.model_3class.classes_]
            if self.model_3class is not None
            else []
        )

    def _preprocess_one(self, obs: dict[str, Any]) -> np.ndarray:
        row = {k: obs.get(k, np.nan) for k in self.features}
        df = pd.DataFrame([row])

        for col in df.columns:
            if col != "handshake_state":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "handshake_state" in df.columns:
            df["handshake_state"] = (
                df["handshake_state"]
                .astype(str)
                .map(self.handshake_mapping)
                .fillna(-1)
                .astype(np.float32)
            )

        x = df[self.features].replace([np.inf, -np.inf], np.nan).astype("float32")
        return self.imputer.transform(x)

    def predict_one(self, obs: dict[str, Any]) -> tuple[str, float]:
        x = self._preprocess_one(obs)
        p_attack = float(self.model_binary.predict_proba(x)[0, self.attack_idx])
        label = "attack" if p_attack >= self.threshold else "benign"
        return label, p_attack

    def predict_binary_proba(self, obs: dict[str, Any]) -> dict[str, float]:
        x = self._preprocess_one(obs)
        probs = self.model_binary.predict_proba(x)[0]
        return {
            cls_name: float(probs[i]) for i, cls_name in enumerate(self.binary_classes)
        }

    def predict_multi_proba(self, obs: dict[str, Any]) -> dict[str, float]:
        if self.model_3class is None:
            raise ValueError("Bundle does not contain model_3class.")
        x = self._preprocess_one(obs)
        probs = self.model_3class.predict_proba(x)[0]
        return {cls_name: float(probs[i]) for i, cls_name in enumerate(self.classes_3)}

    def predict_all(self, obs: dict[str, Any]) -> dict[str, Any]:
        label, p_attack = self.predict_one(obs)
        out = {
            "label_binary": label,
            "p_attack": p_attack,
            "proba_binary": self.predict_binary_proba(obs),
        }
        if self.model_3class is not None:
            proba_multi = self.predict_multi_proba(obs)
            out["proba_multi"] = proba_multi
            out["label_multi"] = max(proba_multi, key=proba_multi.get)
        return out


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Minimal runtime inference helper for saved LGBM bundle."
    )
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=base_dir / "artifacts" / "vae" / "v1" / "models" / "lgbm_bundle.pkl",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override default threshold from bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer = LGBMBundleInfer(args.bundle_path.resolve(), threshold=args.threshold)

    # Demo observation with missing values allowed.
    demo_obs: dict[str, Any] = {k: 0.0 for k in infer.features}
    pred = infer.predict_all(demo_obs)
    print(f"Loaded bundle: {args.bundle_path.resolve()}")
    print(f"Threshold: {infer.threshold:.4f}")
    print("Demo prediction:", pred)


if __name__ == "__main__":
    main()
