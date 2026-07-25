#!/usr/bin/env python3
"""Export sklearn training bundle to pure-JSON/NumPy inference (no sklearn at runtime)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "models" / "streamlit_disease_models.joblib"
OUT = ROOT / "models" / "fast_inference.json"


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing {SRC}. Run train_models.py first.")

    b = joblib.load(SRC)
    prep = b["preprocessor"]
    num_pipe = prep.named_transformers_["num"]
    cat_pipe = prep.named_transformers_["cat"]
    imputer = num_pipe.named_steps["imputer"]
    scaler = num_pipe.named_steps["scaler"]
    ohe = cat_pipe.named_steps["onehot"]

    scale = [float(s) if abs(float(s)) > 1e-12 else 1.0 for s in scaler.scale_]
    categories = [list(map(str, cats)) for cats in ohe.categories_]

    heads = {}
    for name, clf in b["models"].items():
        heads[name] = {
            "coef": [float(x) for x in clf.coef_.ravel().tolist()],
            "intercept": float(clf.intercept_.ravel()[0]),
        }

    export = {
        "format": "numpy_logistic_v1",
        "model_version": "3.0.0-numpy-fast",
        "generator_version": b.get("generator_version"),
        "disclaimer": b.get("disclaimer"),
        "diseases": b["diseases"],
        "disease_labels": b["disease_labels"],
        "numeric_features": b["numeric_features"],
        "categorical_features": b["categorical_features"],
        "all_features": b["all_features"],
        "feature_defaults": b["feature_defaults"],
        "num_fill": [float(x) for x in imputer.statistics_.tolist()],
        "scaler_mean": [float(x) for x in scaler.mean_.tolist()],
        "scaler_scale": scale,
        "categories": categories,
        "heads": heads,
        "metrics": b.get("metrics", []),
    }

    OUT.write_text(json.dumps(export), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes), heads={len(heads)}")


if __name__ == "__main__":
    main()
