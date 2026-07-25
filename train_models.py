#!/usr/bin/env python3
"""
Train lightweight multi-disease models optimized for Streamlit Cloud.

SYNTHETIC DATA ONLY — research / demo / education.
Not for clinical use.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "clean" / "flat_encounter_level.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

DISEASES = [
    "t2dm",
    "ckd",
    "cad",
    "hf",
    "htn",
    "copd",
    "cld_masld",
    "ra",
    "hypothyroid",
    "alzheimers",
]

DISEASE_LABELS = {
    "t2dm": "Type 2 diabetes",
    "ckd": "Chronic kidney disease",
    "cad": "Coronary artery disease",
    "hf": "Heart failure",
    "htn": "Hypertension",
    "copd": "COPD",
    "cld_masld": "Chronic liver disease / MASLD",
    "ra": "Rheumatoid arthritis",
    "hypothyroid": "Hypothyroidism",
    "alzheimers": "Alzheimer’s / cognitive decline",
}

NUMERIC_FEATURES = [
    "age_at_encounter",
    "bmi",
    "waist_cm_baseline",
    "height_cm",
    "pack_years",
    "sbp_mmHg",
    "dbp_mmHg",
    "heart_rate_bpm",
    "spo2_percent",
    "weight_kg",
    "symptom_polyuria",
    "symptom_polydipsia",
    "symptom_fatigue",
    "symptom_chest_pain",
    "symptom_dyspnea",
    "symptom_edema",
    "symptom_cough",
    "symptom_joint_pain",
    "symptom_joint_swelling",
    "symptom_cognitive_complaint",
    "symptom_cold_intolerance",
    "acute_infection",
    "dehydration",
    "strenuous_exercise",
    "recent_surgery",
    "recent_hospitalization",
    "acute_illness",
    "fh_t2dm",
    "fh_ckd",
    "fh_cad",
    "fh_hf",
    "fh_htn",
    "fh_copd",
    "fh_cld_masld",
    "fh_ra",
    "fh_hypothyroid",
    "fh_alzheimers",
    "lab_hba1c",
    "lab_fasting_plasma_glucose",
    "lab_egfr",
    "lab_uacr",
    "lab_ldl_c",
    "lab_hdl_c",
    "lab_triglycerides",
    "lab_alt",
    "lab_ast",
    "lab_tsh",
    "lab_nt_probnp",
    "lab_hemoglobin",
    "lab_crp",
    "lab_anti_ccp",
    "lab_fev1_fvc_ratio",
    "lab_moca",
]

CATEGORICAL_FEATURES = [
    "sex_at_birth",
    "smoking_status",
    "alcohol_use",
    "physical_activity",
    "diet_risk",
    "ethnicity",
    "ses_group",
    "access_to_care",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run: python generate_dataset.py --n-patients 3000 --seed 42"
        )
    df = pd.read_csv(DATA_PATH)
    for d in DISEASES:
        col = f"fh_{d}"
        if col not in df.columns:
            df[col] = 0
    return df


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def eval_binary(y_true: np.ndarray, proba: np.ndarray) -> dict:
    if len(np.unique(y_true)) < 2:
        return {"auroc": None, "auprc": None, "brier": None}
    return {
        "auroc": float(roc_auc_score(y_true, proba)),
        "auprc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def main() -> None:
    print("=" * 70)
    print("Training LIGHTWEIGHT models for fast Streamlit Cloud")
    print("NOT FOR CLINICAL USE")
    print("=" * 70)

    df = load_data()
    train = df[df["partition"] == "train"].copy()
    val = df[df["partition"] == "val"].copy()
    test = df[df["partition"] == "test"].copy()
    print(f"Rows train/val/test: {len(train)} / {len(val)} / {len(test)}")

    # Fit ONE shared preprocessor (major speed win vs 11 full pipelines)
    prep = make_preprocessor()
    X_train_raw = train[ALL_FEATURES]
    X_val_raw = val[ALL_FEATURES]
    X_test_raw = test[ALL_FEATURES]
    X_train = prep.fit_transform(X_train_raw)
    X_val = prep.transform(X_val_raw)
    X_test = prep.transform(X_test_raw)

    models: dict[str, LogisticRegression] = {}
    metrics_all: list[dict] = []

    targets = {d: f"disease_present_at_encounter_{d}" for d in DISEASES}
    targets["hospitalization_12m"] = "hospitalization_within_12_months"

    for key, target in targets.items():
        if target not in train.columns:
            print(f"Skip missing target {target}")
            continue
        print(f"\nTraining {target} ...")
        y_tr = train[target].astype(int)
        y_va = val[target].astype(int)
        y_te = test[target].astype(int)
        m_tr = y_tr >= 0
        m_va = y_va >= 0
        m_te = y_te >= 0
        y_tr, y_va, y_te = y_tr[m_tr], y_va[m_va], y_te[m_te]
        Xt, Xv, Xs = X_train[m_tr.values], X_val[m_va.values], X_test[m_te.values]

        if y_tr.nunique() < 2:
            print("  skipped: single class")
            continue

        clf = LogisticRegression(
            max_iter=400,
            class_weight="balanced",
            solver="lbfgs",
            C=0.5,
            random_state=42,
        )
        clf.fit(Xt, y_tr)
        models[key] = clf

        p_va = clf.predict_proba(Xv)[:, 1]
        p_te = clf.predict_proba(Xs)[:, 1]
        va = eval_binary(y_va.values, p_va)
        te = eval_binary(y_te.values, p_te)
        row = {
            "target": target,
            "n_train": int(len(y_tr)),
            "n_val": int(len(y_va)),
            "n_test": int(len(y_te)),
            "train_prevalence": float(y_tr.mean()),
            "val_auroc": va["auroc"],
            "val_auprc": va["auprc"],
            "val_brier": va["brier"],
            "test_auroc": te["auroc"],
            "test_auprc": te["auprc"],
            "test_brier": te["brier"],
        }
        metrics_all.append(row)
        print(
            f"  val AUROC={va['auroc']:.3f} | test AUROC={te['auroc']:.3f}"
            if va["auroc"] is not None
            else "  metrics unavailable"
        )

    defaults: dict = {}
    for c in NUMERIC_FEATURES:
        defaults[c] = float(train[c].median()) if c in train and train[c].notna().any() else 0.0
    for c in CATEGORICAL_FEATURES:
        mode = train[c].mode() if c in train else pd.Series(["unknown"])
        defaults[c] = str(mode.iloc[0]) if len(mode) else "unknown"

    bundle = {
        "format": "shared_preprocessor_v2",
        "preprocessor": prep,
        "models": models,
        "diseases": DISEASES,
        "disease_labels": DISEASE_LABELS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_features": ALL_FEATURES,
        "feature_defaults": defaults,
        "generator_version": "1.2.0",
        "model_version": "2.0.0-fast",
        "trained_on": "data/clean/flat_encounter_level.csv",
        "disclaimer": (
            "SYNTHETIC RESEARCH / DEMO MODEL ONLY. "
            "Not clinically validated. Not for diagnosis or patient care."
        ),
        "metrics": metrics_all,
    }

    out_path = MODEL_DIR / "streamlit_disease_models.joblib"
    # compress=3 shrinks download/load on Streamlit Cloud
    joblib.dump(bundle, out_path, compress=3)
    with open(MODEL_DIR / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "disclaimer": bundle["disclaimer"],
                "model_version": bundle["model_version"],
                "format": bundle["format"],
                "metrics": metrics_all,
            },
            f,
            indent=2,
        )

    size_mb = out_path.stat().st_size / 1e6
    print("\n" + "=" * 70)
    print(f"Saved: {out_path} ({size_mb:.2f} MB compressed)")
    # Export pure-JSON inference for fast Streamlit Cloud runtime (no sklearn)
    try:
        from export_fast_model import main as export_fast

        export_fast()
    except Exception as exc:  # pragma: no cover
        print(f"Warning: could not export fast_inference.json: {exc}")
        print("Run manually: python export_fast_model.py")
    print("Run: streamlit run streamlit_app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
