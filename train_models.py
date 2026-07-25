#!/usr/bin/env python3
"""
Train multi-disease screening models on the synthetic cohort for Streamlit use.

SYNTHETIC DATA ONLY — research / demo / education.
Models are NOT clinically validated and must not be used for patient care.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

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

# Features available at prediction time (no latent / target leakage)
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
    # Ensure family-history columns exist (from patient merge)
    for d in DISEASES:
        col = f"fh_{d}"
        if col not in df.columns:
            df[col] = 0
    return df


def make_preprocessor() -> ColumnTransformer:
    # HistGradientBoosting handles NaN for numeric; still impute categoricals after OHE
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def train_binary_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target: str,
    random_state: int = 42,
) -> tuple[Pipeline, dict]:
    y_train = train[target].astype(int)
    y_val = val[target].astype(int)

    # Drop ineligible if any (-1)
    tr_mask = y_train >= 0
    va_mask = y_val >= 0
    X_train = train.loc[tr_mask, ALL_FEATURES]
    y_train = y_train.loc[tr_mask]
    X_val = val.loc[va_mask, ALL_FEATURES]
    y_val = y_val.loc[va_mask]

    if y_train.nunique() < 2:
        raise ValueError(f"Target {target} has <2 classes in train")

    # Class imbalance
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    w_pos = n_neg / max(n_pos, 1)
    sample_weight = np.where(y_train == 1, w_pos, 1.0)

    pipe = Pipeline(
        steps=[
            ("prep", make_preprocessor()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=6,
                    max_iter=200,
                    learning_rate=0.08,
                    min_samples_leaf=25,
                    l2_regularization=0.1,
                    random_state=random_state,
                    class_weight=None,  # use sample_weight instead
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train, clf__sample_weight=sample_weight)

    proba = pipe.predict_proba(X_val)[:, 1]
    metrics = {
        "target": target,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "train_prevalence": float(y_train.mean()),
        "val_prevalence": float(y_val.mean()),
        "val_auroc": float(roc_auc_score(y_val, proba)) if y_val.nunique() > 1 else None,
        "val_auprc": float(average_precision_score(y_val, proba)) if y_val.nunique() > 1 else None,
        "val_brier": float(brier_score_loss(y_val, proba)),
    }
    return pipe, metrics


def evaluate_test(pipe: Pipeline, test: pd.DataFrame, target: str) -> dict:
    y = test[target].astype(int)
    mask = y >= 0
    X = test.loc[mask, ALL_FEATURES]
    y = y.loc[mask]
    if len(y) == 0 or y.nunique() < 2:
        return {"target": target, "test_auroc": None, "test_auprc": None, "n_test": int(len(y))}
    proba = pipe.predict_proba(X)[:, 1]
    return {
        "target": target,
        "n_test": int(len(y)),
        "test_prevalence": float(y.mean()),
        "test_auroc": float(roc_auc_score(y, proba)),
        "test_auprc": float(average_precision_score(y, proba)),
        "test_brier": float(brier_score_loss(y, proba)),
    }


def main() -> None:
    print("=" * 70)
    print("Training synthetic multi-disease models for Streamlit")
    print("NOT FOR CLINICAL USE")
    print("=" * 70)

    df = load_data()
    train = df[df["partition"] == "train"].copy()
    val = df[df["partition"] == "val"].copy()
    test = df[df["partition"] == "test"].copy()
    print(f"Rows train/val/test: {len(train)} / {len(val)} / {len(test)}")

    models: dict[str, Pipeline] = {}
    metrics_all: list[dict] = []

    # Per-disease presence at encounter
    for d in DISEASES:
        target = f"disease_present_at_encounter_{d}"
        print(f"\nTraining {target} ...")
        pipe, m = train_binary_model(train, val, target)
        t = evaluate_test(pipe, test, target)
        m.update(t)
        models[d] = pipe
        metrics_all.append(m)
        print(
            f"  val AUROC={m.get('val_auroc'):.3f} AUPRC={m.get('val_auprc'):.3f} | "
            f"test AUROC={t.get('test_auroc')} AUPRC={t.get('test_auprc')}"
        )

    # Hospitalization within 12 months
    print("\nTraining hospitalization_within_12_months ...")
    hosp_target = "hospitalization_within_12_months"
    if hosp_target in train.columns:
        pipe_h, m_h = train_binary_model(train, val, hosp_target)
        t_h = evaluate_test(pipe_h, test, hosp_target)
        m_h.update(t_h)
        models["hospitalization_12m"] = pipe_h
        metrics_all.append(m_h)
        print(
            f"  val AUROC={m_h.get('val_auroc'):.3f} | test AUROC={t_h.get('test_auroc')}"
        )

    # Bundle for Streamlit
    bundle = {
        "models": models,
        "diseases": DISEASES,
        "disease_labels": DISEASE_LABELS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_features": ALL_FEATURES,
        "feature_defaults": _feature_defaults(train),
        "generator_version": "1.2.0",
        "model_version": "1.0.0",
        "trained_on": "data/clean/flat_encounter_level.csv",
        "disclaimer": (
            "SYNTHETIC RESEARCH / DEMO MODEL ONLY. "
            "Not clinically validated. Not for diagnosis or patient care."
        ),
        "metrics": metrics_all,
    }

    out_path = MODEL_DIR / "streamlit_disease_models.joblib"
    joblib.dump(bundle, out_path)
    metrics_path = MODEL_DIR / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "disclaimer": bundle["disclaimer"],
                "model_version": bundle["model_version"],
                "metrics": metrics_all,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print(f"Saved: {out_path}")
    print(f"Metrics: {metrics_path}")
    print("Run Streamlit:  streamlit run streamlit_app.py")
    print("=" * 70)


def _feature_defaults(train: pd.DataFrame) -> dict:
    """Median / mode defaults for Streamlit form prefill."""
    defaults: dict = {}
    for c in NUMERIC_FEATURES:
        if c in train.columns:
            defaults[c] = float(train[c].median()) if train[c].notna().any() else 0.0
    for c in CATEGORICAL_FEATURES:
        if c in train.columns:
            mode = train[c].mode()
            defaults[c] = str(mode.iloc[0]) if len(mode) else ""
    return defaults


if __name__ == "__main__":
    main()
