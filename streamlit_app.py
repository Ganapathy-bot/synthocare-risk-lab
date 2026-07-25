#!/usr/bin/env python3
"""
SynthoCare — fast Streamlit demo for synthetic multi-disease risk models.

NOT FOR CLINICAL USE. Research / education only.
Optimized for Streamlit Community Cloud cold-start and prediction latency.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "streamlit_disease_models.joblib"

st.set_page_config(
    page_title="SynthoCare Risk Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Compact CSS — no external font CDN (faster first paint on Cloud)
APP_CSS = """
<style>
html, body, [class*="css"] { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1100px; }
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b3d4a 0%, #0f5c6e 50%, #147a8a 100%);
}
section[data-testid="stSidebar"] * { color: #f3fbfc !important; }
.hero {
  background: linear-gradient(115deg, #07242d 0%, #0f5c6e 55%, #1a9aa8 100%);
  border-radius: 18px; padding: 1.8rem 2rem; color: #fff;
  box-shadow: 0 14px 36px rgba(11,61,74,.2); margin-bottom: 1rem;
}
.hero h1 { margin: 0 0 .5rem 0; font-size: clamp(1.6rem, 3vw, 2.3rem); line-height: 1.15; }
.hero p { margin: 0; opacity: .92; max-width: 640px; line-height: 1.5; }
.kicker {
  display: inline-block; font-size: .72rem; font-weight: 700; letter-spacing: .08em;
  text-transform: uppercase; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.22);
  padding: .3rem .7rem; border-radius: 999px; margin-bottom: .75rem;
}
.alert {
  background: #fff8ea; border: 1px solid #f0d7a8; border-left: 5px solid #d4a017;
  border-radius: 10px; padding: .8rem 1rem; color: #6a4e12; font-size: .92rem; margin: .6rem 0 1rem 0;
}
.card-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .85rem; margin: .8rem 0 1.2rem; }
@media (max-width: 900px) { .card-grid { grid-template-columns: 1fr; } }
.card {
  background: #fff; border: 1px solid #e4eef0; border-radius: 14px; padding: 1rem 1.1rem;
  box-shadow: 0 6px 18px rgba(11,61,74,.05);
}
.card h3 { margin: 0 0 .35rem 0; color: #0b3d4a; font-size: 1rem; }
.card p { margin: 0; color: #4d646b; font-size: .9rem; line-height: 1.45; }
.stat-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: .65rem; margin: .8rem 0 1.1rem; }
@media (max-width: 900px) { .stat-row { grid-template-columns: 1fr 1fr; } }
.stat { background: #0f5c6e; color: #fff; border-radius: 12px; padding: .85rem 1rem; }
.stat .n { font-size: 1.4rem; font-weight: 700; }
.stat .l { font-size: .78rem; opacity: .9; }
.chip-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0 1rem; }
.chip {
  background: #f2fafb; border: 1px solid #cfe6ea; color: #0f5c6e;
  padding: .35rem .65rem; border-radius: 999px; font-size: .82rem; font-weight: 600;
}
.page-head h1 { color: #0b3d4a; font-size: 1.65rem; margin: 0 0 .25rem 0; }
.page-head p { color: #5a7077; margin: 0 0 .75rem 0; }
</style>
"""


@st.cache_resource(show_spinner="Loading models (first visit only)…")
def load_bundle():
    """Load once per server process — critical for Cloud speed after cold start."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def risk_band(p: float) -> str:
    if p < 0.15:
        return "Lower"
    if p < 0.35:
        return "Moderate"
    if p < 0.55:
        return "Elevated"
    return "Higher"


def build_input_row(bundle: dict, values: dict) -> pd.DataFrame:
    row = {f: values.get(f, bundle["feature_defaults"].get(f, np.nan)) for f in bundle["all_features"]}
    return pd.DataFrame([row])


def predict_all(bundle: dict, X: pd.DataFrame) -> list[dict]:
    """Transform features once, then score all disease heads (fast path)."""
    models = bundle["models"]
    diseases = bundle["diseases"]
    labels = bundle["disease_labels"]

    # New format: shared preprocessor + linear heads
    if bundle.get("format") == "shared_preprocessor_v2" and "preprocessor" in bundle:
        Xt = bundle["preprocessor"].transform(X)
        rows = []
        for d in diseases:
            if d not in models:
                continue
            proba = float(models[d].predict_proba(Xt)[0, 1])
            rows.append({"Condition": labels[d], "Probability": proba, "Risk band": risk_band(proba)})
        return rows

    # Legacy: full sklearn Pipeline per disease
    rows = []
    for d in diseases:
        if d not in models:
            continue
        proba = float(models[d].predict_proba(X)[0, 1])
        rows.append({"Condition": labels[d], "Probability": proba, "Risk band": risk_band(proba)})
    return rows


def main() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🧬 SynthoCare")
        st.caption("Synthetic risk lab · research demo")
        page = st.radio(
            "Navigate",
            ["Home", "Risk prediction", "Model metrics", "About & limits"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("Fast demo · synthetic only · not clinical")

    # Lazy-load models only when needed (Home/About stay light)
    needs_model = page in ("Risk prediction", "Model metrics")
    bundle = None
    if needs_model:
        bundle = load_bundle()
        if bundle is None:
            st.error(
                "Model file missing. On Cloud, ensure `models/streamlit_disease_models.joblib` "
                "is in the GitHub repo. Locally run `python train_models.py`."
            )
            return

    if page == "Home":
        render_home()
    elif page == "Risk prediction":
        render_predict(bundle)
    elif page == "Model metrics":
        render_metrics(bundle)
    else:
        render_about(bundle if bundle else load_bundle())


def render_home() -> None:
    # CSS-only hero (no large base64 image) — much faster first paint
    st.markdown(
        """
<div class="hero">
  <div class="kicker">Synthetic · Research · Education</div>
  <h1>Chronic disease risk models<br/>built on synthetic patients</h1>
  <p>Multi-condition screening demo trained on computer-generated longitudinal data.
  Fast cloud-ready models for pipeline teaching — not for real patient care.</p>
</div>
<div class="alert"><strong>Important:</strong> Outputs are synthetic-model scores only.
They must not be used to diagnose, screen, or manage real patients.</div>
<div class="stat-row">
  <div class="stat"><div class="n">3,000</div><div class="l">Synthetic patients</div></div>
  <div class="stat"><div class="n">10</div><div class="l">Chronic conditions</div></div>
  <div class="stat"><div class="n">11</div><div class="l">Model heads</div></div>
  <div class="stat"><div class="n">70/15/15</div><div class="l">Train / val / test</div></div>
</div>
<div class="card-grid">
  <div class="card"><h3>🎯 Multi-disease demo</h3>
  <p>Score diabetes, CKD, CAD, HF, hypertension, COPD, liver disease, RA, thyroid, cognition.</p></div>
  <div class="card"><h3>⚡ Cloud-optimized</h3>
  <p>Shared preprocessor + light linear models for quick cold starts and instant predictions.</p></div>
  <div class="card"><h3>📊 Transparent metrics</h3>
  <p>Hold-out AUROC/AUPRC on synthetic partitions — not clinical proof.</p></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("#### Conditions in scope")
    chips = [
        "Type 2 diabetes",
        "CKD",
        "CAD",
        "Heart failure",
        "Hypertension",
        "COPD",
        "CLD / MASLD",
        "RA",
        "Hypothyroidism",
        "Cognitive decline",
    ]
    st.markdown(
        '<div class="chip-row">' + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div>",
        unsafe_allow_html=True,
    )
    st.info("Open **Risk prediction** in the sidebar to run a scenario.")


def render_predict(bundle: dict) -> None:
    st.markdown(
        """
<div class="page-head">
  <h1>Risk prediction</h1>
  <p>Enter a synthetic encounter scenario. Leave labs blank if unknown.</p>
</div>
<div class="alert"><strong>Demo only:</strong> Probabilities are not clinical diagnoses.</div>
""",
        unsafe_allow_html=True,
    )
    defaults = bundle["feature_defaults"]

    with st.expander("👤 Demographics & body measures", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", 18, 95, int(defaults.get("age_at_encounter", 55)))
            sex = st.selectbox(
                "Sex at birth",
                ["female", "male"],
                index=0 if defaults.get("sex_at_birth") == "female" else 1,
            )
            bmi = st.number_input("BMI", 15.0, 55.0, float(defaults.get("bmi", 27.5)), 0.1)
        with c2:
            waist = st.number_input(
                "Waist (cm)", 50.0, 160.0, float(defaults.get("waist_cm_baseline", 90.0)), 0.5
            )
            height = st.number_input(
                "Height (cm)", 140.0, 210.0, float(defaults.get("height_cm", 168.0)), 0.5
            )
            weight = st.number_input(
                "Weight (kg)", 35.0, 200.0, float(defaults.get("weight_kg", 75.0)), 0.5
            )
        with c3:
            ethnicity = st.selectbox(
                "Ethnicity (optional)",
                [
                    "White",
                    "Black_or_African_American",
                    "Hispanic_or_Latino",
                    "Asian",
                    "Other_or_Multiple",
                ],
            )
            ses = st.selectbox("SES group", ["low", "medium", "high"], index=1)
            access = st.selectbox("Access to care", ["limited", "standard", "enhanced"], index=1)

    with st.expander("🚬 Lifestyle", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            smoking = st.selectbox("Smoking", ["never", "former", "current"])
        with c2:
            pack_years = st.number_input(
                "Pack-years", 0.0, 100.0, float(defaults.get("pack_years", 0.0)), 0.5
            )
        with c3:
            alcohol = st.selectbox("Alcohol use", ["none", "moderate", "heavy"])
        with c4:
            activity = st.selectbox("Physical activity", ["sedentary", "moderate", "active"])
        diet = st.selectbox("Diet risk", ["low", "moderate", "high"], index=1)

    with st.expander("❤️ Vitals", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sbp = st.number_input("Systolic BP", 80, 230, int(defaults.get("sbp_mmHg", 128)))
        with c2:
            dbp = st.number_input("Diastolic BP", 40, 140, int(defaults.get("dbp_mmHg", 78)))
        with c3:
            hr = st.number_input("Heart rate", 40, 180, int(defaults.get("heart_rate_bpm", 72)))
        with c4:
            spo2 = st.number_input("SpO₂ %", 70, 100, int(defaults.get("spo2_percent", 98)))

    with st.expander("🩺 Symptoms", expanded=False):
        sc = st.columns(4)
        symptoms = {}
        for i, (key, label) in enumerate(
            [
                ("symptom_polyuria", "Polyuria"),
                ("symptom_polydipsia", "Polydipsia"),
                ("symptom_fatigue", "Fatigue"),
                ("symptom_chest_pain", "Chest pain"),
                ("symptom_dyspnea", "Dyspnea"),
                ("symptom_edema", "Edema"),
                ("symptom_cough", "Cough"),
                ("symptom_joint_pain", "Joint pain"),
                ("symptom_joint_swelling", "Joint swelling"),
                ("symptom_cognitive_complaint", "Cognitive complaint"),
                ("symptom_cold_intolerance", "Cold intolerance"),
            ]
        ):
            with sc[i % 4]:
                symptoms[key] = int(st.checkbox(label, value=False, key=key))

    with st.expander("⚡ Acute context", expanded=False):
        ac = st.columns(5)
        acute = {}
        for i, (key, label) in enumerate(
            [
                ("acute_infection", "Acute infection"),
                ("dehydration", "Dehydration"),
                ("strenuous_exercise", "Strenuous exercise"),
                ("recent_surgery", "Recent surgery"),
                ("recent_hospitalization", "Recent hospitalization"),
            ]
        ):
            with ac[i]:
                acute[key] = int(st.checkbox(label, value=False, key=key))
        acute["acute_illness"] = int(
            acute["acute_infection"] or acute["dehydration"] or acute["recent_surgery"]
        )

    with st.expander("👪 Family history", expanded=False):
        fh = {}
        fhc = st.columns(5)
        for i, d in enumerate(bundle["diseases"]):
            with fhc[i % 5]:
                fh[f"fh_{d}"] = int(
                    st.checkbox(bundle["disease_labels"][d], value=False, key=f"fh_{d}")
                )

    with st.expander("🔬 Labs (optional)", expanded=False):
        st.caption("Leave unchecked to treat as missing.")
        lab_defs = [
            ("lab_hba1c", "HbA1c (%)", 4.0, 15.0),
            ("lab_fasting_plasma_glucose", "Fasting glucose", 50.0, 400.0),
            ("lab_egfr", "eGFR", 5.0, 140.0),
            ("lab_uacr", "UACR", 0.0, 3500.0),
            ("lab_ldl_c", "LDL-C", 30.0, 300.0),
            ("lab_hdl_c", "HDL-C", 15.0, 120.0),
            ("lab_triglycerides", "Triglycerides", 30.0, 800.0),
            ("lab_alt", "ALT", 5.0, 400.0),
            ("lab_ast", "AST", 5.0, 400.0),
            ("lab_tsh", "TSH", 0.01, 80.0),
            ("lab_nt_probnp", "NT-proBNP", 5.0, 30000.0),
            ("lab_hemoglobin", "Hemoglobin", 6.0, 20.0),
            ("lab_crp", "CRP", 0.1, 100.0),
            ("lab_anti_ccp", "Anti-CCP", 0.0, 400.0),
            ("lab_fev1_fvc_ratio", "FEV1/FVC", 0.2, 1.0),
            ("lab_moca", "MoCA", 0.0, 30.0),
        ]
        labs = {}
        lc = st.columns(4)
        for i, (key, label, lo, hi) in enumerate(lab_defs):
            with lc[i % 4]:
                use = st.checkbox(f"Provide {label}", value=False, key=f"use_{key}")
                if use:
                    default = float(np.clip(defaults.get(key, (lo + hi) / 2), lo, hi))
                    labs[key] = st.number_input(label, lo, hi, default, key=f"val_{key}")
                else:
                    labs[key] = np.nan

    if not st.button("Run risk estimates", type="primary", use_container_width=True):
        return

    values = {
        "age_at_encounter": age,
        "sex_at_birth": sex,
        "bmi": bmi,
        "waist_cm_baseline": waist,
        "height_cm": height,
        "weight_kg": weight,
        "smoking_status": smoking,
        "pack_years": pack_years,
        "alcohol_use": alcohol,
        "physical_activity": activity,
        "diet_risk": diet,
        "ethnicity": ethnicity,
        "ses_group": ses,
        "access_to_care": access,
        "sbp_mmHg": sbp,
        "dbp_mmHg": dbp,
        "heart_rate_bpm": hr,
        "spo2_percent": spo2,
        **symptoms,
        **acute,
        **fh,
        **labs,
    }
    X = build_input_row(bundle, values)

    with st.spinner("Scoring…"):
        rows = predict_all(bundle, X)

    result = pd.DataFrame(rows).sort_values("Probability", ascending=False)
    display = result.copy()
    display["Probability"] = display["Probability"].map(lambda x: f"{x:.1%}")

    left, right = st.columns([1.1, 1])
    with left:
        st.dataframe(
            display[["Condition", "Probability", "Risk band"]],
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.bar_chart(result.set_index("Condition")["Probability"], color="#0f5c6e")

    models = bundle["models"]
    if "hospitalization_12m" in models:
        if bundle.get("format") == "shared_preprocessor_v2":
            Xt = bundle["preprocessor"].transform(X)
            ph = float(models["hospitalization_12m"].predict_proba(Xt)[0, 1])
        else:
            ph = float(models["hospitalization_12m"].predict_proba(X)[0, 1])
        st.metric("Synthetic 12-month hospitalization proxy", f"{ph:.1%}")

    top = result.iloc[0]
    st.success(
        f"Highest score: **{top['Condition']}** at **{top['Probability']:.1%}** "
        f"({top['Risk band']}) — demo only."
    )


def render_metrics(bundle: dict) -> None:
    st.markdown(
        """
<div class="page-head">
  <h1>Model metrics</h1>
  <p>Synthetic hold-out performance (not clinical validation).</p>
</div>
""",
        unsafe_allow_html=True,
    )
    metrics = bundle.get("metrics", [])
    if not metrics:
        st.write("No metrics stored.")
        return
    mdf = pd.DataFrame(metrics)
    show = [
        c
        for c in [
            "target",
            "n_train",
            "n_val",
            "n_test",
            "val_auroc",
            "val_auprc",
            "test_auroc",
            "test_auprc",
        ]
        if c in mdf.columns
    ]
    st.dataframe(mdf[show], use_container_width=True, hide_index=True)
    if "test_auroc" in mdf.columns:
        st.bar_chart(mdf.dropna(subset=["test_auroc"]).set_index("target")["test_auroc"], color="#147a8a")


def render_about(bundle) -> None:
    st.markdown(
        """
<div class="page-head">
  <h1>About & limits</h1>
  <p>Intended use and non-clinical disclaimer.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ✅ Intended uses\n- Research & education\n- Pipeline demos\n- UI prototyping")
    with c2:
        st.markdown("### 🚫 Prohibited\n- Real diagnosis\n- Clinical decisions\n- Regulatory claims")
    st.markdown(
        """
### Performance tips
- First open after idle can take longer (Streamlit Cloud **cold start**).
- Home page no longer loads the model bundle.
- Predictions use one shared transform + light linear heads.
"""
    )
    if bundle:
        st.code(bundle.get("disclaimer", ""), language=None)
        st.caption(
            f"Model v{bundle.get('model_version')} · format {bundle.get('format')} · "
            f"generator {bundle.get('generator_version')}"
        )


if __name__ == "__main__":
    main()
