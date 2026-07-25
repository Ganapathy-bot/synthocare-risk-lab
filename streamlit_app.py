#!/usr/bin/env python3
"""
Streamlit demo: multi-disease risk screening using models trained on
SYNTHETIC data only.

NOT FOR CLINICAL USE. Research / education / pipeline demonstration.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "streamlit_disease_models.joblib"
HERO_PATH = ROOT / "assets" / "hero_banner.jpg"

st.set_page_config(
    page_title="SynthoCare Risk Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

html, body, [class*="css"] {
  font-family: 'DM Sans', system-ui, -apple-system, sans-serif;
}

/* Hide default streamlit chrome bits slightly */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container {
  padding-top: 1.2rem;
  padding-bottom: 3rem;
  max-width: 1180px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b3d4a 0%, #0f5c6e 45%, #147a8a 100%);
}
section[data-testid="stSidebar"] * {
  color: #f3fbfc !important;
}
section[data-testid="stSidebar"] .stRadio label {
  font-weight: 500;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  color: #d7f0f3 !important;
}

/* Hero */
.hero-wrap {
  position: relative;
  border-radius: 22px;
  overflow: hidden;
  margin-bottom: 1.5rem;
  box-shadow: 0 18px 50px rgba(11, 61, 74, 0.18);
  border: 1px solid rgba(15, 92, 110, 0.12);
}
.hero-bg {
  width: 100%;
  height: 320px;
  object-fit: cover;
  display: block;
  filter: saturate(1.05) brightness(0.92);
}
.hero-overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(105deg,
    rgba(7, 36, 45, 0.88) 0%,
    rgba(11, 61, 74, 0.72) 42%,
    rgba(20, 122, 138, 0.35) 100%);
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 2.4rem 2.8rem;
}
.hero-kicker {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.22);
  color: #e8fbff;
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  width: fit-content;
  margin-bottom: 0.9rem;
}
.hero-title {
  font-family: 'Fraunces', Georgia, serif;
  font-size: clamp(1.9rem, 3.2vw, 2.75rem);
  line-height: 1.12;
  color: #ffffff;
  margin: 0 0 0.75rem 0;
  max-width: 720px;
  font-weight: 700;
}
.hero-sub {
  color: #d7eef2;
  font-size: 1.05rem;
  line-height: 1.55;
  max-width: 640px;
  margin: 0 0 1.25rem 0;
}
.hero-cta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  align-items: center;
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.45rem 0.85rem;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 600;
}
.pill-light {
  background: rgba(255,255,255,0.14);
  color: #fff;
  border: 1px solid rgba(255,255,255,0.22);
}
.pill-warn {
  background: rgba(255, 196, 92, 0.18);
  color: #ffe7b0;
  border: 1px solid rgba(255, 210, 120, 0.35);
}

/* Cards */
.card-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin: 0.5rem 0 1.5rem 0;
}
@media (max-width: 900px) {
  .card-grid { grid-template-columns: 1fr; }
  .hero-bg { height: 420px; }
  .hero-overlay { padding: 1.5rem; }
}
.card {
  background: #ffffff;
  border: 1px solid #e4eef0;
  border-radius: 16px;
  padding: 1.2rem 1.25rem;
  box-shadow: 0 8px 24px rgba(11, 61, 74, 0.05);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 14px 32px rgba(11, 61, 74, 0.10);
}
.card-icon {
  width: 42px;
  height: 42px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.25rem;
  margin-bottom: 0.75rem;
  background: linear-gradient(135deg, #e7f7f9, #d4eef3);
}
.card h3 {
  font-size: 1.05rem;
  margin: 0 0 0.4rem 0;
  color: #0b3d4a;
  font-weight: 700;
}
.card p {
  margin: 0;
  color: #4d646b;
  font-size: 0.92rem;
  line-height: 1.5;
}

.section-title {
  font-family: 'Fraunces', Georgia, serif;
  color: #0b3d4a;
  font-size: 1.55rem;
  margin: 1.4rem 0 0.35rem 0;
}
.section-lead {
  color: #5a7077;
  margin: 0 0 1rem 0;
  font-size: 0.98rem;
}

/* Disease chips */
.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin: 0.75rem 0 1.5rem 0;
}
.chip {
  background: #f2fafb;
  border: 1px solid #cfe6ea;
  color: #0f5c6e;
  padding: 0.4rem 0.75rem;
  border-radius: 999px;
  font-size: 0.84rem;
  font-weight: 600;
}

/* Steps */
.steps {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.85rem;
  margin-bottom: 1.5rem;
}
@media (max-width: 900px) {
  .steps { grid-template-columns: 1fr 1fr; }
}
.step {
  background: linear-gradient(180deg, #ffffff, #f7fcfd);
  border: 1px solid #e0eef1;
  border-radius: 14px;
  padding: 1rem;
}
.step-num {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: #0f5c6e;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.8rem;
  font-weight: 700;
  margin-bottom: 0.55rem;
}
.step h4 {
  margin: 0 0 0.3rem 0;
  color: #0b3d4a;
  font-size: 0.95rem;
}
.step p {
  margin: 0;
  color: #5a7077;
  font-size: 0.85rem;
  line-height: 1.45;
}

/* Banner alert */
.alert-banner {
  background: linear-gradient(90deg, #fff6e8, #fffaf2);
  border: 1px solid #f0d7a8;
  border-left: 5px solid #d4a017;
  border-radius: 12px;
  padding: 0.9rem 1.1rem;
  color: #6a4e12;
  font-size: 0.92rem;
  line-height: 1.5;
  margin: 0.5rem 0 1.2rem 0;
}
.alert-banner strong { color: #5a3f08; }

.stat-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.75rem;
  margin: 1rem 0 1.5rem 0;
}
@media (max-width: 900px) {
  .stat-row { grid-template-columns: 1fr 1fr; }
}
.stat {
  background: #0f5c6e;
  color: white;
  border-radius: 14px;
  padding: 1rem 1.1rem;
}
.stat .n {
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.55rem;
  font-weight: 700;
  line-height: 1.1;
}
.stat .l {
  font-size: 0.8rem;
  opacity: 0.88;
  margin-top: 0.2rem;
}

.footer-note {
  margin-top: 2rem;
  padding-top: 1rem;
  border-top: 1px solid #e4eef0;
  color: #7a9096;
  font-size: 0.82rem;
  line-height: 1.5;
}

/* Predict page */
.page-head {
  margin-bottom: 1rem;
}
.page-head h1 {
  font-family: 'Fraunces', Georgia, serif;
  color: #0b3d4a;
  font-size: 1.8rem;
  margin: 0 0 0.3rem 0;
}
.page-head p {
  color: #5a7077;
  margin: 0;
}
</style>
"""


@st.cache_resource
def load_bundle():
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
    row = {}
    for f in bundle["all_features"]:
        row[f] = values.get(f, bundle["feature_defaults"].get(f, np.nan))
    return pd.DataFrame([row])


def inject_css() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def sidebar_nav() -> str:
    with st.sidebar:
        st.markdown("### 🧬 SynthoCare")
        st.caption("Synthetic risk lab · research demo")
        page = st.radio(
            "Navigate",
            ["Home", "Risk prediction", "Model metrics", "About & limits"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(
            """
**Status**  
✅ Dataset ready  
✅ Models loaded  
⚠️ Not clinical
"""
        )
        st.markdown("---")
        st.caption("v1.0 demo · seed 42 · synthetic only")
    return page


def render_home(bundle) -> None:
    # Hero
    if HERO_PATH.exists():
        import base64

        b64 = base64.b64encode(HERO_PATH.read_bytes()).decode()
        hero_img = f"data:image/jpeg;base64,{b64}"
    else:
        hero_img = (
            "data:image/svg+xml," 
            + "%3Csvg xmlns='http://www.w3.org/2000/svg' width='1600' height='400'%3E"
            + "%3Cdefs%3E%3ClinearGradient id='g' x1='0' y1='0' x2='1' y2='1'%3E"
            + "%3Cstop stop-color='%230b3d4a'/%3E%3Cstop offset='1' stop-color='%23147a8a'/%3E"
            + "%3C/linearGradient%3E%3C/defs%3E"
            + "%3Crect width='1600' height='400' fill='url(%23g)'/%3E%3C/svg%3E"
        )

    st.markdown(
        f"""
<div class="hero-wrap">
  <img class="hero-bg" src="{hero_img}" alt="Research banner" />
  <div class="hero-overlay">
    <div class="hero-kicker">🧪 Synthetic · Research · Education</div>
    <h1 class="hero-title">Chronic disease risk models<br/>built on synthetic patients</h1>
    <p class="hero-sub">
      Explore multi-condition screening models trained on a latent-state synthetic cohort.
      Design pipelines, compare signals, and prototype interfaces — without real patient data.
    </p>
    <div class="hero-cta-row">
      <span class="pill pill-light">10 disease heads</span>
      <span class="pill pill-light">Patient-level train / val / test</span>
      <span class="pill pill-warn">Not for clinical use</span>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="alert-banner">
  <strong>Important:</strong> Every prediction here is produced by models trained only on
  <em>computer-generated</em> records. Outputs are for methodology demos and teaching —
  they must not be used to diagnose, screen, or manage real patients.
</div>
""",
        unsafe_allow_html=True,
    )

    # Stats
    n_models = len(bundle["models"]) if bundle else 0
    n_diseases = len(bundle["diseases"]) if bundle else 10
    st.markdown(
        f"""
<div class="stat-row">
  <div class="stat"><div class="n">3,000</div><div class="l">Synthetic patients</div></div>
  <div class="stat"><div class="n">{n_diseases}</div><div class="l">Chronic conditions</div></div>
  <div class="stat"><div class="n">{n_models}</div><div class="l">Trained model heads</div></div>
  <div class="stat"><div class="n">70/15/15</div><div class="l">Train / val / test split</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<h2 class="section-title">What you can do</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-lead">A full demo stack from synthetic cohort → trained models → interactive UI.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="card-grid">
  <div class="card">
    <div class="card-icon">🎯</div>
    <h3>Multi-disease screening demo</h3>
    <p>Estimate synthetic risk scores for diabetes, CKD, CAD, HF, hypertension, COPD, liver disease, RA, thyroid, and cognitive decline.</p>
  </div>
  <div class="card">
    <div class="card-icon">🧪</div>
    <h3>Missing-lab aware inputs</h3>
    <p>Optional biomarkers with realistic missingness handling — leave labs blank when unknown, just like sparse EHR panels.</p>
  </div>
  <div class="card">
    <div class="card-icon">📊</div>
    <h3>Transparent hold-out metrics</h3>
    <p>Inspect AUROC / AUPRC on synthetic validation and test partitions. Strong scores here are not clinical proof.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<h2 class="section-title">Conditions in scope</h2>', unsafe_allow_html=True)
    labels = (
        bundle["disease_labels"]
        if bundle
        else {
            "t2dm": "Type 2 diabetes",
            "ckd": "Chronic kidney disease",
            "cad": "Coronary artery disease",
            "hf": "Heart failure",
            "htn": "Hypertension",
            "copd": "COPD",
            "cld_masld": "CLD / MASLD",
            "ra": "Rheumatoid arthritis",
            "hypothyroid": "Hypothyroidism",
            "alzheimers": "Cognitive decline",
        }
    )
    chips = "".join(f'<span class="chip">{lab}</span>' for lab in labels.values())
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">How the pipeline works</h2>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="steps">
  <div class="step">
    <div class="step-num">1</div>
    <h4>Generate</h4>
    <p>Latent-state simulator creates longitudinal synthetic patients, labs, and labels.</p>
  </div>
  <div class="step">
    <div class="step-num">2</div>
    <h4>Split</h4>
    <p>Patient-level 70/15/15 partitions prevent the same person leaking across sets.</p>
  </div>
  <div class="step">
    <div class="step-num">3</div>
    <h4>Train</h4>
    <p>Gradient-boosted classifiers learn disease-presence heads from encounter features.</p>
  </div>
  <div class="step">
    <div class="step-num">4</div>
    <h4>Interact</h4>
    <p>Use this app to explore scores for scenario inputs — research demo only.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<h2 class="section-title">Start exploring</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-lead">Open <strong>Risk prediction</strong> in the sidebar to enter a scenario and run the models.</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.info("**Risk prediction**\n\nEnter vitals, symptoms & labs")
    with c2:
        st.info("**Model metrics**\n\nReview synthetic hold-out scores")
    with c3:
        st.info("**About & limits**\n\nIntended / prohibited uses")

    st.markdown(
        """
<div class="footer-note">
  SynthoCare Risk Lab · SyntheticChronicDiseaseGenerator · Models are not medical devices.
  External clinical validation is required before any real-world interpretation.
</div>
""",
        unsafe_allow_html=True,
    )


def render_predict(bundle) -> None:
    st.markdown(
        """
<div class="page-head">
  <h1>Risk prediction</h1>
  <p>Scenario-style inputs for a single encounter. Optional labs may be left blank.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="alert-banner">
  <strong>Demo only:</strong> Probabilities reflect synthetic training labels, not clinical diagnoses.
</div>
""",
        unsafe_allow_html=True,
    )

    if bundle is None:
        st.error("Model bundle not found. Run `python train_models.py` first.")
        return

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
                "Ethnicity (audit / optional)",
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
        symptom_labels = [
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
        for i, (key, label) in enumerate(symptom_labels):
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

    with st.expander("🔬 Laboratory values (optional)", expanded=False):
        st.caption("Uncheck to leave missing — models were trained with incomplete panels.")
        lab_defs = [
            ("lab_hba1c", "HbA1c (%)", 4.0, 15.0),
            ("lab_fasting_plasma_glucose", "Fasting glucose (mg/dL)", 50.0, 400.0),
            ("lab_egfr", "eGFR", 5.0, 140.0),
            ("lab_uacr", "UACR (mg/g)", 0.0, 3500.0),
            ("lab_ldl_c", "LDL-C (mg/dL)", 30.0, 300.0),
            ("lab_hdl_c", "HDL-C (mg/dL)", 15.0, 120.0),
            ("lab_triglycerides", "Triglycerides (mg/dL)", 30.0, 800.0),
            ("lab_alt", "ALT (U/L)", 5.0, 400.0),
            ("lab_ast", "AST (U/L)", 5.0, 400.0),
            ("lab_tsh", "TSH (mIU/L)", 0.01, 80.0),
            ("lab_nt_probnp", "NT-proBNP (pg/mL)", 5.0, 30000.0),
            ("lab_hemoglobin", "Hemoglobin (g/dL)", 6.0, 20.0),
            ("lab_crp", "CRP (mg/L)", 0.1, 100.0),
            ("lab_anti_ccp", "Anti-CCP (U/mL)", 0.0, 400.0),
            ("lab_fev1_fvc_ratio", "FEV1/FVC", 0.2, 1.0),
            ("lab_moca", "MoCA score", 0.0, 30.0),
        ]
        labs = {}
        lc = st.columns(4)
        for i, (key, label, lo, hi) in enumerate(lab_defs):
            with lc[i % 4]:
                use = st.checkbox(f"Provide {label}", value=False, key=f"use_{key}")
                if use:
                    default = float(defaults.get(key, (lo + hi) / 2))
                    default = float(np.clip(default, lo, hi))
                    labs[key] = st.number_input(label, lo, hi, default, key=f"val_{key}")
                else:
                    labs[key] = np.nan

    run = st.button("Run risk estimates", type="primary", use_container_width=True)

    if not run:
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

    st.markdown("### Estimated probabilities")
    st.caption("Synthetic model outputs only — not a clinical diagnosis.")

    rows = []
    models = bundle["models"]
    for d in bundle["diseases"]:
        proba = float(models[d].predict_proba(X)[0, 1])
        rows.append(
            {
                "Condition": bundle["disease_labels"][d],
                "Probability": proba,
                "Risk band": risk_band(proba),
            }
        )

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
        chart_df = result.set_index("Condition")["Probability"]
        st.bar_chart(chart_df, color="#0f5c6e")

    if "hospitalization_12m" in models:
        ph = float(models["hospitalization_12m"].predict_proba(X)[0, 1])
        st.metric("Synthetic 12-month hospitalization proxy", f"{ph:.1%}")

    top = result.iloc[0]
    st.success(
        f"Highest score: **{top['Condition']}** at **{top['Probability']:.1%}** "
        f"({top['Risk band']} band) — demo interpretation only."
    )

    with st.expander("Input vector (debug)"):
        st.dataframe(X.T.rename(columns={0: "value"}), use_container_width=True)


def render_metrics(bundle) -> None:
    st.markdown(
        """
<div class="page-head">
  <h1>Model metrics</h1>
  <p>Hold-out performance on the synthetic validation and test partitions.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="alert-banner">
  <strong>Read carefully:</strong> High AUROC on synthetic data does not establish clinical usefulness,
  calibration in real care, or fairness in real populations.
</div>
""",
        unsafe_allow_html=True,
    )
    if bundle is None:
        st.error("Model bundle not found.")
        return

    metrics = bundle.get("metrics", [])
    if not metrics:
        st.write("No metrics stored.")
        return

    mdf = pd.DataFrame(metrics)
    show_cols = [
        c
        for c in [
            "target",
            "n_train",
            "n_val",
            "n_test",
            "train_prevalence",
            "val_auroc",
            "val_auprc",
            "test_auroc",
            "test_auprc",
            "test_brier",
        ]
        if c in mdf.columns
    ]
    st.dataframe(mdf[show_cols], use_container_width=True, hide_index=True)

    if "test_auroc" in mdf.columns:
        plot_df = mdf.dropna(subset=["test_auroc"]).set_index("target")["test_auroc"]
        st.markdown("#### Test AUROC by target")
        st.bar_chart(plot_df, color="#147a8a")


def render_about(bundle) -> None:
    st.markdown(
        """
<div class="page-head">
  <h1>About & limits</h1>
  <p>Intended use, prohibited use, and methodological notes.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ✅ Intended uses")
        st.markdown(
            """
- Research and education demos  
- ML pipeline development  
- UI / workflow prototyping  
- Teaching leakage, missingness, multimorbidity  
"""
        )
    with c2:
        st.markdown("### 🚫 Prohibited uses")
        st.markdown(
            """
- Real patient diagnosis or triage  
- Clinical decision support in care  
- Claiming population epidemiology  
- Regulatory performance claims  
"""
        )

    st.markdown("### Method (short)")
    st.markdown(
        """
1. Synthetic longitudinal cohort (`generate_dataset.py`)  
2. Patient-level train / val / test split  
3. One `HistGradientBoostingClassifier` per disease for `disease_present_at_encounter_*`  
4. Features: demographics, lifestyle, vitals, symptoms, family history, optional labs  
5. No latent severity / future outcomes used as features  
"""
    )

    st.markdown("### External validation required")
    st.write(
        "Any real-world use requires validation on representative clinical data, "
        "calibration assessment, fairness review, and appropriate governance."
    )

    if bundle:
        st.code(bundle.get("disclaimer", ""), language=None)
        st.caption(
            f"Model v{bundle.get('model_version')} · generator {bundle.get('generator_version')} · "
            f"{bundle.get('trained_on')}"
        )


def main() -> None:
    inject_css()
    page = sidebar_nav()
    bundle = load_bundle()

    if bundle is None and page != "Home":
        st.warning(
            f"Model file not found at `{MODEL_PATH}`.\n\n"
            "Run:\n"
            "```\n"
            "python generate_dataset.py --n-patients 3000 --seed 42\n"
            "python train_models.py\n"
            "```"
        )

    if page == "Home":
        render_home(bundle)
    elif page == "Risk prediction":
        render_predict(bundle)
    elif page == "Model metrics":
        render_metrics(bundle)
    else:
        render_about(bundle)


if __name__ == "__main__":
    main()
