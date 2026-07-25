#!/usr/bin/env python3
"""
SynthoCare — Streamlit demo optimized for Community Cloud cold starts.

Runtime deps: streamlit + numpy only (no scikit-learn / joblib / pandas).
Models loaded from models/fast_inference.json (pure logistic coefficients).

NOT FOR CLINICAL USE — synthetic research demo only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
FAST_MODEL = ROOT / "models" / "fast_inference.json"

st.set_page_config(
    page_title="SynthoCare Risk Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
  text-transform: uppercase; background: rgba(255,255,255,.14);
  border: 1px solid rgba(255,255,255,.22); padding: .3rem .7rem; border-radius: 999px; margin-bottom: .75rem;
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


@st.cache_resource(show_spinner=False)
def load_fast_model() -> dict | None:
    if not FAST_MODEL.exists():
        return None
    with FAST_MODEL.open(encoding="utf-8") as f:
        data = json.load(f)
    # Pre-convert arrays once for speed
    data["_num_fill"] = np.asarray(data["num_fill"], dtype=np.float64)
    data["_mean"] = np.asarray(data["scaler_mean"], dtype=np.float64)
    data["_scale"] = np.asarray(data["scaler_scale"], dtype=np.float64)
    data["_heads_np"] = {
        k: {
            "coef": np.asarray(v["coef"], dtype=np.float64),
            "intercept": float(v["intercept"]),
        }
        for k, v in data["heads"].items()
    }
    return data


def risk_band(p: float) -> str:
    if p < 0.15:
        return "Lower"
    if p < 0.35:
        return "Moderate"
    if p < 0.55:
        return "Elevated"
    return "Higher"


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = np.exp(-z)
        return float(1.0 / (1.0 + ez))
    ez = np.exp(z)
    return float(ez / (1.0 + ez))


def transform_features(model: dict, values: dict) -> np.ndarray:
    """Median-impute + scale numerics, one-hot categoricals — mirrors training preprocessor."""
    num_feats = model["numeric_features"]
    cat_feats = model["categorical_features"]
    fill = model["_num_fill"]
    mean = model["_mean"]
    scale = model["_scale"]
    categories = model["categories"]

    x_num = np.empty(len(num_feats), dtype=np.float64)
    for i, f in enumerate(num_feats):
        v = values.get(f, np.nan)
        try:
            fv = float(v)
            if np.isnan(fv):
                fv = fill[i]
        except (TypeError, ValueError):
            fv = fill[i]
        x_num[i] = (fv - mean[i]) / scale[i]

    cat_blocks = []
    for j, f in enumerate(cat_feats):
        cats = categories[j]
        raw = values.get(f, model["feature_defaults"].get(f, cats[0] if cats else ""))
        s = str(raw)
        one = np.zeros(len(cats), dtype=np.float64)
        if s in cats:
            one[cats.index(s)] = 1.0
        elif cats:
            # unknown → all zeros (OneHotEncoder handle_unknown=ignore)
            pass
        cat_blocks.append(one)

    if cat_blocks:
        return np.concatenate([x_num] + cat_blocks)
    return x_num


def predict_all(model: dict, values: dict) -> list[dict]:
    x = transform_features(model, values)
    rows = []
    labels = model["disease_labels"]
    for d in model["diseases"]:
        head = model["_heads_np"].get(d)
        if head is None:
            continue
        z = float(np.dot(x, head["coef"]) + head["intercept"])
        p = _sigmoid(z)
        rows.append({"Condition": labels[d], "Probability": p, "Risk band": risk_band(p)})
    rows.sort(key=lambda r: r["Probability"], reverse=True)
    return rows


def predict_one(model: dict, values: dict, key: str) -> float | None:
    head = model["_heads_np"].get(key)
    if head is None:
        return None
    x = transform_features(model, values)
    z = float(np.dot(x, head["coef"]) + head["intercept"])
    return _sigmoid(z)


def main() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🧬 SynthoCare")
        st.caption("Fast synthetic risk demo")
        page = st.radio(
            "Navigate",
            ["Home", "Risk prediction", "Model metrics", "About & limits"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("numpy inference · free Cloud may throttle CPU · not clinical")

    # Home never loads the model file
    if page == "Home":
        render_home()
        return

    model = load_fast_model()
    if model is None:
        st.error(
            f"Missing `{FAST_MODEL.name}`. "
            "Rebuild with: `python train_models.py && python export_fast_model.py`"
        )
        return

    if page == "Risk prediction":
        render_predict(model)
    elif page == "Model metrics":
        render_metrics(model)
    else:
        render_about(model)


def render_home() -> None:
    st.markdown(
        """
<div class="hero">
  <div class="kicker">Synthetic · Research · Education · Fast Cloud</div>
  <h1>Chronic disease risk models<br/>built on synthetic patients</h1>
  <p>Lightweight demo: pure NumPy inference for quick Streamlit Cloud starts.
  Not for real patient care.</p>
</div>
<div class="alert"><strong>Important:</strong> Synthetic-model scores only —
do not use for diagnosis or clinical decisions.</div>
<div class="stat-row">
  <div class="stat"><div class="n">3,000</div><div class="l">Synthetic patients</div></div>
  <div class="stat"><div class="n">10</div><div class="l">Chronic conditions</div></div>
  <div class="stat"><div class="n">JSON</div><div class="l">Tiny model file</div></div>
  <div class="stat"><div class="n">Fast</div><div class="l">No sklearn at runtime</div></div>
</div>
<div class="card-grid">
  <div class="card"><h3>🎯 Multi-disease demo</h3>
  <p>Score 10 chronic conditions from encounter-style inputs.</p></div>
  <div class="card"><h3>⚡ Cloud-optimized</h3>
  <p>Only Streamlit + NumPy. Home page loads without the model.</p></div>
  <div class="card"><h3>📊 Transparent metrics</h3>
  <p>Synthetic hold-out scores — not clinical validation.</p></div>
</div>
""",
        unsafe_allow_html=True,
    )
    chips = [
        "Type 2 diabetes", "CKD", "CAD", "Heart failure", "Hypertension",
        "COPD", "CLD / MASLD", "RA", "Hypothyroidism", "Cognitive decline",
    ]
    st.markdown(
        '<div class="chip-row">' + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div>",
        unsafe_allow_html=True,
    )
    st.info("Open **Risk prediction** in the sidebar to run a scenario.")
    st.caption(
        "If Streamlit shows “app has been throttled”, free Community Cloud has temporarily "
        "reduced CPU. Wait until the expiry time, use the app lightly, or run locally."
    )


def render_predict(model: dict) -> None:
    st.markdown(
        """
<div class="page-head">
  <h1>Risk prediction</h1>
  <p>Synthetic encounter scenario. Use the form — scoring runs only when you submit (saves Cloud CPU).</p>
</div>
<div class="alert"><strong>Demo only:</strong> Not a clinical diagnosis.</div>
""",
        unsafe_allow_html=True,
    )
    defaults = model["feature_defaults"]

    # st.form batches widgets so each click does NOT re-run the whole app (reduces throttling)
    with st.form("risk_form", clear_on_submit=False):
        st.markdown("##### Demographics & body")
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

        st.markdown("##### Lifestyle & vitals")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            smoking = st.selectbox("Smoking", ["never", "former", "current"])
            pack_years = st.number_input(
                "Pack-years", 0.0, 100.0, float(defaults.get("pack_years", 0.0)), 0.5
            )
        with c2:
            alcohol = st.selectbox("Alcohol use", ["none", "moderate", "heavy"])
            activity = st.selectbox("Physical activity", ["sedentary", "moderate", "active"])
        with c3:
            diet = st.selectbox("Diet risk", ["low", "moderate", "high"], index=1)
            sbp = st.number_input("Systolic BP", 80, 230, int(defaults.get("sbp_mmHg", 128)))
        with c4:
            dbp = st.number_input("Diastolic BP", 40, 140, int(defaults.get("dbp_mmHg", 78)))
            hr = st.number_input("Heart rate", 40, 180, int(defaults.get("heart_rate_bpm", 72)))
            spo2 = st.number_input("SpO₂ %", 70, 100, int(defaults.get("spo2_percent", 98)))

        st.markdown("##### Symptoms (optional)")
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

        st.markdown("##### Acute context / family history")
        ac = st.columns(5)
        acute = {}
        for i, (key, label) in enumerate(
            [
                ("acute_infection", "Infection"),
                ("dehydration", "Dehydration"),
                ("strenuous_exercise", "Exercise"),
                ("recent_surgery", "Surgery"),
                ("recent_hospitalization", "Hospitalization"),
            ]
        ):
            with ac[i]:
                acute[key] = int(st.checkbox(label, value=False, key=key))
        acute["acute_illness"] = int(
            acute["acute_infection"] or acute["dehydration"] or acute["recent_surgery"]
        )

        fh = {}
        fhc = st.columns(5)
        for i, d in enumerate(model["diseases"]):
            with fhc[i % 5]:
                fh[f"fh_{d}"] = int(
                    st.checkbox(model["disease_labels"][d], value=False, key=f"fh_{d}")
                )

        st.markdown("##### Key labs (leave 0 = treat as missing for sparse labs)")
        st.caption(
            "Tip: set a lab to **0** to leave it missing (except values that can legitimately be 0)."
        )
        # Always-visible optional labs (avoids nested widget thrash / high CPU)
        lab_keys = [
            ("lab_hba1c", "HbA1c %", 0.0, 15.0),
            ("lab_fasting_plasma_glucose", "FPG mg/dL", 0.0, 400.0),
            ("lab_egfr", "eGFR", 0.0, 140.0),
            ("lab_tsh", "TSH", 0.0, 80.0),
            ("lab_ldl_c", "LDL-C", 0.0, 300.0),
            ("lab_alt", "ALT", 0.0, 400.0),
            ("lab_nt_probnp", "NT-proBNP", 0.0, 30000.0),
            ("lab_fev1_fvc_ratio", "FEV1/FVC", 0.0, 1.0),
        ]
        labs = {}
        # remaining labs always missing to keep form light
        for k in model["numeric_features"]:
            if k.startswith("lab_"):
                labs[k] = np.nan
        lc = st.columns(4)
        for i, (key, label, lo, hi) in enumerate(lab_keys):
            with lc[i % 4]:
                v = st.number_input(label, lo, hi, 0.0, key=f"val_{key}")
                labs[key] = np.nan if v == 0.0 else float(v)

        submitted = st.form_submit_button("Run risk estimates", type="primary", use_container_width=True)

    if not submitted:
        if "last_rows" in st.session_state:
            st.markdown("### Last results")
            st.dataframe(st.session_state["last_rows"], use_container_width=True, hide_index=True)
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

    rows = predict_all(model, values)
    display = [
        {
            "Condition": r["Condition"],
            "Probability": f"{r['Probability']:.1%}",
            "Risk band": r["Risk band"],
        }
        for r in rows
    ]
    st.session_state["last_rows"] = display
    st.session_state["last_chart"] = {r["Condition"]: r["Probability"] for r in rows}
    st.session_state["last_hosp"] = predict_one(model, values, "hospitalization_12m")
    st.session_state["last_top"] = rows[0]

    st.markdown("### Results")
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.bar_chart(st.session_state["last_chart"], color="#0f5c6e")
    if st.session_state["last_hosp"] is not None:
        st.metric(
            "Synthetic 12-month hospitalization proxy",
            f"{st.session_state['last_hosp']:.1%}",
        )
    top = st.session_state["last_top"]
    st.success(
        f"Highest score: **{top['Condition']}** at **{top['Probability']:.1%}** "
        f"({top['Risk band']}) — demo only."
    )


def render_metrics(model: dict) -> None:
    st.markdown(
        """
<div class="page-head">
  <h1>Model metrics</h1>
  <p>Synthetic hold-out only — not clinical validation.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    metrics = model.get("metrics", [])
    if not metrics:
        st.write("No metrics stored.")
        return
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    aurocs = {
        m["target"].replace("disease_present_at_encounter_", ""): m["test_auroc"]
        for m in metrics
        if m.get("test_auroc") is not None
    }
    if aurocs:
        st.markdown("#### Test AUROC")
        st.bar_chart(aurocs, color="#147a8a")


def render_about(model: dict) -> None:
    st.markdown(
        """
<div class="page-head">
  <h1>About & limits</h1>
  <p>Intended use and performance notes.</p>
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
### Why this app is faster
- Runtime packages: **streamlit + numpy only** (no scikit-learn/pandas/scipy install)
- Model: small **JSON** coefficients (`models/fast_inference.json`)
- Home page does **not** load the model
- Still subject to Streamlit free-tier **cold start** if the app was idle
"""
    )
    st.code(model.get("disclaimer", ""), language=None)
    st.caption(
        f"Model v{model.get('model_version')} · format {model.get('format')} · "
        f"generator {model.get('generator_version')}"
    )


if __name__ == "__main__":
    main()
