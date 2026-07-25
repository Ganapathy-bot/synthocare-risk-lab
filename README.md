# SynthoCare Risk Lab

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-0f5c6e.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data](https://img.shields.io/badge/data-SYNTHETIC-d4a017.svg)](#disclaimer)

> **SYNTHETIC DATA ONLY** — Research, education, and ML pipeline development.  
> **Not real patients. Not clinically validated. Not for diagnosis or patient care.**

End-to-end demo of a **latent-state synthetic chronic-disease cohort**, multi-label risk models, and a polished **Streamlit** app for interactive scenario scoring.

---

## Features

- **Synthetic longitudinal cohort** (patient, encounter, lab, diagnosis, medication, targets)
- **10 chronic disease areas** + healthy / at-risk / multimorbidity patterns  
  T2DM · CKD · CAD · HF · HTN · COPD · CLD/MASLD · RA · hypothyroidism · cognitive decline
- **Train / val / test** split by **patient ID** (70 / 15 / 15)
- **HistGradientBoosting** disease-presence models with missing-lab support
- **Streamlit UI** — landing page, prediction form, metrics, about/limits
- Full **documentation**: data dictionary, schema, leakage audit, fairness notes

---

## Repository structure

```text
.
├── generate_dataset.py          # CLI: generate synthetic tables
├── train_models.py              # CLI: train & save model bundle
├── streamlit_app.py             # Streamlit demo app
├── requirements.txt
├── LICENSE
├── DATASET_CARD.md
├── assets/                      # UI hero image
├── docs/                        # Schema, dictionary, assumptions, …
├── models/                      # streamlit_disease_models.joblib (+ metrics)
├── reports/                     # Validation / fairness / leakage reports
└── src/
    ├── config.py
    ├── generate.py
    ├── reference_ranges.py
    └── validate_and_report.py
```

Large raw CSV/Parquet tables under `data/` are **not committed** (size). Regenerate locally (below). Manifests and reports remain for documentation.

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/Ganapathy-bot/synthocare-risk-lab.git
cd synthocare-risk-lab
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Generate synthetic data

```bash
python generate_dataset.py --n-patients 3000 --seed 42
```

Outputs: `data/clean/`, `data/noisy/`, `reports/`.

### 3. Train models

```bash
python train_models.py
```

Writes: `models/streamlit_disease_models.joblib`, `models/training_metrics.json`.

> A pre-trained model bundle may already be in `models/` so the Streamlit app can run without re-training. Re-train after regenerating data for consistency.

### 4. Launch Streamlit (local)

```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501**

### 5. Deploy live on Streamlit Community Cloud

1. Open: [https://share.streamlit.io/deploy](https://share.streamlit.io/deploy)
2. Sign in with **GitHub**
3. Select repository: **`Ganapathy-bot/synthocare-risk-lab`**
4. Branch: **`main`**
5. Main file path: **`streamlit_app.py`**
6. Click **Deploy**

Direct deploy link (after GitHub auth):

[https://share.streamlit.io/deploy?repository=Ganapathy-bot/synthocare-risk-lab&branch=main&mainModule=streamlit_app.py](https://share.streamlit.io/deploy?repository=Ganapathy-bot/synthocare-risk-lab&branch=main&mainModule=streamlit_app.py)

Cloud settings used by this repo:

| Setting | Value |
|---------|--------|
| App file | `streamlit_app.py` |
| Requirements | `requirements.txt` |
| Python | `runtime.txt` → 3.11 |
| Model bundle | `models/streamlit_disease_models.joblib` (committed) |
| Theme | `.streamlit/config.toml` |

No dataset generation is required on Cloud — the pre-trained model is already in the repo.

| Page | Description |
|------|-------------|
| **Home** | Landing page & pipeline overview |
| **Risk prediction** | Enter scenario features, get multi-disease scores |
| **Model metrics** | Synthetic hold-out AUROC / AUPRC |
| **About & limits** | Intended / prohibited uses |

---

## Modeling notes

| Item | Detail |
|------|--------|
| Targets | `disease_present_at_encounter_{disease}` (+ hospitalization proxy) |
| Features | Demographics, lifestyle, vitals, symptoms, family history, optional labs |
| Split | Patient-level only — never split rows across patients |
| Leakage | No future labs/meds/diagnoses; no latent severity as features |
| Metrics | AUROC, AUPRC, Brier on synthetic val/test (not clinical proof) |

See `DATASET_CARD.md` and `docs/` for full definitions, units, and leakage controls.

---

## Documentation

| Document | Path |
|----------|------|
| Dataset card | [DATASET_CARD.md](DATASET_CARD.md) |
| Schema | [docs/schema.md](docs/schema.md) |
| Data dictionary | [docs/data_dictionary.md](docs/data_dictionary.md) |
| Disease / outcome definitions | [docs/disease_definitions.md](docs/disease_definitions.md) |
| Units & reference ranges | [docs/units_reference_ranges.md](docs/units_reference_ranges.md) |
| Generation assumptions | [docs/generation_assumptions.md](docs/generation_assumptions.md) |
| Causal map | [docs/causal_dependency_map.md](docs/causal_dependency_map.md) |
| Reproducibility | [docs/reproducible_generation.md](docs/reproducible_generation.md) |

---

## Requirements

- Python 3.10+
- See `requirements.txt`: `pandas`, `numpy`, `scipy`, `scikit-learn`, `joblib`, `streamlit`, `pyarrow`

---

## Disclaimer

This project provides **synthetic** healthcare-style data and **demonstration** models only.

**Do not** use outputs for:

- Clinical diagnosis, triage, or treatment decisions  
- Regulatory claims without real-world evidence  
- Representing the data as real patients or population statistics  

Any model intended for real care **must** be externally validated on representative clinical data under appropriate governance.

---

## License

MIT — see [LICENSE](LICENSE). Medical non-use notice is included in the license file.

---

## Citation

If you use this repository in teaching or methods work, please note that the data are synthetic and cite the repository URL and generator version (`SyntheticChronicDiseaseGenerator` v1.2.0, seed 42 by default).
