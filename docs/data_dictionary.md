# Data Dictionary (SYNTHETIC)

> Every record is synthetic. Role = identifier | feature | target | audit | metadata.  
> **Appropriate for model training** means “allowed as input in methodological pipelines,” not “clinically validated.”

## Legend
- **Role:** identifier / feature / target / audit / metadata / sensitive_audit  
- **Train?** Yes = usable as model input if available at cutoff; No = leakage or label; Audit = fairness/leakage only  

---

## patients

| Column | Description | Type | Allowed / range | Unit | Missingness | Temporal | Role | Train? | Source/rationale |
|--------|-------------|------|-----------------|------|-------------|----------|------|--------|------------------|
| patient_id | Synthetic patient key | string | SYN-P###### | — | none | static | identifier | No (ID) | Sequential synthetic |
| age_at_index | Age at index | float | 18–95 | years | none | index | feature | Yes | Mixture of midlife/older adults |
| sex_at_birth | Sex recorded at birth | string | female, male | — | none | static | sensitive_audit/feature | Caution | 52/48 design prior |
| gender | Gender identity | string | female, male, non_binary | — | none | static | sensitive_audit | Caution | Mostly aligned; small non-cis fraction |
| ethnicity | Population group | string | 5 categories | — | none | static | sensitive_audit | Audit preferred | Design mix; external holdout shifted |
| ses_group | SES band | string | low, medium, high | — | none | static | sensitive_audit | Caution | Affects access/adherence mildly |
| access_to_care | Care access | string | limited, standard, enhanced | — | none | static | sensitive_audit/feature | Caution | Affects test density & dx lag |
| region | Geography band | string | urban, suburban, rural | — | none | static | sensitive_audit | Caution | Design prior |
| height_cm | Height | float | 140–205 | cm | none | baseline | feature | Yes | Sex-specific normal |
| weight_kg_baseline | Weight | float | plausible | kg | none | baseline | feature | Yes | From BMI×height² |
| bmi_baseline | BMI | float | 16–55 | kg/m² | none | baseline | feature | Yes | Lognormal-ish |
| waist_cm_baseline | Waist | float | 55–160 | cm | none | baseline | feature | Yes | Linked to BMI |
| smoking_status | Smoking | string | never, former, current | — | none | baseline | feature | Yes | |
| pack_years | Exposure | float | 0–100 | pack-years | none | baseline | feature | Yes | 0 if never |
| alcohol_use | Alcohol | string | none, moderate, heavy | — | none | baseline | feature | Yes | |
| physical_activity | Activity | string | sedentary, moderate, active | — | none | baseline | feature | Yes | |
| diet_risk | Diet risk | string | low, moderate, high | — | none | baseline | feature | Yes | |
| pregnancy_status_baseline | Pregnancy | string | not_applicable, pregnant | — | none | baseline | feature | Yes | Rare in F 18–45 |
| menopausal_status | Menopause | string | categories / NA | — | none | baseline | feature | Yes | Age-linked |
| fh_{disease} | Family history flags | int | 0/1 | — | none | static | feature | Yes | Probabilistic |
| susceptibility_{d} | Latent susceptibility | float | 0–1 | — | none | static | audit | No | Generator internal |
| latent_onset_day_{d} | True onset day | int | −2000–obs or −1 | days | none | latent | audit | No | −1 = never |
| trajectory_{d} | Trajectory class | string | trajectory labels | — | none | latent | audit | No | |
| treatment_start_day_{d} | Tx start | int | day or −1 | days | none | latent | audit | Partial | Use meds table for features |
| adherence_{d} | Adherence | float | 0–1 | — | none | latent | audit | Partial | |
| observation_end_day | Admin censor | int | 180–3650 | days | none | static | metadata | For censoring | |
| partition | Split label | string | train/val/test | — | none | static | metadata | Split only | Patient-level 70/15/15 |
| is_external_style_holdout | Shifted test subgroup | int | 0/1 | — | none | static | metadata | Eval only | |
| death_within_5y_latent | Death flag | int | 0/1 | — | none | latent | target/audit | Target only | Synthetic |
| death_day | Day of death | int | day or −1 | days | none | latent | audit | No | |
| is_synthetic | Always true | bool | True | — | none | static | metadata | No | Safeguard |
| dataset_disclaimer | Disclaimer | string | SYNTHETIC_RESEARCH_ONLY | — | none | static | metadata | No | Safeguard |
| generator_* / random_seed | Provenance | mixed | — | — | none | static | metadata | No | Reproducibility |

## encounters

| Column | Description | Type | Range | Unit | Missingness | Temporal | Role | Train? | Rationale |
|--------|-------------|------|-------|------|-------------|----------|------|--------|-----------|
| encounter_id | Visit key | string | SYN-P…-E## | — | none | — | identifier | No | |
| patient_id | Patient key | string | | — | none | — | identifier | No | |
| days_from_index | Time | int | ≥0 | days | none | time | feature/meta | Time index | Irregular intervals |
| sbp_mmHg, dbp_mmHg | Blood pressure | float | physio bounds | mmHg | low | at visit | feature | Yes | From HTN severity + noise |
| heart_rate_bpm | HR | float | | /min | low | at visit | feature | Yes | |
| spo2_percent | Oxygen sat | float | | % | low | at visit | feature | Yes | COPD/HF |
| weight_kg, bmi | Anthropometrics | float | | | low | at visit | feature | Yes | HF fluid effect possible |
| symptom_* | Symptom flags | int | 0/1 | — | none | at visit | feature | Yes | Probabilistic from severity |
| acute_* / dehydration / exercise | Confounders | int | 0/1 | — | none | at visit | feature | Yes | Alternative lab explanations |
| latent_* | Ground truth state | mixed | | — | none | at visit | audit/target | Task-dependent | Do not leak if predicting from clinical data only |
| observed_diagnosis_{d} | Dx known by visit | int | 0/1 | — | none | at visit | target/feature | Task-dependent | Lagged vs latent |
| partition | Split | string | | — | none | static | metadata | Split | |

## labs

| Column | Description | Type | Role | Train? | Notes |
|--------|-------------|------|------|--------|-------|
| lab_result_id | Result key | string | identifier | No | |
| analyte / value / unit | Measurement | mixed | feature | Yes if ≤ cutoff | See units doc |
| ref_low, ref_high | Reference bounds | float | metadata | Optional | Synthetic adult intervals |
| abnormality_flag | L/H/N/ERROR_* | string | feature | Yes | From value vs ref |
| specimen_type | Specimen | string | feature | Yes | |
| fasting_status | 0/1/null | int | feature | Yes | When relevant |
| hours_collection_to_processing | Preanalytics | float | feature | Yes | |
| assay_category | Method lot proxy | string | feature/audit | Yes | Noise version varies |

**Missingness:** structural non-ordering (MCAR/MAR/MNAR mixture). Absence of row = not measured.

## diagnoses / medications

See schema. Diagnoses after evidence date only (except intentional coding noise in noisy version). Medications only on/after treatment initiation logic.

## prediction_targets

| Column pattern | Role | Train? | Definition summary |
|----------------|------|--------|--------------------|
| prediction_timestamp_day | metadata | — | T |
| feature_cutoff_timestamp_day | metadata | — | Features must be ≤ this day |
| outcome_window_* | metadata | — | Window bounds |
| censoring_timestamp_day | metadata | — | Admin censor |
| disease_present_at_encounter_{d} | target | No | Latent present at T |
| new_disease_within_*_{d} | target | No | 1/0/−1 |
| disease_stage_at_encounter_{d} | target | No | Stage label |
| progression_within_12_months_{d} | target | No | Severity rise |
| major_complication_within_12_months_{d} | target | No | Proxy |
| hospitalization_within_12_months | target | No | Proxy |
| treatment_response_at_next_followup | target | No | |
| biomarker_control_at_next_followup | target | No | |
| all_cause_mortality_within_5_years | target | No | Index only |
| true_latent_* / observed_diagnosis_present_* | audit/target | Task-dependent | Layer comparison |

Full machine generation of every dynamic column is in code: `src/generate.py`, `src/reference_ranges.py`.
