# Dataset Schema (SYNTHETIC)

All tables include:
- `is_synthetic` (bool/True)
- `dataset_disclaimer` = `SYNTHETIC_RESEARCH_ONLY`

Relative time uses `days_from_index` (integer; 0 = index/baseline encounter).

## 1. `patients` (1 row per patient)

| Column group | Examples |
|--------------|----------|
| Identifiers | `patient_id` |
| Demographics | `age_at_index`, `sex_at_birth`, `gender`, `ethnicity` |
| Social | `ses_group`, `access_to_care`, `region` |
| Anthropometrics | `height_cm`, `weight_kg_baseline`, `bmi_baseline`, `waist_cm_baseline` |
| Lifestyle | `smoking_status`, `pack_years`, `alcohol_use`, `physical_activity`, `diet_risk` |
| Reproductive | `pregnancy_status_baseline`, `menopausal_status` |
| Family history | `fh_{disease}` |
| Latent process (audit) | `susceptibility_{d}`, `latent_onset_day_{d}`, `trajectory_{d}`, `treatment_start_day_{d}`, `adherence_{d}` |
| Follow-up | `observation_end_day`, `n_encounters`, `loss_to_followup` |
| Mortality (synthetic) | `death_within_5y_latent`, `death_day` |
| Split | `partition` ∈ {train,val,test}, `is_external_style_holdout` |
| Metadata | `generator_name`, `generator_version`, `random_seed` |

## 2. `encounters` (1 row per visit)

| Column group | Examples |
|--------------|----------|
| Keys | `encounter_id`, `patient_id`, `encounter_number`, `days_from_index`, `is_baseline` |
| Vitals | `sbp_mmHg`, `dbp_mmHg`, `heart_rate_bpm`, `spo2_percent`, `weight_kg`, `bmi` |
| Symptoms | `symptom_*` binary flags |
| Acute confounders | `acute_infection`, `dehydration`, `strenuous_exercise`, `recent_surgery`, `recent_hospitalization` |
| Latent state | `latent_severity_{d}`, `latent_disease_present_{d}`, `latent_stage_{d}` |
| Observed dx | `observed_diagnosis_{d}` |
| Split | `partition` |

## 3. `labs` (long format; 1 row per result)

| Column | Description |
|--------|-------------|
| `lab_result_id` | Unique synthetic ID |
| `patient_id`, `encounter_id`, `days_from_index` | Keys / time |
| `analyte`, `analyte_name` | Test code / label |
| `value`, `unit`, `unit_standardized` | Result |
| `ref_low`, `ref_high`, `abnormality_flag` | Reference framing |
| `specimen_type`, `fasting_status` | Context |
| `hours_collection_to_processing` | Preanalytical proxy |
| `assay_category`, `loinc_like_code` | Method metadata (synthetic codes) |

## 4. `diagnoses`

| Column | Description |
|--------|-------------|
| `diagnosis_id` | Unique |
| `patient_id`, `encounter_id`, `days_from_index` | When recorded |
| `disease_code`, `disease_label` | Condition |
| `diagnosis_type` | incident / historical / coding_noise_false_positive (noisy) |
| `clinical_stage_at_diagnosis`, `evidence_score` | Context |
| `is_true_latent_disease`, `is_observed_diagnosis` | Layer flags |

## 5. `medications`

| Column | Description |
|--------|-------------|
| `medication_event_id` | Unique initiation event |
| `medication_class`, `indication_disease` | What / why |
| `start_day`, `stop_day`, `adherence_estimate`, `is_active` | Exposure |

## 6. `prediction_targets` (1 row per encounter)

| Column group | Examples |
|--------------|----------|
| Time controls | `prediction_timestamp_day`, `feature_cutoff_timestamp_day`, `censoring_timestamp_day`, `outcome_window_start_{m}m`, `outcome_window_end_{m}m` |
| Instantaneous | `disease_present_at_encounter_{d}`, `disease_stage_at_encounter_{d}` |
| Incidence | `new_disease_within_{6,12,36}_months_{d}` (−1 censored/ineligible) |
| Progression / complications | `progression_within_12_months_{d}`, `major_complication_within_12_months_{d}` |
| Utilization / response | `hospitalization_within_12_months`, `treatment_response_at_next_followup`, `biomarker_control_at_next_followup` |
| Mortality | `all_cause_mortality_within_5_years`, `eligible_mortality_5y` |
| Layers | `true_latent_disease_present_{d}`, `observed_diagnosis_present_{d}` |
| Split | `partition`, `is_external_style_holdout` |

## Entity-relationship

```text
patients 1---* encounters 1---* labs
patients 1---* diagnoses
patients 1---* medications
encounters 1---1 prediction_targets
```

## 7. `flat_encounter_level` (optional denormalized)

One row per encounter with:
- Encounter vitals/symptoms/latent flags
- Repeated stable patient attributes
- Wide subset of common labs (`lab_hba1c`, `lab_egfr`, …)
- Core prediction targets and leakage timestamps

Use the relational tables for full fidelity; use flat for quick prototyping.

## File locations

| Version | Path |
|---------|------|
| Clean | `data/clean/*.csv` (+ `.parquet` when available) |
| Noisy | `data/noisy/*.csv` |
| Manifest | `data/{clean|noisy}/manifest.json` |
