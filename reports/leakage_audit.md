# Leakage Audit (SYNTHETIC DATA)

Status: **PASS**

- OK: each patient in exactly one partition
- OK: feature_cutoff equals prediction_timestamp
- Documented 94 target/label columns excluded from feature sets
- OK: no single screened biomarker perfectly determines disease labels
- OK by design: medication effects applied only when day >= treatment_start; prediction features should use labs/vitals/meds with days_from_index <= feature_cutoff
- OK by design: train/val/test split is patient-level; temporal holdout flagged via is_external_style_holdout

## Required modeling practice
- Split by `patient_id` only (pre-assigned `partition` column).
- At prediction time T, use only rows with `days_from_index <= feature_cutoff_timestamp_day`.
- Do not use columns listed as targets/labels as features.
- Do not use future medications, diagnoses, or labs beyond cutoff.
- Do not use `latent_*` audit fields if the task is to predict observed diagnosis from clinical features only — choose target explicitly.