# Dataset Card — Synthetic Chronic Disease Cohort (v1.0.0)

## Status banner

| | |
|--|--|
| **Data type** | **SYNTHETIC** — computer-generated, not derived from real individuals |
| **Intended use** | Research, education, ML pipeline development, methodology |
| **Prohibited use** | Clinical decision-making, diagnosis, treatment selection, deployment without external validation on real data, claiming population representativeness |
| **Generator** | `SyntheticChronicDiseaseGenerator` v1.2.0 |
| **Default seed** | 42 |
| **Default N** | 3000 patients (configurable via `--n-patients`) |

---

## 1. Intended uses

- Early risk / pre-diagnostic screening model prototypes  
- Differential diagnosis feature engineering experiments  
- Disease-onset and progression prediction methods  
- Severity/stage estimation  
- Longitudinal monitoring pipelines  
- Complication / hospitalization risk modeling  
- Treatment-response labeling experiments  
- Fairness, missingness, and leakage-control methodology  
- Train/val/test and temporal holdout engineering  

## 2. Prohibited uses

- Direct patient care or clinical decision support in production  
- Regulatory claims without real-world evidence  
- Re-identification or “realistic patient” storytelling as if real  
- Presenting enriched or synthetic prevalence as epidemiology  

## 3. Population represented

- **Synthetic adults** with mixed chronic-disease burden  
- Design demographic priors (not a national survey)  
- Includes healthy, at-risk, early/moderate/advanced disease, multimorbidity, and alternative-cause biomarker abnormalities  
- `is_external_style_holdout=1`: shifted ethnicity/access mix for stress tests  

## 4. Disease definitions

See `docs/disease_definitions.md`. Ten chronic conditions + controls. Biomarkers are **probabilistic evidence**, not perfect oracles.

## 5. Generation method

Latent-state simulator with explicit causal order (risk → susceptibility → onset/progression → observations → treatment → diagnosis → noise).  
Code: `src/generate.py`. Assumptions: `docs/generation_assumptions.md`. Map: `docs/causal_dependency_map.md`.

## 6. Data tables

| Table | Grain |
|-------|-------|
| patients | 1 / patient |
| encounters | 1 / visit |
| labs | 1 / result (long) |
| diagnoses | 1 / diagnosis event |
| medications | 1 / initiation event |
| prediction_targets | 1 / encounter |

Clean + noisy versions under `data/clean` and `data/noisy`.

## 7. Partitions

- **Patient-level** split: ~70% train / 15% val / 15% test  
- No patient in more than one partition  
- Optional external-style subset of test  

## 8. Leakage controls

- Explicit `prediction_timestamp_day`, `feature_cutoff_timestamp_day`, outcome windows, censoring  
- Treatment effects only after initiation  
- Observed diagnoses only after evidence process  
- Target columns documented; must not be used as features  
- Audit: `reports/leakage_audit.md`  

## 9. Missingness & quality

Mixture of MCAR/MAR/MNAR via test-ordering and access. Noisy version adds assay noise, duplicates, rare negative entry errors (flagged), and false-positive diagnosis codes. Reports in `reports/`.

## 10. Bias and fairness considerations

Sensitive variables retained for **auditing**. Mild non-deterministic associations only. Models may learn care-intensity proxies (`access_to_care`). Subgroup metrics on synthetic data **do not** prove real-world fairness. See `reports/fairness_report.md`.

## 11. Known limitations

- Simplified eGFR and BP diagnostic logic  
- Limited imaging/procedure detail  
- Cognitive domain without CSF/PET  
- Medication effects class-average  
- Complication/hospitalization/mortality are **proxies**  
- Prevalence is a design target, not epidemiology  

## 12. Validation results

Run `python generate_dataset.py` then read `reports/validation_summary.md`. Checks include bounds, partition exclusivity, trajectory order, prevalence tables, correlation summaries, leakage audit.

## 13. Recommended evaluation metrics

| Task family | Metrics |
|-------------|---------|
| Binary risk / diagnosis | AUROC, AUPRC, sensitivity, specificity, PPV, NPV at pre-specified thresholds |
| Calibration | Calibration slope/intercept, Brier, reliability plots |
| Clinical utility | Decision-curve analysis (net benefit) — **required** before claiming usefulness |
| Ordinal stage | Quadratic weighted kappa, ordinal logistic metrics, MAE on mapped scores |
| Longitudinal / onset | Time-dependent AUC, C-index, calibration over time; respect censoring (−1) |
| Subgroups | Metric gaps across age/sex/ethnicity/SES/access; sample-size weighted interpretation |
| Survival (mortality) | Time-dependent AUC, C-index, IBS — synthetic only |

**Do not** treat strong discrimination alone as proof of clinical usefulness.

## 14. Conditions requiring external clinical validation

Any model intended for screening, diagnosis support, prognosis, or monitoring in real care **must** be validated on representative real-world data with clinical governance, prospective evaluation where appropriate, and regulatory review as applicable.

## 15. Versioning & reproducibility

```bash
python generate_dataset.py --n-patients 3000 --seed 42
```

- Generator version: 1.2.0  
- Seed: 42 (default)  
- Manifest: `data/clean/manifest.json`  
- Optional flat table: `data/clean/flat_encounter_level.csv` (one row per encounter)  

### Class balance / enrichment note
Some lower-prevalence conditions (e.g., RA) are modestly enriched relative to crude population rates so that training sets contain usable positives. Manifest `prevalence_note` documents this. Do **not** interpret synthetic prevalence as epidemiology. For rare-outcome experiments, filter or reweight using design knowledge; no separate sampling-weight column is required for the default cohort because enrichment is mild and disclosed.

## 16. Citation / credit

Synthetic dataset generated for research education. Not affiliated with any health system. Do not cite as clinical evidence.
