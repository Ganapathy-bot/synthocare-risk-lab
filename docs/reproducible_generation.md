# Reproducible Generation Code & Pseudocode

## Command

```bash
python generate_dataset.py --n-patients 3000 --seed 42
```

- **Generator name:** `SyntheticChronicDiseaseGenerator`  
- **Version:** `1.2.0`  
- **Default seed:** `42`  
- **Entry:** `generate_dataset.py` → `src/generate.py`  

## Pseudocode

```
set_seed(SEED)
assign patient-level partitions (70/15/15); mark external-style subset of test

for i in 1..N_PATIENTS:
    draw demographics, lifestyle, anthropometrics, SES/access
    draw family history
    compute shared metabolic, CV, inflammatory risk scores
    for each disease D:
        susceptibility[D] <- base_risk * age/sex/lifestyle/FH adjustments
    apply multimorbidity soft boosts
    for each disease D:
        with prob 1-susceptibility: stable low risk (no onset)
        else with prob 0.25: at-risk only (no diagnostic onset)
        else: onset_day (historical or incident), trajectory, max_severity
              maybe treatment_start after delayed care access
    draw irregular encounter schedule over 6m–10y

    for each encounter day t:
        severity[D,t] <- f(onset, trajectory, treatment, adherence)  # deterministic given seed
        vitals, symptoms <- g(severity, acute_confounders)
        all_labs <- h(severity, confounders, meds after t0 only)
        ordered_labs <- test_ordering_policy(risk, access, acuity)
        maybe record diagnosis if evidence_score high enough (lagged)
        maybe initiate medications if treatment_start==t
        write targets at t with windows and censor codes (-1)

write clean tables
apply noise layer -> noisy tables
run validation & reports
```

## Module map

| Module | Responsibility |
|--------|----------------|
| `src/config.py` | Seeds, paths, disease list, priors |
| `src/reference_ranges.py` | Units, ref intervals, flags |
| `src/generate.py` | Latent process + tables + noise |
| `src/validate_and_report.py` | QA, fairness, leakage, class balance |
| `generate_dataset.py` | CLI |

## Version stamp

Every `manifest.json` and patient row records `generator_version` and `random_seed`.
