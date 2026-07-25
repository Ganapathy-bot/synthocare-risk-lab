# Fairness Report (SYNTHETIC DATA)

## Sample sizes
- **n_by_sex_at_birth**: `{'female': 1596, 'male': 1404}`
- **n_by_ethnicity**: `{'White': 1609, 'Hispanic_or_Latino': 458, 'Black_or_African_American': 458, 'Asian': 239, 'Other_or_Multiple': 236}`
- **n_by_ses_group**: `{'medium': 1414, 'low': 831, 'high': 755}`
- **n_by_access_to_care**: `{'standard': 1709, 'limited': 711, 'enhanced': 580}`
- **n_by_region**: `{'urban': 1338, 'suburban': 1075, 'rural': 587}`
- **n_by_age_group**: `{'55-69': 1073, '40-54': 889, '70+': 690, '18-39': 348}`

## Underrepresented groups
- ethnicity=Asian share=0.080
- ethnicity=Other_or_Multiple share=0.079

## Proxy warnings
- ses_group and access_to_care may proxy structural inequities; do not use as clinical features for deployment decisions
- ethnicity is for fairness auditing only; avoid as a direct predictive feature without causal justification
- region and access correlate with test density — models may learn care intensity rather than biology

## Caution
- Synthetic demographic effects are mild and non-deterministic; real-world disparities differ
- Subgroup performance on this dataset does not validate fairness in clinical use
- Smaller groups (e.g., RA positives within ethnicity strata) may be unstable