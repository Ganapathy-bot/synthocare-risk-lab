# Generation Assumptions (SYNTHETIC)

## Purpose
This document states modeling assumptions of `SyntheticChronicDiseaseGenerator` v1.0.0 so users can judge fitness for **methodological** research only.

## Population
- Synthetic adult cohort (ages ~18–95) with enrichment of middle-aged chronic disease and older adults for cognitive outcomes.
- Demographic mix is a **design choice**, not a census or EHR extract.
- External-style holdout shifts ethnicity and access distributions.

## Causal order implemented
1. Demographics, lifestyle, family history, SES/access  
2. Shared metabolic / CV / inflammatory risk scores  
3. Disease susceptibility with multimorbidity boosts  
4. Onset day (historical or incident) and trajectory type  
5. Time-varying latent severity  
6. Symptoms, vitals, biomarkers  
7. Test ordering (access- and risk-dependent)  
8. Probabilistic clinical diagnosis from evidence  
9. Treatment start, adherence, biomarker response  
10. Targets with explicit windows  
11. Optional noise layer (assay noise, coding FP, duplicates, rare entry errors)

## Biological / clinical simplifications
- eGFR is a **simplified function** of age/creatinine/severity — not full CKD-EPI implementation.
- BP diagnosis uses encounter BP + latent state (no full multi-day ambulatory average).
- CAD/HF lack ECG/echo/Cath structured reports (summary via biomarkers/symptoms).
- MASLD lacks elastography/biopsy; fibrosis approximated via lab pattern + severity.
- Alzheimer’s lacks CSF/PET; cognitive scales only.
- Medication effects are class-level average effects with adherence heterogeneity.
- Pregnancy is rare and only lightly modulates glucose.

## Multimorbidity
Pairwise susceptibility boosts for clinically common clusters (e.g., T2DM–HTN–CKD, CAD–HF). Joint prevalence can exceed independent products by design.

## Confounders explicitly modeled
Acute infection, dehydration, strenuous exercise, pregnancy (glucose), systemic corticosteroids (glucose), diuretics/ACEI-ARB (electrolytes/creatinine), statins (LDL), smoking, alcohol, temporary kidney injury pattern, troponin after exercise/acute stress.

## Missingness
- **MCAR:** random gaps in routine panels  
- **MAR:** confirmatory/specialty tests after risk or abnormal screens  
- **MNAR:** limited access → fewer specialty tests; loss to follow-up patterns  

## What is intentionally *not* claimed
- National prevalence accuracy  
- Transportability to any real health system  
- Suitability for regulatory device training without external data  
- Replacement for prospective clinical studies  
