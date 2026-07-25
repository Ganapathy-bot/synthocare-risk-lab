# Disease and Outcome Label Definitions (SYNTHETIC)

> Synthetic labels for methodological research. **Not** clinical diagnostic criteria for patient care.

## Label layers (critical distinction)

| Layer | Meaning | Typical columns |
|-------|---------|-----------------|
| **True latent disease** | Generator ground truth chronic disease state | `latent_disease_present_*`, `true_latent_disease_present_*`, `latent_severity_*` |
| **Clinically observed diagnosis** | Diagnosis recorded only after probabilistic evidence + care access | `observed_diagnosis_*`, diagnoses table |
| **Model-prediction target** | Task-specific label chosen by the user | `disease_present_at_encounter_*`, `new_disease_within_*`, etc. |

Delayed and missed diagnoses are intentional: observed diagnosis can lag latent onset.

---

## Disease definitions (latent process)

### Type 2 diabetes mellitus (`t2dm`)
- Latent onset when susceptibility and metabolic risk cross internal threshold.
- Biomarker evidence (non-deterministic): HbA1c, FPG, symptoms (polyuria/polydipsia).
- Stages: none → prediabetes → controlled → uncontrolled → with complications.
- **Not** diagnosed by a single lab in the generator.

### Chronic kidney disease (`ckd`)
- Latent severity drives eGFR decline and UACR rise.
- Acute dehydration/infection can raise creatinine **without** setting CKD latent present.
- Stages map roughly to KDIGO-like eGFR/albuminuria severity bands (simplified).

### Coronary artery disease (`cad`)
- Driven by CV risk score, age, smoking, lipids, comorbidity.
- Symptoms (chest pain) and lipids contribute evidence; imaging not fully simulated.
- Troponin used mainly in acute contexts (not a chronic CAD screening label alone).

### Heart failure (`hf`)
- Severity drives NT-proBNP, dyspnea, edema, heart rate, SpO2 interactions with COPD.
- Stages approximate A/B through NYHA-like severity.

### Hypertension (`htn`)
- Latent severity elevates SBP/DBP; treatment lowers BP after initiation.
- Diagnosis evidence uses repeated-style BP elevation probability (single-encounter simplification).

### COPD (`copd`)
- Strongly linked to smoking pack-years.
- Spirometry FEV1/FVC and FEV1% predicted, cough, SpO2, eosinophils (subset).

### Chronic liver disease / MASLD (`cld_masld`)
- Linked to metabolic risk, BMI, alcohol.
- ALT/AST/GGT/albumin/INR/platelets reflect steatosis → cirrhosis spectrum.
- Alternative causes (alcohol, exercise-related AST) can elevate enzymes without diagnosis.

### Rheumatoid arthritis (`ra`)
- Joint symptoms + RF/anti-CCP/CRP/ESR; **seronegative** patterns included.
- Treatment with DMARDs/steroids; steroids may raise glucose (confounding).

### Hypothyroidism (`hypothyroid`)
- TSH up / free T4 down with severity; levothyroxine improves after adherence-weighted treatment.
- TPO antibodies enriched when autoimmune process present.

### Alzheimer’s / related cognitive decline (`alzheimers`)
- Strong age skew; cognitive complaints; MMSE/MoCA/FAQ.
- **Not** diagnosed by a single cognitive score; specialized CSF/imaging biomarkers are **not** claimed as routine.

---

## Prediction targets

### `disease_present_at_encounter_{disease}`
- **Prediction time:** encounter day T  
- **Look-back:** features with `days_from_index <= T`  
- **Outcome window:** instantaneous state at T  
- **Positive:** latent disease present at T  
- **Negative:** not present  
- **Censoring:** none for this instantaneous label  

### `new_disease_within_{6|12|36}_months_{disease}`
- **Eligible:** no latent disease at T  
- **Positive:** latent onset in (T, T+window] and on/before censor  
- **Negative:** no onset through window end  
- **Censored/ineligible:** `-1` if already diseased or censored before adequate follow-up  

### `disease_stage_at_encounter_{disease}`
- Ordinal/categorical stage string at T from latent severity.

### `progression_within_12_months_{disease}`
- Positive if severity rises by ≥0.15 (and clinically meaningful floor) within 12 months.

### `major_complication_within_12_months_{disease}`
- Synthetic complication proxy conditional on disease presence and future severity; not a specific ICD list.

### `hospitalization_within_12_months`
- Any-cause hospitalization risk proxy from cardiorenal-pulmonary severity.

### `treatment_response_at_next_followup` / `biomarker_control_at_next_followup`
- Response: severity reduction after treatment by next encounter.
- Control: disease-relevant biomarker thresholds (e.g., HbA1c <7%, SBP <130, TSH in range).

### `all_cause_mortality_within_5_years`
- Defined at **index encounter only** (`eligible_mortality_5y=1`).
- Synthetic mortality process; **not** actuarial.

---

## Outcome coding conventions

| Value | Meaning |
|-------|---------|
| 1 | Positive |
| 0 | Negative |
| -1 | Ineligible or censored |

Always filter to eligible rows before training classifiers.
