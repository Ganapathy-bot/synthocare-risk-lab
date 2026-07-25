# Units and Reference Ranges (SYNTHETIC)

> **Disclaimer:** These are simplified adult reference-style intervals used to flag abnormalities in **synthetic** data. They are **not** assay-specific clinical reference ranges and must **not** be used for real patient interpretation.

Conceptual sources informing design: ADA Standards of Care (glycemia); KDIGO (CKD staging and albuminuria); ACC/AHA (BP and lipids risk framing); GOLD (spirometry obstruction threshold); AASLD/EASL-style liver enzyme panels; ACR RA serology concepts; ATA thyroid testing; NIA-AA cognitive staging concepts. Exact cutoffs are simplified for simulation.

| Analyte | Unit | Ref low | Ref high | Specimen | Notes |
|---------|------|---------|----------|----------|-------|
| Fasting plasma glucose | mg/dL | 70 | 99 | plasma | Diabetes criteria use clinical rules, not this alone |
| Random plasma glucose | mg/dL | 70 | 140 | plasma | Context-dependent |
| HbA1c | % | 4.0 | 5.6 | whole blood | Prediabetes/diabetes thresholds modeled separately |
| Fasting insulin | uIU/mL | 2 | 25 | serum | Not routine for T2DM diagnosis |
| C-peptide | ng/mL | 0.8 | 3.1 | serum | Specialty context |
| Triglycerides | mg/dL | 0 | 149 | serum | Fasting preferred |
| HDL-C | mg/dL | 40 (50 F risk) | 100 | serum | Risk-oriented floor |
| LDL-C | mg/dL | 0 | 99 | serum | Risk-based targets differ clinically |
| Total cholesterol | mg/dL | 0 | 199 | serum | |
| Serum creatinine | mg/dL | sex-specific | sex-specific | serum | F ~0.5–1.1; M ~0.7–1.3 |
| eGFR | mL/min/1.73m² | 90 | 120 | calculated | KDIGO categories for staging |
| BUN | mg/dL | 7 | 20 | serum | |
| UACR | mg/g | 0 | 29 | urine | A2 ≥30; A3 >300 |
| Sodium | mmol/L | 136 | 145 | serum | |
| Potassium | mmol/L | 3.5 | 5.1 | serum | |
| Bicarbonate | mmol/L | 22 | 29 | serum | |
| Phosphate | mg/dL | 2.5 | 4.5 | serum | |
| Calcium | mg/dL | 8.6 | 10.2 | serum | |
| Hemoglobin | g/dL | sex-specific | sex-specific | whole blood | |
| hs-CRP | mg/L | 0 | 3 | serum | Inflammation / CV risk context |
| CRP | mg/L | 0 | 10 | serum | |
| ESR | mm/h | 0 | 20 | whole blood | Age/sex variation simplified |
| NT-proBNP | pg/mL | 0 | 125 | plasma | Age-dependent cutoffs exist |
| hs-Troponin | ng/L | 0 | 14 | plasma | Acute context; assay-specific |
| ApoB | mg/dL | 0 | 100 | serum | |
| Lp(a) | nmol/L | 0 | 75 | serum | Genetic; threshold simplified |
| ALT / AST | U/L | ~7–10 | 40 | serum | |
| ALP | U/L | 40 | 129 | serum | |
| GGT | U/L | 5 | 40 | serum | |
| Total bilirubin | mg/dL | 0.1 | 1.2 | serum | |
| Direct bilirubin | mg/dL | 0 | 0.3 | serum | |
| Albumin | g/dL | 3.5 | 5.0 | serum | |
| INR | ratio | 0.8 | 1.2 | plasma | |
| Platelets | 10^9/L | 150 | 400 | whole blood | |
| WBC | 10^9/L | 4 | 11 | whole blood | |
| Eosinophils | cells/µL | 0 | 500 | whole blood | |
| FEV1 % predicted | % | 80 | 120 | spirometry | |
| FVC % predicted | % | 80 | 120 | spirometry | |
| FEV1/FVC | ratio | 0.70 | 0.90 | spirometry | Obstruction often <0.70 |
| SpO2 | % | 95 | 100 | pulse ox | |
| RF | IU/mL | 0 | 14 | serum | Not specific |
| Anti-CCP | U/mL | 0 | 20 | serum | Higher specificity than RF |
| TSH | mIU/L | 0.4 | 4.0 | serum | |
| Free T4 | ng/dL | 0.8 | 1.8 | serum | |
| Free T3 | pg/mL | 2.3 | 4.2 | serum | |
| TPO Ab | IU/mL | 0 | 34 | serum | |
| MMSE | points | 24 | 30 | scale | Lower abnormal; not diagnostic alone |
| MoCA | points | 26 | 30 | scale | Education adjustment not fully modeled |
| FAQ | points | 0 | 8 | scale | Higher = more impairment |

Machine-readable specs live in `src/reference_ranges.py`.
