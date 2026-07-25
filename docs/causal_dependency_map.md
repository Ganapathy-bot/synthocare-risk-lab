# Causal / Dependency Map (SYNTHETIC)

```text
[Demographics: age, sex, ethnicity*]
[Lifestyle: smoking, alcohol, activity, diet]
[Anthropometrics: BMI, waist]
[SES / access / region*]
[Family history]
        |
        v
[Shared latent risks: metabolic, CV, inflammatory]
        |
        +-----> [Disease susceptibility vector]
        |              |
        |              +-- multimorbidity edges (soft coupling)
        |              v
        |       [Onset day + trajectory type + max severity]
        |              |
        |              v
        |       [Latent severity(t)] <--- treatment_start(t0), adherence
        |              |
        +------+-------+--------+
               |                |
               v                v
        [Symptoms/Vitals]  [Biomarkers(t)]
               |                |
               v                v
        [Test ordering policy (access, risk, prior abnormalities)]
               |
               v
        [Observed labs subset]
               |
               v
        [Evidence score] --> [Observed diagnosis date]
               |
               v
        [Medication initiation events]
               |
               v
        [Prediction targets at time T with windows (T, T+w]]
               |
               v
        [Noise layer: assay error, missingness already applied, coding FP]

* Sensitive / audit variables — not deterministic disease switches
```

## Dependency highlights

| Downstream | Parents (primary) |
|------------|-------------------|
| T2DM severity | metabolic risk, BMI, age, FH, treatment |
| FPG / HbA1c | T2DM severity, infection, steroids, pregnancy, treatment |
| CKD severity | age, T2DM, HTN, susceptibility |
| Creatinine / eGFR / UACR | CKD severity, dehydration, ACEI effect, acute illness |
| HTN severity | metabolic + CV risk, age |
| SBP/DBP | HTN severity, treatment, HF interactions |
| CAD severity | CV risk, smoking, lipids drivers, T2DM, HTN |
| HF severity | CAD, HTN, CKD, age |
| NT-proBNP | HF severity, age, CKD |
| COPD severity | pack-years, smoking |
| FEV1/FVC | COPD severity |
| MASLD severity | metabolic risk, alcohol, BMI |
| Liver enzymes | MASLD severity, alcohol, exercise (AST) |
| RA severity | inflammatory risk, sex (soft) |
| RF/anti-CCP | RA severity with seronegative noise |
| Hypothyroid severity | sex (soft), susceptibility |
| TSH/FT4 | severity, levothyroxine adherence |
| Cognitive severity | age (strong), susceptibility |
| MMSE/MoCA/FAQ | cognitive severity, age |
| Observed diagnosis | evidence, access, time since onset |
| Hospitalization label | HF/COPD/CKD/CAD severity |

## Treatment effect timing invariant
For any disease D: if `day < treatment_start_day_D`, biomarkers **must not** reflect treatment for D. Enforced in `severity_at()` and `_labs_for_state()`.
