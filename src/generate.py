"""
Latent-state synthetic chronic-disease cohort generator.

Causal order:
1. Demographics, family history, lifestyle, baseline risk
2. Latent disease susceptibility
3. Onset and progression over time
4. Symptoms, biomarkers, vitals from latent state
5. Medication / intervention effects after initiation
6. Healthcare utilization and test-ordering
7. Observed diagnoses from available evidence only
8. Measurement error, missingness, coding noise (noisy version)

SYNTHETIC DATA ONLY — research / education / model-development use.
Not real patients. Not for clinical decision-making.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import (
    ACCESS,
    ACTIVITY,
    ALCOHOL,
    BASE_LIFETIME_RISK,
    CLEAN_DIR,
    COMORBIDITY_BOOST,
    DEFAULT_N_PATIENTS,
    DIET_RISK,
    DISEASES,
    DISEASE_LABELS,
    ETHNICITY,
    EXTERNAL_HOLDOUT_FRAC_OF_TEST,
    GENERATOR_NAME,
    GENERATOR_VERSION,
    MAX_FOLLOWUPS,
    MAX_OBS_DAYS,
    MEDICATION_CLASSES,
    MIN_FOLLOWUPS,
    MIN_OBS_DAYS,
    NOISY_DIR,
    RANDOM_SEED,
    REGION,
    REPORTS_DIR,
    SES,
    SEX_PROBS,
    SMOKING,
    STAGE_LABELS,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
)
from .reference_ranges import LAB_SPECS, abnormality_flag


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def _clip(v: float, lo: float, hi: float) -> float:
    return float(np.clip(v, lo, hi))


def _choice(rng: np.random.Generator, mapping: dict[str, float]) -> str:
    keys = list(mapping.keys())
    probs = np.array([mapping[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(keys, p=probs))


def _patient_id(i: int) -> str:
    return f"SYN-P{i:06d}"


def _hash_seed(base: int, *parts: Any) -> int:
    h = hashlib.md5(f"{base}|{'|'.join(map(str, parts))}".encode()).hexdigest()
    return int(h[:8], 16)


@dataclass
class PatientLatent:
    patient_id: str
    age_at_index: float
    sex_at_birth: str
    gender: str
    ethnicity: str
    ses_group: str
    access_to_care: str
    region: str
    height_cm: float
    weight_kg: float
    bmi: float
    waist_cm: float
    smoking_status: str
    pack_years: float
    alcohol_use: str
    physical_activity: str
    diet_risk: str
    family_history: dict[str, int]
    pregnancy_status: str
    menopausal_status: str
    obs_days: int
    n_followups: int
    encounter_days: list[int]
    partition: str
    is_external_style: bool
    # susceptibility 0-1, onset day or -1 if never, severity trajectory params
    susceptibility: dict[str, float] = field(default_factory=dict)
    onset_day: dict[str, int] = field(default_factory=dict)
    max_severity: dict[str, float] = field(default_factory=dict)
    progression_rate: dict[str, float] = field(default_factory=dict)
    trajectory_type: dict[str, str] = field(default_factory=dict)
    treatment_start: dict[str, int] = field(default_factory=dict)
    adherence: dict[str, float] = field(default_factory=dict)
    # confounders / acute factors (time-varying drawn per encounter)
    base_metabolic_risk: float = 0.0
    base_cv_risk: float = 0.0
    base_inflam_risk: float = 0.0


class SyntheticGenerator:
    def __init__(self, n_patients: int = DEFAULT_N_PATIENTS, seed: int = RANDOM_SEED):
        self.n_patients = n_patients
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.patients: list[PatientLatent] = []

    # ------------------------------------------------------------------
    # 1–3: demographics, susceptibility, onset/progression
    # ------------------------------------------------------------------
    def generate_patients(self) -> None:
        # Patient-level partition assignment (no patient in multiple splits)
        n = self.n_patients
        idx = self.rng.permutation(n)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        partitions = np.array(["train"] * n)
        partitions[idx[n_train : n_train + n_val]] = "val"
        partitions[idx[n_train + n_val :]] = "test"
        test_idx = np.where(partitions == "test")[0]
        n_ext = max(1, int(len(test_idx) * EXTERNAL_HOLDOUT_FRAC_OF_TEST))
        external_set = set(test_idx[:n_ext].tolist())

        for i in range(n):
            pid = _patient_id(i)
            sex = _choice(self.rng, SEX_PROBS)
            # Gender mostly aligns with sex_at_birth; small non-cis fraction
            if self.rng.random() < 0.015:
                gender = "non_binary" if self.rng.random() < 0.4 else ("male" if sex == "female" else "female")
            else:
                gender = sex

            # Age: mix of middle-aged chronic disease + older cognition cohort
            if self.rng.random() < 0.20:
                age = float(self.rng.normal(72, 8))
            else:
                age = float(self.rng.normal(55, 14))
            age = _clip(age, 18, 95)

            is_ext = i in external_set
            # External-style: shifted demographics / access / assay (flag only; noise applied later)
            eth_map = ETHNICITY
            if is_ext:
                eth_map = {
                    "White": 0.35,
                    "Black_or_African_American": 0.25,
                    "Hispanic_or_Latino": 0.25,
                    "Asian": 0.08,
                    "Other_or_Multiple": 0.07,
                }
            ethnicity = _choice(self.rng, eth_map)
            ses = _choice(self.rng, SES)
            access = _choice(self.rng, ACCESS if not is_ext else {"limited": 0.40, "standard": 0.45, "enhanced": 0.15})
            region = _choice(self.rng, REGION)

            height = float(self.rng.normal(162 if sex == "female" else 176, 7))
            height = _clip(height, 140, 205)
            # BMI distribution with right skew
            bmi = float(np.exp(self.rng.normal(np.log(27.5), 0.22)))
            bmi = _clip(bmi, 16, 55)
            weight = bmi * (height / 100) ** 2
            waist = float(0.55 * height + 0.9 * (bmi - 22) + self.rng.normal(0, 4))
            waist = _clip(waist, 55, 160)

            smoking = _choice(self.rng, SMOKING)
            if smoking == "never":
                pack_years = 0.0
            elif smoking == "former":
                pack_years = float(_clip(self.rng.lognormal(2.2, 0.7), 0.5, 80))
            else:
                pack_years = float(_clip(self.rng.lognormal(2.8, 0.6), 1, 100))

            alcohol = _choice(self.rng, ALCOHOL)
            activity = _choice(self.rng, ACTIVITY)
            diet = _choice(self.rng, DIET_RISK)

            # Family history (probabilistic, not deterministic disease)
            fh = {}
            for d in DISEASES:
                base = {
                    "t2dm": 0.25, "ckd": 0.10, "cad": 0.22, "hf": 0.12, "htn": 0.35,
                    "copd": 0.10, "cld_masld": 0.12, "ra": 0.08, "hypothyroid": 0.15,
                    "alzheimers": 0.18,
                }[d]
                fh[d] = int(self.rng.random() < base)

            pregnancy = "not_applicable"
            menopause = "not_applicable"
            if sex == "female":
                if 18 <= age <= 45 and self.rng.random() < 0.03:
                    pregnancy = "pregnant"
                elif age < 45:
                    menopause = "premenopausal"
                elif age < 55:
                    menopause = "perimenopausal" if self.rng.random() < 0.5 else "postmenopausal"
                else:
                    menopause = "postmenopausal"

            obs_days = int(self.rng.integers(MIN_OBS_DAYS, MAX_OBS_DAYS + 1))
            # Access affects follow-up count
            access_mult = {"limited": 0.6, "standard": 1.0, "enhanced": 1.25}[access]
            max_fu = max(MIN_FOLLOWUPS, int(MAX_FOLLOWUPS * access_mult))
            n_fu = int(self.rng.integers(MIN_FOLLOWUPS, max_fu + 1))
            # Irregular encounter days
            if n_fu == 0:
                enc_days = [0]
            else:
                raw = np.sort(self.rng.uniform(30, obs_days, size=n_fu))
                enc_days = [0] + [int(x) for x in raw]

            # Shared risk scores
            met = 0.0
            met += 0.04 * max(0, bmi - 25)
            met += {"sedentary": 0.15, "moderate": 0.05, "active": 0.0}[activity]
            met += {"high": 0.15, "moderate": 0.05, "low": 0.0}[diet]
            met += 0.08 * fh.get("t2dm", 0)
            met += 0.01 * max(0, age - 40)
            met += self.rng.normal(0, 0.08)

            cv = 0.0
            cv += 0.012 * max(0, age - 35)
            cv += {"current": 0.20, "former": 0.08, "never": 0.0}[smoking]
            cv += 0.03 * max(0, bmi - 27)
            cv += 0.10 * fh.get("cad", 0)
            cv += 0.08 * fh.get("htn", 0)
            cv += self.rng.normal(0, 0.08)

            inflam = self.rng.normal(0.1, 0.1) + 0.15 * fh.get("ra", 0)

            # Susceptibility with comorbidity coupling
            sus = {d: float(BASE_LIFETIME_RISK[d]) for d in DISEASES}
            # Age effects
            sus["alzheimers"] *= float(1 / (1 + np.exp(-(age - 72) / 6))) * 8  # strong age skew
            sus["alzheimers"] = min(0.45, sus["alzheimers"])
            sus["hf"] *= 1 + 0.03 * max(0, age - 50)
            sus["ckd"] *= 1 + 0.02 * max(0, age - 45)
            sus["cad"] *= 1 + 0.025 * max(0, age - 40)
            sus["htn"] *= 1 + 0.015 * max(0, age - 30)
            sus["copd"] *= 1 + 0.8 * (pack_years / 40) + (0.5 if smoking == "current" else 0)
            sus["t2dm"] *= 1 + 1.2 * max(0, met)
            sus["cld_masld"] *= 1 + 1.0 * max(0, met) + (0.4 if alcohol == "heavy" else 0)
            sus["htn"] *= 1 + 0.6 * max(0, met) + 0.5 * max(0, cv)
            sus["cad"] *= 1 + 0.8 * max(0, cv)
            sus["hypothyroid"] *= 1.4 if sex == "female" else 0.7
            sus["ra"] *= 1.5 if sex == "female" else 0.8
            for d in DISEASES:
                sus[d] *= 1 + 0.35 * fh.get(d, 0)
                sus[d] = float(np.clip(sus[d], 0.001, 0.85))

            # Soft comorbidity boosts (not deterministic labels)
            for a, b, strength in COMORBIDITY_BOOST:
                shared = np.sqrt(sus[a] * sus[b])
                sus[a] = float(np.clip(sus[a] + strength * 0.35 * shared, 0, 0.92))
                sus[b] = float(np.clip(sus[b] + strength * 0.35 * shared, 0, 0.92))
            # Conditional onset coupling: if A onset, raise B residual susceptibility further
            # (applied after onset draws would be circular; instead correlate max severity later)

            # SES / access affect detection later; mild effect on true risk via environment
            if ses == "low":
                for d in ("t2dm", "htn", "copd", "ckd"):
                    sus[d] = float(np.clip(sus[d] * 1.08, 0, 0.9))

            onset: dict[str, int] = {}
            max_sev: dict[str, float] = {}
            prog: dict[str, float] = {}
            traj: dict[str, str] = {}
            tx_start: dict[str, int] = {}
            adh: dict[str, float] = {}

            for d in DISEASES:
                p_onset = sus[d]
                # Healthy controls / never disease
                if self.rng.random() > p_onset:
                    onset[d] = -1
                    max_sev[d] = float(self.rng.uniform(0, 0.15))  # residual risk signal
                    prog[d] = 0.0
                    traj[d] = "stable_low_risk"
                    tx_start[d] = -1
                    adh[d] = 0.0
                    continue

                # At-risk only (never crosses diagnostic threshold) ~15% of susceptibles
                if self.rng.random() < 0.15:
                    onset[d] = -1
                    max_sev[d] = float(self.rng.uniform(0.15, 0.35))
                    prog[d] = float(self.rng.uniform(0.0, 0.02))
                    traj[d] = "gradually_increasing_risk"
                    tx_start[d] = -1
                    adh[d] = 0.0
                    continue

                # Onset during observation or historical
                if self.rng.random() < 0.55:
                    # historical disease before index
                    onset[d] = int(self.rng.integers(-2000, -30))
                else:
                    onset[d] = int(self.rng.integers(0, max(1, obs_days - 60)))

                max_sev[d] = float(self.rng.beta(2.5, 1.8))  # slightly higher typical severity
                traj_types = [
                    "pre_to_confirmed",
                    "controlled",
                    "poorly_controlled",
                    "rapid_progression",
                    "relapse_exacerbation",
                    "treatment_response",
                    "treatment_failure",
                ]
                weights = np.array([0.15, 0.20, 0.15, 0.08, 0.12, 0.18, 0.12])
                traj[d] = str(self.rng.choice(traj_types, p=weights / weights.sum()))
                base_rate = {
                    "pre_to_confirmed": 0.05,
                    "controlled": 0.02,
                    "poorly_controlled": 0.06,
                    "rapid_progression": 0.10,
                    "relapse_exacerbation": 0.05,
                    "treatment_response": 0.04,
                    "treatment_failure": 0.07,
                }[traj[d]]
                prog[d] = float(base_rate * self.rng.uniform(0.7, 1.3))

                # Treatment start after onset (if care access allows)
                if onset[d] < obs_days:
                    delay = int(self.rng.lognormal(3.5, 0.7))  # days from onset to treatment
                    if access == "limited":
                        delay = int(delay * 1.6)
                    t0 = max(onset[d], 0) + delay
                    treat_p = {"limited": 0.55, "standard": 0.78, "enhanced": 0.88}[access]
                    if t0 < obs_days and self.rng.random() < treat_p:
                        tx_start[d] = min(t0, obs_days - 1)
                        adh[d] = float(_clip(self.rng.beta(5, 2) if ses != "low" else self.rng.beta(3, 3), 0.1, 1.0))
                        if traj[d] in ("poorly_controlled", "treatment_failure"):
                            adh[d] *= 0.6
                    else:
                        tx_start[d] = -1
                        adh[d] = 0.0
                else:
                    tx_start[d] = -1
                    adh[d] = 0.0

            # Post-onset multimorbidity: raise partner disease max severity when both present
            for a, b, strength in COMORBIDITY_BOOST:
                if onset[a] != -1 and onset[b] != -1:
                    max_sev[a] = float(np.clip(max_sev[a] + 0.08 * strength, 0, 1))
                    max_sev[b] = float(np.clip(max_sev[b] + 0.08 * strength, 0, 1))
                elif onset[a] != -1 and onset[b] == -1 and self.rng.random() < 0.12 * strength:
                    # Secondary onset of partner disease
                    onset[b] = max(onset[a], 0) + int(self.rng.integers(60, 800))
                    if onset[b] > obs_days:
                        onset[b] = int(self.rng.integers(0, max(1, obs_days // 2)))
                    max_sev[b] = float(np.clip(0.3 + 0.4 * strength + self.rng.random() * 0.3, 0.2, 1))
                    prog[b] = float(0.04 * self.rng.uniform(0.7, 1.3))
                    traj[b] = "pre_to_confirmed"
                    if self.rng.random() < 0.7:
                        tx_start[b] = min(obs_days - 1, max(onset[b], 0) + int(self.rng.integers(30, 200)))
                        adh[b] = float(_clip(self.rng.beta(4, 2), 0.2, 1.0))

            pl = PatientLatent(
                patient_id=pid,
                age_at_index=round(age, 1),
                sex_at_birth=sex,
                gender=gender,
                ethnicity=ethnicity,
                ses_group=ses,
                access_to_care=access,
                region=region,
                height_cm=round(height, 1),
                weight_kg=round(weight, 1),
                bmi=round(bmi, 1),
                waist_cm=round(waist, 1),
                smoking_status=smoking,
                pack_years=round(pack_years, 1),
                alcohol_use=alcohol,
                physical_activity=activity,
                diet_risk=diet,
                family_history=fh,
                pregnancy_status=pregnancy,
                menopausal_status=menopause,
                obs_days=obs_days,
                n_followups=n_fu,
                encounter_days=enc_days,
                partition=partitions[i],
                is_external_style=is_ext,
                susceptibility=sus,
                onset_day=onset,
                max_severity=max_sev,
                progression_rate=prog,
                trajectory_type=traj,
                treatment_start=tx_start,
                adherence=adh,
                base_metabolic_risk=float(met),
                base_cv_risk=float(cv),
                base_inflam_risk=float(inflam),
            )
            self.patients.append(pl)

    def severity_at(self, p: PatientLatent, disease: str, day: int) -> float:
        """Latent severity in [0,1] at day relative to index.

        Stochastic terms are seeded by (patient, disease, day) so repeated
        calls for the same state are consistent (important for targets).
        """
        local = np.random.default_rng(_hash_seed(self.seed, p.patient_id, disease, int(day)))
        onset = p.onset_day[disease]
        max_s = p.max_severity[disease]
        rate = p.progression_rate[disease]
        traj = p.trajectory_type[disease]

        if onset < 0 and traj in ("stable_low_risk", "gradually_increasing_risk"):
            t = max(0, day)
            return float(np.clip(
                max_s * (0.5 + 0.5 * (1 - np.exp(-rate * t / 100))) + local.normal(0, 0.01),
                0, 0.45,
            ))

        if onset < 0 or day < onset:
            return float(np.clip(
                0.05 + 0.2 * p.susceptibility[disease] + local.normal(0, 0.02),
                0, 0.35,
            ))

        t = day - onset
        raw = max_s * (1 - np.exp(-rate * t / 30))
        if traj == "rapid_progression":
            raw = min(1.0, raw * 1.4)
        if traj == "relapse_exacerbation":
            # Deterministic spike schedule from local RNG
            if local.random() < 0.08:
                raw = min(1.0, raw + 0.25)

        tx = p.treatment_start[disease]
        if tx >= 0 and day >= tx:
            adh = p.adherence[disease]
            if traj in ("controlled", "treatment_response"):
                effect = 0.45 * adh
            elif traj == "treatment_failure":
                effect = 0.10 * adh
            elif traj == "poorly_controlled":
                effect = 0.15 * adh
            else:
                effect = 0.30 * adh
            potency = {
                "t2dm": 0.9, "htn": 0.85, "ckd": 0.4, "cad": 0.35, "hf": 0.5,
                "copd": 0.45, "cld_masld": 0.25, "ra": 0.7, "hypothyroid": 0.9,
                "alzheimers": 0.15,
            }[disease]
            raw = raw * (1 - effect * potency)

        return float(np.clip(raw + local.normal(0, 0.02), 0, 1))

    def stage_from_severity(self, disease: str, sev: float, present: bool) -> str:
        labels = STAGE_LABELS[disease]
        if not present and sev < 0.25:
            return labels[0]
        if not present:
            return labels[1] if len(labels) > 1 else labels[0]
        # Map severity to stages 1..4 of disease stages
        bins = [0.25, 0.45, 0.65, 0.85]
        idx = 1
        for b in bins:
            if sev >= b:
                idx += 1
        idx = min(idx, len(labels) - 1)
        return labels[idx]

    # ------------------------------------------------------------------
    # 4–7: encounters, labs, diagnoses, meds
    # ------------------------------------------------------------------
    def build_tables(self) -> dict[str, pd.DataFrame]:
        patient_rows = []
        encounter_rows = []
        lab_rows = []
        dx_rows = []
        med_rows = []
        target_rows = []

        lab_id = 0
        med_id = 0
        dx_id = 0

        for p in self.patients:
            # Patient table
            row = {
                "is_synthetic": True,
                "dataset_disclaimer": "SYNTHETIC_RESEARCH_ONLY",
                "generator_name": GENERATOR_NAME,
                "generator_version": GENERATOR_VERSION,
                "random_seed": self.seed,
                "patient_id": p.patient_id,
                "age_at_index": p.age_at_index,
                "sex_at_birth": p.sex_at_birth,
                "gender": p.gender,
                "ethnicity": p.ethnicity,
                "ses_group": p.ses_group,
                "access_to_care": p.access_to_care,
                "region": p.region,
                "height_cm": p.height_cm,
                "weight_kg_baseline": p.weight_kg,
                "bmi_baseline": p.bmi,
                "waist_cm_baseline": p.waist_cm,
                "smoking_status": p.smoking_status,
                "pack_years": p.pack_years,
                "alcohol_use": p.alcohol_use,
                "physical_activity": p.physical_activity,
                "diet_risk": p.diet_risk,
                "pregnancy_status_baseline": p.pregnancy_status,
                "menopausal_status": p.menopausal_status,
                "index_date_relative": 0,
                "observation_end_day": p.obs_days,
                "n_encounters": len(p.encounter_days),
                "partition": p.partition,
                "is_external_style_holdout": int(p.is_external_style),
                "base_metabolic_risk": round(p.base_metabolic_risk, 3),
                "base_cv_risk": round(p.base_cv_risk, 3),
                "base_inflam_risk": round(p.base_inflam_risk, 3),
            }
            for d in DISEASES:
                row[f"fh_{d}"] = p.family_history[d]
                row[f"susceptibility_{d}"] = round(p.susceptibility[d], 4)
                row[f"latent_onset_day_{d}"] = p.onset_day[d]
                row[f"trajectory_{d}"] = p.trajectory_type[d]
                row[f"treatment_start_day_{d}"] = p.treatment_start[d]
                row[f"adherence_{d}"] = round(p.adherence[d], 3)
            patient_rows.append(row)

            # Precompute severity path for diagnoses / targets
            # Also track first day evidence supports clinical diagnosis
            first_clinical_dx_day: dict[str, int] = {d: -1 for d in DISEASES}
            meds_started: set[str] = set()

            for ei, day in enumerate(p.encounter_days):
                enc_id = f"{p.patient_id}-E{ei:02d}"
                age = p.age_at_index + day / 365.25

                # Time-varying acute confounders
                acute_infection = int(self.rng.random() < 0.06)
                dehydration = int(self.rng.random() < 0.04)
                strenuous_exercise = int(self.rng.random() < 0.05)
                recent_surgery = int(self.rng.random() < 0.02)
                hospitalization_recent = int(self.rng.random() < 0.03)
                acute_illness = int(acute_infection or dehydration or recent_surgery)

                sevs = {d: self.severity_at(p, d, day) for d in DISEASES}
                # Present after onset; include milder true disease (normal-ish biomarkers possible)
                latent_present = {
                    d: int(p.onset_day[d] >= 0 and day >= p.onset_day[d] and sevs[d] >= 0.12)
                    for d in DISEASES
                }

                stages = {
                    d: self.stage_from_severity(d, sevs[d], bool(latent_present[d]))
                    for d in DISEASES
                }

                # Vitals from latent states + noise
                sbp = 118 + 25 * sevs["htn"] + 8 * sevs["ckd"] + 5 * sevs["t2dm"] + self.rng.normal(0, 8)
                dbp = 75 + 12 * sevs["htn"] + self.rng.normal(0, 5)
                if p.treatment_start["htn"] >= 0 and day >= p.treatment_start["htn"]:
                    sbp -= 12 * p.adherence["htn"]
                    dbp -= 6 * p.adherence["htn"]
                hr = 72 + 10 * sevs["hf"] + 5 * sevs["copd"] + (8 if acute_infection else 0) + self.rng.normal(0, 6)
                spo2 = 98 - 8 * sevs["copd"] - 3 * sevs["hf"] + self.rng.normal(0, 0.8)
                weight = p.weight_kg + self.rng.normal(0, 1.2) + 3 * sevs["hf"] * (1 if sevs["hf"] > 0.4 else 0)
                bmi = weight / (p.height_cm / 100) ** 2

                # Symptoms
                polyuria = int(self.rng.random() < 0.1 + 0.5 * sevs["t2dm"])
                polydipsia = int(self.rng.random() < 0.08 + 0.45 * sevs["t2dm"])
                fatigue = int(self.rng.random() < 0.15 + 0.3 * sevs["t2dm"] + 0.25 * sevs["hypothyroid"] + 0.2 * sevs["hf"])
                chest_pain = int(self.rng.random() < 0.05 + 0.4 * sevs["cad"])
                dyspnea = int(self.rng.random() < 0.05 + 0.55 * sevs["hf"] + 0.5 * sevs["copd"])
                edema = int(self.rng.random() < 0.03 + 0.45 * sevs["hf"] + 0.2 * sevs["ckd"])
                cough = int(self.rng.random() < 0.08 + 0.5 * sevs["copd"] + 0.2 * acute_infection)
                joint_pain = int(self.rng.random() < 0.05 + 0.7 * sevs["ra"])
                joint_swelling = int(self.rng.random() < 0.02 + 0.55 * sevs["ra"])
                cognitive_complaint = int(self.rng.random() < 0.05 + 0.6 * sevs["alzheimers"] + 0.01 * max(0, age - 65))
                cold_intolerance = int(self.rng.random() < 0.04 + 0.5 * sevs["hypothyroid"])

                # Biomarkers (latent + confounders + assay noise)
                labs = self._labs_for_state(
                    p, day, sevs, acute_infection, dehydration, strenuous_exercise, age
                )

                # Clinical diagnosis rule: requires evidence available by this day;
                # probabilistic detection lag; not deterministic from single biomarker.
                for d in DISEASES:
                    if first_clinical_dx_day[d] >= 0:
                        continue
                    if not latent_present[d]:
                        # False positive diagnosis risk low (coding noise handled in noisy)
                        continue
                    evidence = self._diagnosis_evidence(d, labs, sevs[d], sbp, stages[d],
                                                        polyuria, chest_pain, dyspnea,
                                                        joint_swelling, cognitive_complaint)
                    detect_p = min(0.95, 0.20 + evidence * {"limited": 0.55, "standard": 0.85, "enhanced": 0.95}[p.access_to_care])
                    # Delay: fewer detections very early after onset
                    if p.onset_day[d] >= 0 and day - max(p.onset_day[d], 0) < 30:
                        detect_p *= 0.5
                    if self.rng.random() < detect_p:
                        first_clinical_dx_day[d] = day
                        dx_id += 1
                        dx_rows.append({
                            "is_synthetic": True,
                            "dataset_disclaimer": "SYNTHETIC_RESEARCH_ONLY",
                            "diagnosis_id": f"SYN-DX{dx_id:07d}",
                            "patient_id": p.patient_id,
                            "encounter_id": enc_id,
                            "days_from_index": day,
                            "disease_code": d,
                            "disease_label": DISEASE_LABELS[d],
                            "diagnosis_type": "incident" if p.onset_day[d] >= 0 else "historical",
                            "clinical_stage_at_diagnosis": stages[d],
                            "latent_severity_at_diagnosis": round(sevs[d], 3),
                            "evidence_score": round(evidence, 3),
                            "is_true_latent_disease": 1,
                            "is_observed_diagnosis": 1,
                        })

                # Medications initiated
                meds_today = self._medications_for_state(p, day, sevs, first_clinical_dx_day, meds_started)
                for m in meds_today:
                    med_id += 1
                    med_rows.append({
                        "is_synthetic": True,
                        "dataset_disclaimer": "SYNTHETIC_RESEARCH_ONLY",
                        "medication_event_id": f"SYN-MED{med_id:07d}",
                        "patient_id": p.patient_id,
                        "encounter_id": enc_id,
                        "days_from_index": day,
                        "medication_class": m["class"],
                        "indication_disease": m["indication"],
                        "start_day": day,
                        "stop_day": m.get("stop_day", -1),
                        "adherence_estimate": m["adherence"],
                        "is_active": 1,
                    })
                    meds_started.add(m["class"] + "|" + m["indication"])

                # Lab rows with ordering behavior
                ordered = self._order_labs(p, sevs, labs, acute_illness, first_clinical_dx_day)
                for analyte, val in labs.items():
                    if analyte not in ordered:
                        continue
                    lab_id += 1
                    spec = LAB_SPECS[analyte]
                    fasting = None
                    if spec.fasting_relevant:
                        fasting = int(self.rng.random() < 0.75)
                    processing_hours = float(_clip(self.rng.lognormal(1.0, 0.5), 0.5, 48))
                    lab_rows.append({
                        "is_synthetic": True,
                        "dataset_disclaimer": "SYNTHETIC_RESEARCH_ONLY",
                        "lab_result_id": f"SYN-LAB{lab_id:08d}",
                        "patient_id": p.patient_id,
                        "encounter_id": enc_id,
                        "days_from_index": day,
                        "analyte": analyte,
                        "analyte_name": spec.analyte,
                        "value": round(float(val), 3 if analyte in ("fev1_fvc_ratio", "inr") else 2),
                        "unit": spec.unit,
                        "unit_standardized": spec.unit,
                        "ref_low": spec.ref_low,
                        "ref_high": spec.ref_high,
                        "abnormality_flag": abnormality_flag(analyte, float(val), p.sex_at_birth),
                        "specimen_type": spec.specimen,
                        "fasting_status": fasting,
                        "hours_collection_to_processing": round(processing_hours, 1),
                        "assay_category": "standard_immunoassay_or_chemistry",
                        "loinc_like_code": spec.loinc_like,
                    })

                # Encounter row
                enc = {
                    "is_synthetic": True,
                    "dataset_disclaimer": "SYNTHETIC_RESEARCH_ONLY",
                    "encounter_id": enc_id,
                    "patient_id": p.patient_id,
                    "encounter_number": ei,
                    "days_from_index": day,
                    "is_baseline": int(ei == 0),
                    "age_at_encounter": round(age, 1),
                    "encounter_type": self.rng.choice(
                        ["outpatient", "outpatient", "outpatient", "urgent_care", "inpatient"],
                    ),
                    "weight_kg": round(weight, 1),
                    "bmi": round(bmi, 1),
                    "sbp_mmHg": round(_clip(sbp, 80, 230), 0),
                    "dbp_mmHg": round(_clip(dbp, 40, 130), 0),
                    "heart_rate_bpm": round(_clip(hr, 40, 160), 0),
                    "spo2_percent": round(_clip(spo2, 70, 100), 0),
                    "symptom_polyuria": polyuria,
                    "symptom_polydipsia": polydipsia,
                    "symptom_fatigue": fatigue,
                    "symptom_chest_pain": chest_pain,
                    "symptom_dyspnea": dyspnea,
                    "symptom_edema": edema,
                    "symptom_cough": cough,
                    "symptom_joint_pain": joint_pain,
                    "symptom_joint_swelling": joint_swelling,
                    "symptom_cognitive_complaint": cognitive_complaint,
                    "symptom_cold_intolerance": cold_intolerance,
                    "acute_infection": acute_infection,
                    "dehydration": dehydration,
                    "strenuous_exercise": strenuous_exercise,
                    "recent_surgery": recent_surgery,
                    "recent_hospitalization": hospitalization_recent,
                    "acute_illness": acute_illness,
                    "partition": p.partition,
                }
                for d in DISEASES:
                    enc[f"latent_severity_{d}"] = round(sevs[d], 3)
                    enc[f"latent_disease_present_{d}"] = latent_present[d]
                    enc[f"latent_stage_{d}"] = stages[d]
                    enc[f"observed_diagnosis_{d}"] = int(
                        first_clinical_dx_day[d] >= 0 and day >= first_clinical_dx_day[d]
                    )
                encounter_rows.append(enc)

                # Prediction targets at this encounter (feature cutoff = this day)
                trow = self._targets_for_encounter(
                    p, enc_id, day, sevs, latent_present, stages,
                    first_clinical_dx_day, labs, sbp,
                )
                target_rows.append(trow)

            # Loss to follow-up flag on patient
            patient_rows[-1]["loss_to_followup"] = int(
                p.access_to_care == "limited" and p.n_followups <= 1 and self.rng.random() < 0.5
            )
            # Mortality (rare, age/disease linked) — synthetic only
            mort_risk = 0.01 + 0.0008 * max(0, p.age_at_index - 50)
            for d in DISEASES:
                if p.onset_day[d] >= 0:
                    mort_risk += 0.01 * p.max_severity[d]
            mort_risk += 0.02 * max(sevs.get("hf", 0), sevs.get("ckd", 0), sevs.get("copd", 0)) if encounter_rows else 0
            patient_rows[-1]["death_within_5y_latent"] = int(self.rng.random() < min(0.35, mort_risk))
            if patient_rows[-1]["death_within_5y_latent"]:
                patient_rows[-1]["death_day"] = int(self.rng.integers(30, min(p.obs_days + 365, 1825)))
            else:
                patient_rows[-1]["death_day"] = -1

        tables = {
            "patients": pd.DataFrame(patient_rows),
            "encounters": pd.DataFrame(encounter_rows),
            "labs": pd.DataFrame(lab_rows),
            "diagnoses": pd.DataFrame(dx_rows),
            "medications": pd.DataFrame(med_rows),
            "prediction_targets": pd.DataFrame(target_rows),
        }
        return tables

    def _labs_for_state(
        self,
        p: PatientLatent,
        day: int,
        sevs: dict[str, float],
        acute_infection: int,
        dehydration: int,
        strenuous_exercise: int,
        age: float,
    ) -> dict[str, float]:
        sex = p.sex_at_birth
        met = p.base_metabolic_risk
        # Glycemia
        fpg = 88 + 90 * sevs["t2dm"] + 18 * met + self.rng.normal(0, 8)
        if p.treatment_start["t2dm"] >= 0 and day >= p.treatment_start["t2dm"]:
            fpg -= 45 * p.adherence["t2dm"]
        if acute_infection:
            fpg += 15
        if p.pregnancy_status == "pregnant":
            fpg += 5
        hba1c = 5.2 + 4.5 * sevs["t2dm"] + 0.35 * met + self.rng.normal(0, 0.15)
        if p.treatment_start["t2dm"] >= 0 and day >= p.treatment_start["t2dm"]:
            hba1c -= 1.6 * p.adherence["t2dm"]
        fpg = _clip(fpg, 55, 450)
        hba1c = _clip(hba1c, 4.0, 15.0)
        rpg = _clip(fpg + self.rng.uniform(10, 50), 55, 500)
        fasting_insulin = _clip(8 + 18 * met + 10 * sevs["t2dm"] + self.rng.normal(0, 3), 1, 80)
        cpep = _clip(1.5 + 1.2 * met + self.rng.normal(0, 0.3), 0.2, 8)

        # Lipids
        tg = _clip(100 + 120 * met + 40 * sevs["t2dm"] + 30 * sevs["cld_masld"] + self.rng.normal(0, 25), 40, 800)
        hdl = _clip(55 - 12 * met - 8 * sevs["t2dm"] + (5 if sex == "female" else 0) + self.rng.normal(0, 6), 20, 100)
        ldl = _clip(110 + 20 * p.base_cv_risk - (25 if False else 0) + self.rng.normal(0, 15), 40, 280)
        if p.treatment_start["cad"] >= 0 and day >= p.treatment_start["cad"]:
            ldl -= 35 * p.adherence.get("cad", 0.5)
        # Statin effect if on statin via htn/cad trajectory
        if p.treatment_start["htn"] >= 0 and day >= p.treatment_start["htn"] and self.rng.random() < 0.3:
            ldl -= 20
        tc = ldl + hdl + tg / 5 + self.rng.normal(0, 5)
        apo_b = _clip(0.9 * ldl + self.rng.normal(0, 8), 40, 200)
        lp_a = float(np.exp(self.rng.normal(3.2, 0.9)))  # largely genetic
        lp_a = _clip(lp_a, 5, 300)

        # Kidney
        base_cr = 0.75 if sex == "female" else 0.95
        cr = base_cr + 1.8 * sevs["ckd"] + 0.3 * sevs["hf"] + 0.15 * dehydration + self.rng.normal(0, 0.08)
        if acute_infection and dehydration:
            cr += 0.25  # temporary AKI-like without CKD diagnosis
        cr = _clip(cr, 0.3, 8.0)
        # Simplified eGFR (not full CKD-EPI; synthetic approximation)
        egfr = 140 - 1.1 * (age - 20) - 70 * max(0, cr - 0.8) - 25 * sevs["ckd"] + self.rng.normal(0, 5)
        egfr = _clip(egfr, 5, 140)
        bun = _clip(12 + 40 * sevs["ckd"] + 8 * dehydration + self.rng.normal(0, 2), 4, 120)
        uacr = _clip(np.exp(self.rng.normal(2.0, 0.6)) + 400 * sevs["ckd"] ** 1.5 + 80 * sevs["t2dm"], 2, 3500)
        ualb = _clip(uacr / 20 + self.rng.normal(0, 0.5), 0, 500)
        ucr = float(self.rng.uniform(40, 200))
        na = _clip(140 - 3 * sevs["hf"] + self.rng.normal(0, 1.5), 120, 155)
        k = _clip(4.1 + 0.6 * sevs["ckd"] + self.rng.normal(0, 0.25), 2.5, 7.0)
        # ACEI/ARB / diuretic effects
        if p.treatment_start["htn"] >= 0 and day >= p.treatment_start["htn"]:
            k += 0.2 * p.adherence["htn"]
            cr += 0.05
        if p.treatment_start["hf"] >= 0 and day >= p.treatment_start["hf"]:
            na -= 1.5 * p.adherence["hf"]
        hco3 = _clip(25 - 4 * sevs["ckd"] + self.rng.normal(0, 1.2), 8, 35)
        po4 = _clip(3.5 + 1.5 * max(0, sevs["ckd"] - 0.5) + self.rng.normal(0, 0.3), 1.5, 9)
        ca = _clip(9.2 - 0.8 * max(0, sevs["ckd"] - 0.5) + self.rng.normal(0, 0.25), 6.5, 12)
        hgb = (13.5 if sex == "female" else 15.0) - 3.5 * sevs["ckd"] - 1.5 * sevs["ra"] - 0.8 * sevs["hf"]
        hgb += self.rng.normal(0, 0.4)
        hgb = _clip(hgb, 6, 19)

        # CV markers
        hs_crp = _clip(np.exp(self.rng.normal(0.5, 0.6)) + 5 * sevs["ra"] + 2 * sevs["cad"] + 4 * acute_infection, 0.1, 80)
        crp = hs_crp * (1.2 if acute_infection else 1.0)
        esr = _clip(10 + 40 * sevs["ra"] + 15 * acute_infection + self.rng.normal(0, 5), 1, 120)
        ntprobnp = _clip(50 + 2500 * sevs["hf"] ** 1.3 + 200 * sevs["ckd"] + 5 * max(0, age - 50) + self.rng.normal(0, 40), 10, 30000)
        # Troponin mainly if acute context
        trop = _clip(3 + 80 * sevs["cad"] * (1 if sevs["cad"] > 0.6 else 0.1) + self.rng.normal(0, 2), 0, 500)
        if strenuous_exercise:
            trop += 8

        # Liver
        alt = _clip(22 + 60 * sevs["cld_masld"] + 15 * (p.alcohol_use == "heavy") + self.rng.normal(0, 6), 5, 400)
        ast = _clip(alt * (0.9 + 0.3 * (p.alcohol_use == "heavy")) + self.rng.normal(0, 5), 5, 400)
        if strenuous_exercise:
            ast += 20
        alp = _clip(70 + 40 * max(0, sevs["cld_masld"] - 0.5) + self.rng.normal(0, 10), 30, 400)
        ggt = _clip(25 + 50 * sevs["cld_masld"] + 40 * (p.alcohol_use == "heavy") + self.rng.normal(0, 8), 5, 500)
        tbili = _clip(0.6 + 2.5 * max(0, sevs["cld_masld"] - 0.6) + self.rng.normal(0, 0.15), 0.1, 15)
        dbili = _clip(0.15 + 0.6 * max(0, sevs["cld_masld"] - 0.6) + self.rng.normal(0, 0.05), 0.0, 10)
        alb = _clip(4.2 - 1.5 * max(0, sevs["cld_masld"] - 0.55) - 0.8 * sevs["hf"] + self.rng.normal(0, 0.15), 1.5, 5.3)
        inr = _clip(1.0 + 0.8 * max(0, sevs["cld_masld"] - 0.65) + self.rng.normal(0, 0.05), 0.8, 4)
        plt = _clip(250 - 120 * max(0, sevs["cld_masld"] - 0.5) - 30 * sevs["ra"] + self.rng.normal(0, 20), 30, 550)
        wbc = _clip(7 + 4 * acute_infection + self.rng.normal(0, 1.2), 2, 25)
        if p.treatment_start.get("ra", -1) >= 0 and day >= p.treatment_start["ra"]:
            # MTX / immuno can lower WBC slightly
            wbc -= 0.5
        eos = _clip(150 + 200 * sevs["copd"] * self.rng.random() + self.rng.normal(0, 40), 0, 1500)

        # Pulmonary
        fev1 = _clip(95 - 55 * sevs["copd"] - 5 * (p.pack_years / 40) + self.rng.normal(0, 5), 15, 120)
        fvc = _clip(95 - 30 * sevs["copd"] + self.rng.normal(0, 5), 25, 120)
        ratio = _clip(0.80 - 0.25 * sevs["copd"] + self.rng.normal(0, 0.03), 0.25, 0.95)
        spo2 = _clip(98 - 8 * sevs["copd"] - 3 * sevs["hf"] + self.rng.normal(0, 0.7), 70, 100)

        # Autoimmune
        rf = _clip(8 + 80 * sevs["ra"] + self.rng.normal(0, 5), 0, 300)
        # Seronegative RA possible
        if sevs["ra"] > 0.4 and self.rng.random() < 0.25:
            rf = float(self.rng.uniform(5, 14))
        ccp = _clip(10 + 120 * sevs["ra"] + self.rng.normal(0, 8), 0, 400)
        if sevs["ra"] > 0.4 and self.rng.random() < 0.20:
            ccp = float(self.rng.uniform(5, 19))  # seronegative pattern

        # Thyroid
        tsh = _clip(1.8 + 12 * sevs["hypothyroid"] + self.rng.normal(0, 0.3), 0.01, 80)
        if p.treatment_start["hypothyroid"] >= 0 and day >= p.treatment_start["hypothyroid"]:
            tsh = _clip(tsh - 10 * p.adherence["hypothyroid"] + self.rng.normal(0, 0.4), 0.05, 40)
        ft4 = _clip(1.2 - 0.7 * sevs["hypothyroid"] + self.rng.normal(0, 0.1), 0.2, 3)
        if p.treatment_start["hypothyroid"] >= 0 and day >= p.treatment_start["hypothyroid"]:
            ft4 = _clip(ft4 + 0.5 * p.adherence["hypothyroid"], 0.2, 3)
        ft3 = _clip(3.1 - 1.0 * sevs["hypothyroid"] + self.rng.normal(0, 0.2), 0.5, 6)
        tpo = _clip(15 + 150 * sevs["hypothyroid"] * (0.7 + 0.3 * self.rng.random()) + self.rng.normal(0, 10), 0, 1000)

        # Cognitive (lower worse for MMSE/MoCA)
        mmse = _clip(29 - 12 * sevs["alzheimers"] - 0.03 * max(0, age - 70) + self.rng.normal(0, 1.0), 0, 30)
        moca = _clip(27 - 14 * sevs["alzheimers"] - 0.04 * max(0, age - 70) + self.rng.normal(0, 1.2), 0, 30)
        faq = _clip(1 + 20 * sevs["alzheimers"] + self.rng.normal(0, 1.5), 0, 30)

        # Corticosteroid effect on glucose
        if sevs["ra"] > 0.5 and p.treatment_start.get("ra", -1) >= 0 and day >= p.treatment_start["ra"]:
            if self.rng.random() < 0.3:
                fpg += 20
                hba1c += 0.3

        return {
            "fasting_plasma_glucose": fpg,
            "random_plasma_glucose": rpg,
            "hba1c": hba1c,
            "fasting_insulin": fasting_insulin,
            "c_peptide": cpep,
            "triglycerides": tg,
            "hdl_c": hdl,
            "ldl_c": ldl,
            "total_cholesterol": tc,
            "serum_creatinine": cr,
            "egfr": egfr,
            "bun": bun,
            "uacr": uacr,
            "urine_albumin": ualb,
            "urine_creatinine": ucr,
            "sodium": na,
            "potassium": k,
            "bicarbonate": hco3,
            "phosphate": po4,
            "calcium": ca,
            "hemoglobin": hgb,
            "hs_crp": hs_crp,
            "crp": crp,
            "esr": esr,
            "nt_probnp": ntprobnp,
            "troponin_hs": trop,
            "apo_b": apo_b,
            "lp_a": lp_a,
            "alt": alt,
            "ast": ast,
            "alp": alp,
            "ggt": ggt,
            "total_bilirubin": tbili,
            "direct_bilirubin": dbili,
            "albumin": alb,
            "inr": inr,
            "platelet_count": plt,
            "wbc": wbc,
            "eosinophils": eos,
            "fev1_pct_predicted": fev1,
            "fvc_pct_predicted": fvc,
            "fev1_fvc_ratio": ratio,
            "spo2": spo2,
            "rf": rf,
            "anti_ccp": ccp,
            "tsh": tsh,
            "free_t4": ft4,
            "free_t3": ft3,
            "tpo_ab": tpo,
            "mmse": mmse,
            "moca": moca,
            "faq": faq,
        }

    def _order_labs(
        self,
        p: PatientLatent,
        sevs: dict[str, float],
        labs: dict[str, float],
        acute_illness: int,
        first_dx: dict[str, int],
    ) -> set[str]:
        """Realistic test-ordering: routine panels + risk-triggered + expensive rare tests."""
        ordered: set[str] = set()
        # Basic metabolic-ish panel often ordered
        routine = [
            "fasting_plasma_glucose", "serum_creatinine", "egfr", "bun", "sodium",
            "potassium", "bicarbonate", "hemoglobin", "wbc", "platelet_count",
        ]
        for a in routine:
            if self.rng.random() < 0.85:
                ordered.add(a)

        # Lipids often annual
        if self.rng.random() < 0.55:
            ordered.update(["triglycerides", "hdl_c", "ldl_c", "total_cholesterol"])

        # HbA1c if metabolic risk or known T2DM
        if sevs["t2dm"] > 0.2 or p.base_metabolic_risk > 0.3 or self.rng.random() < 0.25:
            if self.rng.random() < 0.8:
                ordered.add("hba1c")
            if self.rng.random() < 0.35:
                ordered.add("random_plasma_glucose")

        # Insulin/C-peptide rare
        if sevs["t2dm"] > 0.35 and self.rng.random() < 0.12:
            ordered.update(["fasting_insulin", "c_peptide"])

        # Kidney advanced
        if sevs["ckd"] > 0.2 or sevs["t2dm"] > 0.3 or labs.get("egfr", 100) < 75:
            if self.rng.random() < 0.7:
                ordered.update(["uacr", "urine_albumin", "urine_creatinine"])
            if sevs["ckd"] > 0.45 and self.rng.random() < 0.6:
                ordered.update(["phosphate", "calcium", "albumin"])

        # CV
        if sevs["hf"] > 0.25 or sevs["cad"] > 0.4:
            if self.rng.random() < 0.55:
                ordered.add("nt_probnp")
        if acute_illness and sevs["cad"] > 0.3 and self.rng.random() < 0.4:
            ordered.add("troponin_hs")
        if self.rng.random() < 0.15:
            ordered.add("hs_crp")
        if self.rng.random() < 0.08:
            ordered.add("apo_b")
        if self.rng.random() < 0.05:
            ordered.add("lp_a")

        # Liver
        if sevs["cld_masld"] > 0.2 or p.alcohol_use == "heavy" or p.bmi > 30 or self.rng.random() < 0.3:
            ordered.update(["alt", "ast"])
            if self.rng.random() < 0.6:
                ordered.update(["alp", "ggt", "total_bilirubin", "albumin"])
            if sevs["cld_masld"] > 0.55 and self.rng.random() < 0.5:
                ordered.update(["direct_bilirubin", "inr"])

        # Pulmonary
        if sevs["copd"] > 0.2 or p.smoking_status != "never" or self.rng.random() < 0.08:
            if self.rng.random() < 0.5:
                ordered.update(["fev1_pct_predicted", "fvc_pct_predicted", "fev1_fvc_ratio"])
            ordered.add("spo2")
            if sevs["copd"] > 0.3 and self.rng.random() < 0.4:
                ordered.add("eosinophils")

        # RA serologies — expensive / specialty
        if sevs["ra"] > 0.25 or self.rng.random() < 0.03:
            if self.rng.random() < 0.7:
                ordered.update(["rf", "anti_ccp", "crp", "esr"])

        # Thyroid
        if sevs["hypothyroid"] > 0.2 or self.rng.random() < 0.2:
            ordered.add("tsh")
            if labs.get("tsh", 2) > 4 or sevs["hypothyroid"] > 0.3:
                if self.rng.random() < 0.75:
                    ordered.add("free_t4")
                if self.rng.random() < 0.35:
                    ordered.add("free_t3")
                if self.rng.random() < 0.4:
                    ordered.add("tpo_ab")

        # Cognitive testing age/risk
        if p.age_at_index >= 60 or sevs["alzheimers"] > 0.15:
            if self.rng.random() < 0.25 + 0.5 * sevs["alzheimers"]:
                ordered.add("moca" if self.rng.random() < 0.6 else "mmse")
                if sevs["alzheimers"] > 0.35 and self.rng.random() < 0.5:
                    ordered.add("faq")

        # Access limits specialty tests
        if p.access_to_care == "limited":
            specialty = {"anti_ccp", "nt_probnp", "apo_b", "lp_a", "tpo_ab", "c_peptide", "fasting_insulin", "moca", "faq"}
            ordered = {a for a in ordered if a not in specialty or self.rng.random() < 0.25}

        return ordered

    def _diagnosis_evidence(
        self,
        disease: str,
        labs: dict[str, float],
        sev: float,
        sbp: float,
        stage: str,
        polyuria: int,
        chest_pain: int,
        dyspnea: int,
        joint_swelling: int,
        cognitive_complaint: int,
    ) -> float:
        """Probabilistic evidence score in ~[0,1]; not a clinical algorithm."""
        e = 0.15 * sev
        if disease == "t2dm":
            if labs.get("hba1c", 5) >= 6.5:
                e += 0.45
            elif labs.get("hba1c", 5) >= 5.7:
                e += 0.15
            if labs.get("fasting_plasma_glucose", 90) >= 126:
                e += 0.35
            e += 0.1 * polyuria
        elif disease == "ckd":
            if labs.get("egfr", 100) < 60:
                e += 0.45
            if labs.get("uacr", 10) >= 30:
                e += 0.25
            if labs.get("egfr", 100) < 45:
                e += 0.15
        elif disease == "htn":
            if sbp >= 140:
                e += 0.5
            elif sbp >= 130:
                e += 0.25
            e += 0.2 * sev
        elif disease == "cad":
            e += 0.35 * sev + 0.2 * chest_pain
            if labs.get("ldl_c", 100) > 160:
                e += 0.1
        elif disease == "hf":
            e += 0.3 * sev + 0.25 * dyspnea
            if labs.get("nt_probnp", 50) > 300:
                e += 0.35
        elif disease == "copd":
            if labs.get("fev1_fvc_ratio", 0.8) < 0.70:
                e += 0.5
            e += 0.25 * sev
        elif disease == "cld_masld":
            if labs.get("alt", 20) > 40:
                e += 0.3
            e += 0.35 * sev
        elif disease == "ra":
            e += 0.25 * joint_swelling + 0.2 * sev
            if labs.get("anti_ccp", 10) > 20:
                e += 0.35
            if labs.get("rf", 10) > 14:
                e += 0.2
        elif disease == "hypothyroid":
            if labs.get("tsh", 2) > 4.5:
                e += 0.45
            if labs.get("free_t4", 1.2) < 0.8:
                e += 0.25
            e += 0.15 * sev
        elif disease == "alzheimers":
            e += 0.3 * cognitive_complaint + 0.35 * sev
            if labs.get("moca", 28) < 26:
                e += 0.25
            if labs.get("mmse", 28) < 24:
                e += 0.2
        return float(np.clip(e, 0, 1))

    def _medications_for_state(
        self,
        p: PatientLatent,
        day: int,
        sevs: dict[str, float],
        first_dx: dict[str, int],
        already: set[str],
    ) -> list[dict]:
        out = []
        mapping = {
            "t2dm": ["metformin", "sglt2i", "glp1ra", "insulin"],
            "htn": ["acei_arb", "calcium_channel_blocker", "thiazide"],
            "ckd": ["acei_arb", "sglt2i"],
            "cad": ["statin", "antiplatelet", "beta_blocker"],
            "hf": ["acei_arb", "beta_blocker", "loop_diuretic", "sglt2i"],
            "copd": ["inhaled_bronchodilator", "inhaled_corticosteroid"],
            "cld_masld": [],
            "ra": ["dmard_methotrexate", "nsaid", "systemic_corticosteroid", "biologic_dmard"],
            "hypothyroid": ["levothyroxine"],
            "alzheimers": [],
        }
        for d, classes in mapping.items():
            if p.treatment_start[d] < 0 or day < p.treatment_start[d]:
                continue
            # Only start near treatment day or when dx observed
            if abs(day - p.treatment_start[d]) > 90 and first_dx[d] >= 0 and day != first_dx[d]:
                # chronic continuation not re-listed each visit; only initiation events
                continue
            # Record initiation at treatment_start (and allow same-day diagnosis intensification)
            if day != p.treatment_start[d] and first_dx[d] != day:
                continue
            for j, cls in enumerate(classes):
                key = cls + "|" + d
                if key in already:
                    continue
                # First-line almost always; intensify with severity
                if j == 0 or sevs[d] > 0.20 + 0.12 * j:
                    if j == 0 or self.rng.random() < 0.75:
                        out.append({
                            "class": cls,
                            "indication": d,
                            "adherence": round(p.adherence[d], 3),
                        })
        return out

    def _targets_for_encounter(
        self,
        p: PatientLatent,
        enc_id: str,
        day: int,
        sevs: dict[str, float],
        latent_present: dict[str, int],
        stages: dict[str, str],
        first_dx: dict[str, int],
        labs: dict[str, float],
        sbp: float,
    ) -> dict:
        """Labels with explicit prediction time and windows; features must use cutoff=day."""
        row = {
            "is_synthetic": True,
            "dataset_disclaimer": "SYNTHETIC_RESEARCH_ONLY",
            "patient_id": p.patient_id,
            "encounter_id": enc_id,
            "prediction_timestamp_day": day,
            "feature_cutoff_timestamp_day": day,
            "partition": p.partition,
            "is_external_style_holdout": int(p.is_external_style),
        }
        # Censoring: end of observation or death
        death_day = None  # filled later from patient; use obs end
        censor = p.obs_days
        row["censoring_timestamp_day"] = censor

        for d in DISEASES:
            # True latent vs observed
            row[f"true_latent_disease_present_{d}"] = latent_present[d]
            row[f"observed_diagnosis_present_{d}"] = int(first_dx[d] >= 0 and day >= first_dx[d])
            row[f"disease_stage_at_encounter_{d}"] = stages[d]
            row[f"disease_present_at_encounter_{d}"] = latent_present[d]  # modeling target (latent)

            # Future onset windows from prediction time (latent)
            for months, days_w in [(6, 182), (12, 365), (36, 1095)]:
                start = day + 1
                end = day + days_w
                row[f"outcome_window_start_{months}m"] = start
                row[f"outcome_window_end_{months}m"] = end
                # Eligible if not already diseased and not censored before window start
                already = p.onset_day[d] >= 0 and p.onset_day[d] <= day
                if already or day >= censor:
                    row[f"new_disease_within_{months}_months_{d}"] = -1  # ineligible
                else:
                    onset = p.onset_day[d]
                    if onset >= 0 and start <= onset <= min(end, censor):
                        row[f"new_disease_within_{months}_months_{d}"] = 1
                    elif censor < end and not (onset >= 0 and onset <= censor):
                        row[f"new_disease_within_{months}_months_{d}"] = -1  # censored
                    else:
                        row[f"new_disease_within_{months}_months_{d}"] = 0

            # Progression: increase in stage within 12m
            if day >= censor:
                row[f"progression_within_12_months_{d}"] = -1
            else:
                future_days = [dd for dd in p.encounter_days if day < dd <= day + 365]
                if not future_days:
                    # use latent severity path at day+365 or censor
                    fut = self.severity_at(p, d, min(day + 365, censor))
                    prog = int(fut >= sevs[d] + 0.15 and fut >= 0.35)
                    row[f"progression_within_12_months_{d}"] = prog if day + 90 <= censor else -1
                else:
                    fut = max(self.severity_at(p, d, fd) for fd in future_days)
                    row[f"progression_within_12_months_{d}"] = int(fut >= sevs[d] + 0.15)

            # Major complication proxy within 12m
            row[f"major_complication_within_12_months_{d}"] = -1
            if day + 90 <= censor and latent_present[d]:
                # Higher severity -> higher complication risk
                pr = 0.05 + 0.35 * sevs[d] ** 2
                # look ahead: use max severity in window as proxy realization
                fut = self.severity_at(p, d, min(day + 365, censor))
                row[f"major_complication_within_12_months_{d}"] = int(fut > 0.75 and self.rng.random() < pr + 0.2)

        # Hospitalization within 12m
        if day + 30 > censor:
            row["hospitalization_within_12_months"] = -1
        else:
            risk = 0.05 + 0.15 * sevs["hf"] + 0.1 * sevs["copd"] + 0.08 * sevs["ckd"] + 0.05 * sevs["cad"]
            row["hospitalization_within_12_months"] = int(self.rng.random() < risk)

        # Treatment response at next follow-up
        next_days = [dd for dd in p.encounter_days if dd > day]
        if not next_days:
            row["treatment_response_at_next_followup"] = -1
            row["biomarker_control_at_next_followup"] = -1
        else:
            nd = next_days[0]
            # Any disease on treatment improving
            resp = 0
            ctrl = 0
            for d in DISEASES:
                if p.treatment_start[d] >= 0 and p.treatment_start[d] <= day:
                    s0 = sevs[d]
                    s1 = self.severity_at(p, d, nd)
                    if s1 < s0 - 0.08:
                        resp = 1
                    # Biomarker control proxies
                    if d == "t2dm" and labs.get("hba1c", 9) < 7.0:
                        ctrl = 1
                    if d == "htn" and sbp < 130:
                        ctrl = 1
                    if d == "hypothyroid" and 0.4 <= labs.get("tsh", 10) <= 4.0:
                        ctrl = 1
            row["treatment_response_at_next_followup"] = resp
            row["biomarker_control_at_next_followup"] = ctrl

        # Mortality 5y — use patient-level later join; placeholder eligibility
        row["all_cause_mortality_within_5_years"] = -1  # filled in postprocess
        row["eligible_mortality_5y"] = int(day == 0)  # index prediction only

        return row

    # ------------------------------------------------------------------
    # Noise layer
    # ------------------------------------------------------------------
    def apply_noise(self, tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        rng = np.random.default_rng(self.seed + 999)
        noisy = {k: v.copy() for k, v in tables.items()}

        labs = noisy["labs"]
        if len(labs):
            # Measurement noise
            for idx in labs.sample(frac=0.9, random_state=self.seed).index:
                val = labs.at[idx, "value"]
                if pd.isna(val):
                    continue
                labs.at[idx, "value"] = round(float(val) * float(rng.normal(1.0, 0.02)) + float(rng.normal(0, 0.01)), 3)
            # Rounding
            for idx in labs.sample(frac=0.3, random_state=self.seed + 1).index:
                labs.at[idx, "value"] = round(float(labs.at[idx, "value"]), 0 if rng.random() < 0.3 else 1)
            # Unit variation note (keep standardized)
            mask = labs["analyte"].isin(["fasting_plasma_glucose", "serum_creatinine"]) & (rng.random(len(labs)) < 0.02)
            # Don't convert values incorrectly; flag only
            labs.loc[mask.values if hasattr(mask, "values") else mask, "assay_category"] = "alternate_assay_lot"
            # Recompute flags
            for idx, r in labs.iterrows():
                labs.at[idx, "abnormality_flag"] = abnormality_flag(r["analyte"], float(r["value"]), "female")
            # Duplicate measurements ~1%
            dup = labs.sample(frac=0.01, random_state=self.seed + 2).copy()
            dup["lab_result_id"] = [f"SYN-LABDUP{i:08d}" for i in range(len(dup))]
            labs = pd.concat([labs, dup], ignore_index=True)
            # Occasional impossible entry errors flagged
            n_err = max(1, int(0.0005 * len(labs)))
            err_idx = rng.choice(labs.index, size=min(n_err, len(labs)), replace=False)
            labs.loc[err_idx, "value"] = -1 * labs.loc[err_idx, "value"].abs()
            labs.loc[err_idx, "abnormality_flag"] = "ERROR_NEGATIVE"
            labs.loc[err_idx, "assay_category"] = "data_entry_error_flagged"
            noisy["labs"] = labs

        # Coding inconsistencies in diagnoses
        dx = noisy["diagnoses"]
        if len(dx) > 10:
            n_flip = max(1, int(0.01 * len(dx)))
            # Add a few false-positive diagnoses
            enc = noisy["encounters"]
            sample_enc = enc.sample(n=min(n_flip, len(enc)), random_state=self.seed)
            extra = []
            for j, (_, er) in enumerate(sample_enc.iterrows()):
                d = str(rng.choice(DISEASES))
                extra.append({
                    "is_synthetic": True,
                    "dataset_disclaimer": "SYNTHETIC_RESEARCH_ONLY",
                    "diagnosis_id": f"SYN-DXNOISE{j:07d}",
                    "patient_id": er["patient_id"],
                    "encounter_id": er["encounter_id"],
                    "days_from_index": er["days_from_index"],
                    "disease_code": d,
                    "disease_label": DISEASE_LABELS[d],
                    "diagnosis_type": "coding_noise_false_positive",
                    "clinical_stage_at_diagnosis": "none",
                    "latent_severity_at_diagnosis": 0.0,
                    "evidence_score": 0.0,
                    "is_true_latent_disease": 0,
                    "is_observed_diagnosis": 1,
                })
            noisy["diagnoses"] = pd.concat([dx, pd.DataFrame(extra)], ignore_index=True)

        # Encounter vital noise
        enc = noisy["encounters"]
        for col in ["sbp_mmHg", "dbp_mmHg", "heart_rate_bpm"]:
            if col in enc.columns:
                enc[col] = (enc[col].astype(float) + rng.normal(0, 2, len(enc))).round(0)
        noisy["encounters"] = enc

        return noisy

    def postprocess_targets(self, tables: dict[str, pd.DataFrame]) -> None:
        pts = tables["patients"].set_index("patient_id")
        tg = tables["prediction_targets"]
        mort = []
        for _, r in tg.iterrows():
            if r["eligible_mortality_5y"] != 1:
                mort.append(-1)
                continue
            pid = r["patient_id"]
            death = int(pts.at[pid, "death_within_5y_latent"])
            death_day = int(pts.at[pid, "death_day"])
            if death == 1 and 0 < death_day <= 1825:
                mort.append(1)
            else:
                mort.append(0)
        tg["all_cause_mortality_within_5_years"] = mort
        tables["prediction_targets"] = tg

    def build_flat_encounter_table(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Optional denormalized row-per-encounter table with stable patient attrs."""
        pts = tables["patients"].drop(
            columns=[c for c in tables["patients"].columns if c in ("is_synthetic", "dataset_disclaimer")],
            errors="ignore",
        )
        # Avoid duplicate partition from both sides
        enc = tables["encounters"]
        flat = enc.merge(pts, on="patient_id", how="left", suffixes=("", "_patient"))
        if "partition_patient" in flat.columns:
            flat = flat.drop(columns=["partition_patient"])
        # Wide selected common labs
        labs = tables["labs"]
        key_analytes = [
            "hba1c", "fasting_plasma_glucose", "egfr", "uacr", "ldl_c", "hdl_c",
            "triglycerides", "alt", "ast", "tsh", "nt_probnp", "hemoglobin",
            "fev1_fvc_ratio", "crp", "anti_ccp", "moca",
        ]
        sub = labs[labs["analyte"].isin(key_analytes)][
            ["encounter_id", "analyte", "value"]
        ].drop_duplicates(["encounter_id", "analyte"])
        if len(sub):
            wide = sub.pivot(index="encounter_id", columns="analyte", values="value")
            wide.columns = [f"lab_{c}" for c in wide.columns]
            flat = flat.merge(wide, left_on="encounter_id", right_index=True, how="left")
        # Attach core targets
        tcols = [
            c for c in tables["prediction_targets"].columns
            if c.startswith((
                "disease_present_", "new_disease_within_12_", "hospitalization_",
                "progression_within_12_", "prediction_", "feature_cutoff_", "censoring_",
            )) or c in ("encounter_id", "patient_id")
        ]
        tg = tables["prediction_targets"][tcols]
        flat = flat.merge(tg, on=["encounter_id", "patient_id"], how="left", suffixes=("", "_tgt"))
        flat["is_synthetic"] = True
        flat["dataset_disclaimer"] = "SYNTHETIC_RESEARCH_ONLY"
        # Ensure disclaimer columns lead the table
        lead = ["is_synthetic", "dataset_disclaimer"]
        other = [c for c in flat.columns if c not in lead]
        return flat[lead + other]
    def save(self, tables: dict[str, pd.DataFrame], out_dir: Path, tag: str) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        flat = self.build_flat_encounter_table(tables)
        tables = {**tables, "flat_encounter_level": flat}
        for name, df in tables.items():
            path = out_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            try:
                df.to_parquet(out_dir / f"{name}.parquet", index=False)
            except Exception:
                pass
        meta = {
            "is_synthetic": True,
            "disclaimer": (
                "SYNTHETIC DATA FOR RESEARCH, EDUCATION, AND MODEL DEVELOPMENT ONLY. "
                "Not real patient data. Not clinically validated. Not for clinical decision-making."
            ),
            "generator_name": GENERATOR_NAME,
            "generator_version": GENERATOR_VERSION,
            "random_seed": self.seed,
            "n_patients": self.n_patients,
            "version_tag": tag,
            "tables": {name: {"rows": int(len(df)), "cols": int(df.shape[1])} for name, df in tables.items()},
            "prevalence_note": (
                "Realized prevalence is a design outcome of the latent process, not a census estimate. "
                "RA and some rarer outcomes are modestly enriched for ML learnability; "
                "do not treat as population-representative."
            ),
        }
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def run(self) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        self.generate_patients()
        clean = self.build_tables()
        self.postprocess_targets(clean)
        noisy = self.apply_noise(clean)
        self.save(clean, CLEAN_DIR, "clean")
        self.save(noisy, NOISY_DIR, "noisy")
        return clean, noisy


def main(n_patients: int = DEFAULT_N_PATIENTS, seed: int = RANDOM_SEED) -> None:
    gen = SyntheticGenerator(n_patients=n_patients, seed=seed)
    clean, noisy = gen.run()
    print(f"Generated {n_patients} synthetic patients.")
    for k, df in clean.items():
        print(f"  clean/{k}: {len(df):,} rows x {df.shape[1]} cols")
    print(f"Saved to {CLEAN_DIR} and {NOISY_DIR}")


if __name__ == "__main__":
    main()
