"""Clinically conventional units and approximate adult reference intervals.

Ranges are simplified adult population intervals for synthetic data generation.
They are not assay-specific or patient-specific clinical reference ranges and
must not be used for real-world interpretation.

Sources informing design (conceptual, not copied thresholds as absolute truth):
ADA Standards of Care (glycemia), KDIGO (CKD), ACC/AHA (BP/lipids),
GOLD (COPD), AASLD/EASL (liver), ACR (RA), ATA (thyroid), NIA-AA (cognition).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LabSpec:
    analyte: str
    unit: str
    ref_low: Optional[float]
    ref_high: Optional[float]
    specimen: str
    fasting_relevant: bool
    loinc_like: str  # synthetic identifier, not a real LOINC claim
    notes: str = ""


# Conventional adult reference-style intervals used for abnormality flags.
# Sex-specific adjustments applied in generator where noted.
LAB_SPECS: dict[str, LabSpec] = {
    "fasting_plasma_glucose": LabSpec(
        "Fasting plasma glucose", "mg/dL", 70.0, 99.0, "plasma", True, "SYN-FPG",
        "ADA: normal FPG <100; prediabetes 100-125; diabetes >=126 (with criteria).",
    ),
    "random_plasma_glucose": LabSpec(
        "Random plasma glucose", "mg/dL", 70.0, 140.0, "plasma", False, "SYN-RPG",
    ),
    "hba1c": LabSpec(
        "Hemoglobin A1c", "%", 4.0, 5.6, "whole_blood", False, "SYN-A1C",
        "ADA: normal <5.7; prediabetes 5.7-6.4; diabetes >=6.5.",
    ),
    "fasting_insulin": LabSpec(
        "Fasting insulin", "uIU/mL", 2.0, 25.0, "serum", True, "SYN-INS",
    ),
    "c_peptide": LabSpec(
        "C-peptide", "ng/mL", 0.8, 3.1, "serum", True, "SYN-CPEP",
    ),
    "triglycerides": LabSpec(
        "Triglycerides", "mg/dL", 0.0, 149.0, "serum", True, "SYN-TG",
    ),
    "hdl_c": LabSpec(
        "HDL cholesterol", "mg/dL", 40.0, 100.0, "serum", True, "SYN-HDL",
        "Lower bound is risk-oriented; women often use 50 mg/dL as risk threshold.",
    ),
    "ldl_c": LabSpec(
        "LDL cholesterol", "mg/dL", 0.0, 99.0, "serum", True, "SYN-LDL",
    ),
    "total_cholesterol": LabSpec(
        "Total cholesterol", "mg/dL", 0.0, 199.0, "serum", True, "SYN-TC",
    ),
    "serum_creatinine": LabSpec(
        "Serum creatinine", "mg/dL", 0.6, 1.2, "serum", False, "SYN-CR",
        "Sex-specific typical: F ~0.5-1.1, M ~0.7-1.3 (simplified).",
    ),
    "egfr": LabSpec(
        "eGFR (CKD-EPI style)", "mL/min/1.73m2", 90.0, 120.0, "calculated", False, "SYN-EGFR",
        "KDIGO staging uses eGFR categories; lower values abnormal.",
    ),
    "bun": LabSpec(
        "Blood urea nitrogen", "mg/dL", 7.0, 20.0, "serum", False, "SYN-BUN",
    ),
    "uacr": LabSpec(
        "Urine albumin-creatinine ratio", "mg/g", 0.0, 29.0, "urine", False, "SYN-UACR",
        "KDIGO: A1 <30, A2 30-300, A3 >300.",
    ),
    "urine_albumin": LabSpec(
        "Urine albumin", "mg/dL", 0.0, 2.0, "urine", False, "SYN-UALB",
    ),
    "urine_creatinine": LabSpec(
        "Urine creatinine", "mg/dL", 20.0, 300.0, "urine", False, "SYN-UCR",
    ),
    "sodium": LabSpec("Sodium", "mmol/L", 136.0, 145.0, "serum", False, "SYN-NA"),
    "potassium": LabSpec("Potassium", "mmol/L", 3.5, 5.1, "serum", False, "SYN-K"),
    "bicarbonate": LabSpec("Bicarbonate", "mmol/L", 22.0, 29.0, "serum", False, "SYN-HCO3"),
    "phosphate": LabSpec("Phosphate", "mg/dL", 2.5, 4.5, "serum", False, "SYN-PO4"),
    "calcium": LabSpec("Calcium", "mg/dL", 8.6, 10.2, "serum", False, "SYN-CA"),
    "hemoglobin": LabSpec(
        "Hemoglobin", "g/dL", 12.0, 16.0, "whole_blood", False, "SYN-HGB",
        "Sex-specific typical ranges applied in generator.",
    ),
    "hs_crp": LabSpec(
        "High-sensitivity CRP", "mg/L", 0.0, 3.0, "serum", False, "SYN-HSCRP",
    ),
    "crp": LabSpec("C-reactive protein", "mg/L", 0.0, 10.0, "serum", False, "SYN-CRP"),
    "esr": LabSpec(
        "Erythrocyte sedimentation rate", "mm/h", 0.0, 20.0, "whole_blood", False, "SYN-ESR",
    ),
    "nt_probnp": LabSpec(
        "NT-proBNP", "pg/mL", 0.0, 125.0, "plasma", False, "SYN-NTP",
        "Age-dependent cutoffs exist; simplified adult flag used.",
    ),
    "troponin_hs": LabSpec(
        "High-sensitivity cardiac troponin", "ng/L", 0.0, 14.0, "plasma", False, "SYN-TNI",
        "Ordered mainly in acute contexts; assay cutoffs vary.",
    ),
    "apo_b": LabSpec("Apolipoprotein B", "mg/dL", 0.0, 100.0, "serum", True, "SYN-APOB"),
    "lp_a": LabSpec("Lipoprotein(a)", "nmol/L", 0.0, 75.0, "serum", False, "SYN-LPA"),
    "alt": LabSpec("ALT", "U/L", 7.0, 40.0, "serum", False, "SYN-ALT"),
    "ast": LabSpec("AST", "U/L", 10.0, 40.0, "serum", False, "SYN-AST"),
    "alp": LabSpec("Alkaline phosphatase", "U/L", 40.0, 129.0, "serum", False, "SYN-ALP"),
    "ggt": LabSpec("GGT", "U/L", 5.0, 40.0, "serum", False, "SYN-GGT"),
    "total_bilirubin": LabSpec("Total bilirubin", "mg/dL", 0.1, 1.2, "serum", False, "SYN-TBILI"),
    "direct_bilirubin": LabSpec("Direct bilirubin", "mg/dL", 0.0, 0.3, "serum", False, "SYN-DBILI"),
    "albumin": LabSpec("Albumin", "g/dL", 3.5, 5.0, "serum", False, "SYN-ALB"),
    "inr": LabSpec("INR", "ratio", 0.8, 1.2, "plasma", False, "SYN-INR"),
    "platelet_count": LabSpec("Platelet count", "10^9/L", 150.0, 400.0, "whole_blood", False, "SYN-PLT"),
    "wbc": LabSpec("White blood cell count", "10^9/L", 4.0, 11.0, "whole_blood", False, "SYN-WBC"),
    "eosinophils": LabSpec("Blood eosinophils", "cells/uL", 0.0, 500.0, "whole_blood", False, "SYN-EOS"),
    "fev1_pct_predicted": LabSpec(
        "FEV1 percent predicted", "%", 80.0, 120.0, "spirometry", False, "SYN-FEV1",
    ),
    "fvc_pct_predicted": LabSpec(
        "FVC percent predicted", "%", 80.0, 120.0, "spirometry", False, "SYN-FVC",
    ),
    "fev1_fvc_ratio": LabSpec(
        "FEV1/FVC ratio", "ratio", 0.70, 0.90, "spirometry", False, "SYN-RATIO",
        "GOLD obstruction threshold often FEV1/FVC <0.70 (post-bronchodilator).",
    ),
    "spo2": LabSpec("Resting SpO2", "%", 95.0, 100.0, "pulse_ox", False, "SYN-SPO2"),
    "rf": LabSpec("Rheumatoid factor", "IU/mL", 0.0, 14.0, "serum", False, "SYN-RF"),
    "anti_ccp": LabSpec("Anti-CCP antibodies", "U/mL", 0.0, 20.0, "serum", False, "SYN-CCP"),
    "tsh": LabSpec("TSH", "mIU/L", 0.4, 4.0, "serum", False, "SYN-TSH"),
    "free_t4": LabSpec("Free T4", "ng/dL", 0.8, 1.8, "serum", False, "SYN-FT4"),
    "free_t3": LabSpec("Free T3", "pg/mL", 2.3, 4.2, "serum", False, "SYN-FT3"),
    "tpo_ab": LabSpec("TPO antibodies", "IU/mL", 0.0, 34.0, "serum", False, "SYN-TPO"),
    "mmse": LabSpec(
        "MMSE total score", "points", 24.0, 30.0, "cognitive_scale", False, "SYN-MMSE",
        "Screening instrument; not diagnostic alone. Lower scores worse.",
    ),
    "moca": LabSpec(
        "MoCA total score", "points", 26.0, 30.0, "cognitive_scale", False, "SYN-MOCA",
        "Screening instrument; education-adjusted interpretation not fully modeled.",
    ),
    "faq": LabSpec(
        "Functional Activities Questionnaire", "points", 0.0, 8.0, "functional_scale", False, "SYN-FAQ",
        "Higher scores indicate greater functional impairment.",
    ),
}


def abnormality_flag(analyte: str, value: float, sex: str = "female") -> str:
    """Return L / H / N / NA using simplified reference bounds."""
    if analyte not in LAB_SPECS or value is None:
        return "NA"
    spec = LAB_SPECS[analyte]
    low, high = spec.ref_low, spec.ref_high

    # Sex-specific hemoglobin / creatinine soft adjustments
    if analyte == "hemoglobin":
        if sex == "male":
            low, high = 13.0, 17.0
        else:
            low, high = 12.0, 16.0
    if analyte == "serum_creatinine":
        if sex == "male":
            low, high = 0.7, 1.3
        else:
            low, high = 0.5, 1.1
    if analyte == "hdl_c" and sex == "female":
        low = 50.0

    # Cognitive scales: low is abnormal
    if analyte in {"mmse", "moca"}:
        if low is not None and value < low:
            return "L"
        return "N"
    if analyte == "faq":
        if high is not None and value > high:
            return "H"
        return "N"
    if analyte == "egfr":
        if low is not None and value < low:
            return "L"
        return "N"
    if analyte == "hdl_c":
        if low is not None and value < low:
            return "L"
        return "N"

    flags = []
    if low is not None and value < low:
        flags.append("L")
    if high is not None and value > high:
        flags.append("H")
    if not flags:
        return "N"
    return flags[0] if len(flags) == 1 else "LH"
